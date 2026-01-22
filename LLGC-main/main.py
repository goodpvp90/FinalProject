
def run_pipeline(dataset_path, progress_cb=None):
    def report(msg):
        if progress_cb is not None:
            progress_cb(msg)
        print(msg)

    import pandas as pd
    import networkx as nx
    import ast
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import scipy.sparse as sp
    import random
    import os
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

    # Import your custom modules
    from model import LLGC, PageRankAgg

    # --------------------------
    # 0. System & Seeds Setup
    # --------------------------
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------
    # 1. Load Data
    # --------------------------
    file_name = "final_filtered_by_fos_and_reference.csv"
    report(f"Loading data from: {file_name}")
    df = pd.read_csv(file_name)
    df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    df['n_citation_clipped'] = df['n_citation'].clip(upper=500)

    # --------------------------
    # 2. Graph Construction
    # --------------------------
    G = nx.DiGraph()
    paper_ids = set(df['id'])
    for pid in paper_ids: G.add_node(pid)
    for _, row in df.iterrows():
        for cited_id in row['references']:
            if cited_id in paper_ids: G.add_edge(row['id'], cited_id)

    id_to_idx_global = {node_id: i for i, node_id in enumerate(G.nodes())}
    idx_to_id_global = {i: node_id for i, node_id in enumerate(G.nodes())}

    # --------------------------
    # 3. Temporal Setup
    # --------------------------
    DELTA_T = 5
    years = df['year'].dropna().astype(int)
    time_steps = list(range(years.min(), years.max() + DELTA_T, DELTA_T))

    # --------------------------
    # 4. Training Function
    # --------------------------
    def train_unsupervised_with_prior(model, X_features, adj_indices, global_indices,
                                      embedding_registry=None, epochs=100, lr=0.01):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            Z = model(X_features)

            row, col = adj_indices
            loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()

            # Variance Regularization
            std_loss = torch.mean(torch.relu(1.0 - torch.std(Z, dim=0)))
            loss += 0.1 * std_loss

            # Correct Temporal Loss
            if embedding_registry:
                current_ids = [idx_to_id_global[gi] for gi in global_indices]
                past_embs = []
                curr_embs = []
                for i, pid in enumerate(current_ids):
                    if pid in embedding_registry:
                        past_embs.append(embedding_registry[pid])
                        curr_embs.append(Z[i])

                if past_embs:
                    t_loss = (torch.stack(curr_embs) - torch.stack(past_embs).to(DEVICE)).pow(2).mean()
                    loss += 0.5 * t_loss

            loss.backward()
            optimizer.step()
        return model

    # --------------------------
    # 5. Main Loop
    # --------------------------
    K_PROP, ALPHA, EMBEDDING_DIM = 5, 0.8, 128
    embedding_registry = {}
    all_results_rows = []

    report("\nStarting Temporal Execution...")

    def quick_norm(adj):
        adj = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).dot(d_mat).tocoo()

    for i in range(len(time_steps)-1):
        t_start, t_end = time_steps[i], time_steps[i+1]

        past_df = df[df['year'] <= t_end].copy()
        current_ids = past_df['id'].tolist()
        global_indices = [id_to_idx_global[pid] for pid in current_ids]

        if not global_indices: continue

        vec = TfidfVectorizer(stop_words='english', max_features=512).fit(past_df['text_combined'])
        scal = StandardScaler().fit(past_df[['n_citation_clipped', 'year']])
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(past_df[['fos.name']].fillna('Unknown'))

        X_stack = np.hstack([
            vec.transform(past_df['text_combined']).toarray(),
            scal.transform(past_df[['n_citation_clipped', 'year']]),
            enc.transform(past_df[['fos.name']].fillna('Unknown'))
        ])
        X_t = torch.FloatTensor(X_stack).to(DEVICE)

        adj_matrix = nx.adjacency_matrix(G, nodelist=[idx_to_id_global[gi] for gi in global_indices])
        adj_norm = quick_norm(adj_matrix)

        # FIX: Ensure indices are int64 (Long)
        indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64)).to(DEVICE)
        values = torch.from_numpy(adj_norm.data.astype(np.float32)).to(DEVICE)

        sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
        X_gconv, _ = sgconv(X_t, indices, values)

        model = LLGC(X_gconv.size(1), EMBEDDING_DIM, 0.0, 1).to(DEVICE)
        model = train_unsupervised_with_prior(model, X_gconv, indices, global_indices, embedding_registry, epochs=50)

        model.eval()
        with torch.no_grad():
            Z_t = model(X_gconv).cpu()

        # --- ADDED: Record anomaly scores for this temporal segment ---
        clf_temp = IsolationForest(contamination=0.01, random_state=42)
        pred_temp = clf_temp.fit_predict(Z_t.numpy())
        scores_temp = clf_temp.decision_function(Z_t.numpy())

        for idx, pid in enumerate(current_ids):
            all_results_rows.append({
                "paper_id": pid,
                "t_start": t_start,
                "t_end": t_end,
                "anomaly_score": scores_temp[idx],
                "prediction": pred_temp[idx],
                "is_synthetic": False
            })
        # ---------------------------------------------------------------

        for idx, pid in enumerate(current_ids):
            embedding_registry[pid] = Z_t[idx]

        report(f"Segment up to {t_end} processed.")

    # ---------------------------------------------------------
    # 6. Injection & Final Detection (FIXED Runtime Error)
    # ---------------------------------------------------------
    report("\nInjecting Synthetic Nodes...")
    df_fake = pd.read_csv("fakes.csv")
    df_fake['is_synthetic'] = True
    df_aug = pd.concat([df, df_fake], ignore_index=True)
    df_aug['text_combined'] = df_aug['title'].fillna('') + ' ' + df_aug['abstract'].fillna('')
    df_aug['n_citation_clipped'] = df_aug['n_citation'].clip(upper=500)

    X_final_stack = np.hstack([
        vec.transform(df_aug['text_combined']).toarray(),
        scal.transform(df_aug[['n_citation_clipped', 'year']]),
        enc.transform(df_aug[['fos.name']].fillna('Unknown'))
    ])
    X_final = torch.FloatTensor(X_final_stack).to(DEVICE)

    aug_id_to_idx = {pid: i for i, pid in enumerate(df_aug['id'])}
    r, c = [], []
    for i, row in df_aug.iterrows():
        for ref in row['references']:
            if ref in aug_id_to_idx:
                r += [i, aug_id_to_idx[ref]]
                c += [aug_id_to_idx[ref], i]

    adj_aug_raw = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(len(df_aug), len(df_aug)))
    adj_aug = quick_norm(adj_aug_raw)

    # --- FIX: explicitly cast indices to int64 (Long) to resolve RuntimeError ---
    indices_aug = torch.from_numpy(np.vstack((adj_aug.row, adj_aug.col)).astype(np.int64)).to(DEVICE)
    values_aug = torch.from_numpy(adj_aug.data.astype(np.float32)).to(DEVICE)

    X_gconv_aug, _ = sgconv(X_final, indices_aug, values_aug)

    model.eval()
    with torch.no_grad():
        Z_final = model(X_gconv_aug).cpu().numpy()

    contamination = 0.024
    clf = IsolationForest(contamination=contamination, random_state=42)
    pred = clf.fit_predict(Z_final)

    # --- ADDED: Record final scores after injection ---
    scores_final = clf.decision_function(Z_final)
    # --------------------------
    # 7. Evaluation Metrics
    # --------------------------

    # Ground truth: synthetic = anomaly
    y_true = df_aug['is_synthetic'].fillna(False).astype(int).values

    # Model output: IsolationForest (-1 = anomaly, 1 = normal)
    y_pred = np.where(pred == -1, 1, 0)

    # ---- Standard Metrics (UNCHANGED) ----
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    report("\n📊 Final Detection Metrics (POST_INJECTION)")
    report(f"Precision: {precision:.4f}")
    report(f"Recall:    {recall:.4f}")
    report(f"F1-score:  {f1:.4f}")

    # Save the final results to a CSV
    pd.DataFrame(all_results_rows).to_csv("temporal_anomaly_results.csv", index=False)
    report("\n✅ Results saved to: temporal_anomaly_results.csv")

    # --- TSNE Visualization (2D) ---
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        Z_2d = tsne.fit_transform(Z_final)
        normal_mask = (pred == 1)
        anomaly_mask = (pred == -1)
        plt.figure(figsize=(8, 6))
        plt.scatter(Z_2d[normal_mask, 0], Z_2d[normal_mask, 1], s=10, c='blue', label='Normal', alpha=0.5)
        plt.scatter(Z_2d[anomaly_mask, 0], Z_2d[anomaly_mask, 1], s=20, c='red', label='Anomaly', alpha=0.7)
        plt.title('t-SNE of Paper Embeddings (Normal vs Anomaly)')
        plt.xlabel('TSNE-1')
        plt.ylabel('TSNE-2')
        plt.legend()
        plt.tight_layout()
        plt.savefig('tsne_papers.png', dpi=150)
        plt.show()
        report("\n🖼️ t-SNE plot saved as tsne_papers.png")
    except Exception as e:
        report(f"[TSNE Plot Error] {e}")

    fake_indices = df_aug[df_aug['is_synthetic'] == True].index
    detected = sum(1 for idx in fake_indices if pred[idx] == -1)
    report(f"\n🔥 Final Result: {detected}/{len(fake_indices)} ({detected/len(fake_indices):.1%})")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve
    import pandas as pd
    import numpy as np

    sns.set(style="whitegrid")

    # -------------------------------
    # 1. Anomaly Score Distribution
    # -------------------------------
    real_scores = scores_final[y_true == 0]
    fake_scores = scores_final[y_true == 1]

    plt.figure(figsize=(8, 5))
    sns.histplot(real_scores, bins=50, kde=True, color="blue", label="Real", stat="density", alpha=0.5)
    sns.histplot(fake_scores, bins=50, kde=True, color="red", label="Synthetic", stat="density", alpha=0.6)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("IsolationForest Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("anomaly_score_distribution.png", dpi=150)
    plt.show()

    results_df = pd.DataFrame(all_results_rows)

    # Create safe lookup from final augmented dataframe
    synthetic_lookup = dict(zip(df_aug["id"], df_aug["is_synthetic"]))

    # Add is_synthetic column safely
    results_df["is_synthetic"] = results_df["paper_id"].map(synthetic_lookup).fillna(False)

    # ------------------------------------------------
    # 6. Citation Count vs Anomaly Score (Sanity Check)
    # ------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df_aug["n_citation_clipped"],
        scores_final,
        c=y_true,
        cmap="coolwarm",
        alpha=0.6
    )
    plt.xlabel("Citation Count (Clipped)")
    plt.ylabel("Anomaly Score")
    plt.title("Citation Count vs Anomaly Score")
    plt.colorbar(label="Synthetic (1) / Real (0)")
    plt.tight_layout()
    plt.savefig("citation_vs_anomaly.png", dpi=150)
    plt.show()


    # ------------------------------------------
    # 7. Confusion Matrix
    # ------------------------------------------
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    return {
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "figures": [
            "tsne_papers.png",
            "anomaly_score_distribution.png",
            "citation_vs_anomaly.png",
            "confusion_matrix.png"
        ]
    }

if __name__ == "__main__":
    run_pipeline("final_filtered_by_fos_and_reference.csv")
