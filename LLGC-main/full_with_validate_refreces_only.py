import pandas as pd
import networkx as nx
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import random

from model import LLGC, PageRankAgg

# --------------------------
# 0. Set random seeds for reproducibility
# --------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# --------------------------
# 1. Load & Preprocess Data
# --------------------------
file_name = "C:\\Users\\nadir\\FinalProject\\LLGC-main\\final_filtered_by_fos_and_reference.csv"
print(f"Loading data from: {file_name}")
df = pd.read_csv(file_name)

# Convert 'references' string representation of list into actual Python list
df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# --------------------------
# 2. Build Graph
# --------------------------
G = nx.DiGraph()  # directed
#G = nx.Graph()  # undirected graph

paper_ids = set(df['id'])

# Add nodes with attributes (excluding textual/graph-specific fields)
for _, row in df.iterrows():
    paper_id = row['id']
    attributes = row.drop(['id','title','authors.name','year','fos.name','n_citation','references','abstract']).to_dict()
    G.add_node(paper_id, **attributes)

# Add edges based on citations (references)
for _, row in df.iterrows():
    citing_paper_id = row['id']
    for cited_paper_id in row['references']:
        if cited_paper_id in paper_ids:
            G.add_edge(citing_paper_id, cited_paper_id)

# --------------------------
# 3. Feature Extraction (STRUCTURE ONLY)
# --------------------------
print("Building STRUCTURE ONLY features (ignoring text/metadata)...")

# במקום לחשב TF-IDF ונתונים  מספריים, ניצור וקטורים אקראיים.
# המידע היחיד שהמודל יקבל הוא המבנה של הגרף שיעובד דרך ה-PageRankAgg.

num_nodes = len(df)
structure_dim = 128  # גודל הוקטור לכל מאמר. אפשר לשחק עם זה (64, 128, 256)

# יצירת פיצ'רים אקראיים מהתפלגות נורמלית
X_static = np.random.randn(num_nodes, structure_dim).astype(np.float32)

print(f"Final STRUCTURE-ONLY feature matrix shape: {X_static.shape}")

# # --------------------------
# # 3. Feature Extraction
# # --------------------------
# # Combine title + abstract for textual features
# df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# X_text = vectorizer.fit_transform(df['text_combined']).toarray()  # TF-IDF features
#
# # Numerical features (citations, year)
# numerical_features = df[['n_citation', 'year']].copy()
# scaler = StandardScaler()
# X_numerical = scaler.fit_transform(numerical_features)
#
# # One-hot encode field-of-study (fos.name)
# fos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# X_fos = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))
#
# # Number of authors as numerical feature
# df['num_authors'] = df['authors.name'].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) and x != '[]' else 1)
# X_authors = scaler.fit_transform(df[['num_authors']])
#
# # Combine all static features into one matrix
# X_static = np.hstack((X_text, X_numerical, X_fos, X_authors))
# print(f"Final static feature matrix shape: {X_static.shape}")
# --------------------------
# 4. Graph Matrices & Tensor Setup
# --------------------------
# Map between node IDs and integer indices
idx_to_id = {i: node_id for i, node_id in enumerate(G.nodes())}
id_to_idx = {node_id: i for i, node_id in enumerate(G.nodes())}

# Convert edges to index-based representation for adjacency matrix
edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
N = G.number_of_nodes()
rows = [u for u, v in edges]
cols = [v for u, v in edges]
data = np.ones(len(edges))

# Create sparse adjacency matrix
adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
adj_matrix_sliceable = adj_matrix.tocsr()

# Symmetric normalization of adjacency matrix
def normalize_adj_sym(adj):
    # Make adjacency symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# Convert sparse matrix to PyTorch sparse tensor
def sparse_to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# Convert static features to torch tensor
X_tensor = torch.FloatTensor(X_static)

# --------------------------
# 5. Temporal Segmentation
# --------------------------
DELTA_T = 61
years = df['year'].dropna().astype(int)
min_year = years.min()
max_year = years.max()
time_steps = list(range(min_year, max_year + DELTA_T, DELTA_T))

# Split nodes into temporal segments
temporal_segments = {}
for i in range(len(time_steps)-1):
    t_start = time_steps[i]
    t_end = time_steps[i+1]
    current_segment_df = df[df['year'] <= t_end]
    node_indices_in_segment = [id_to_idx[id] for id in current_segment_df['id'] if id in id_to_idx]
    temporal_segments[(t_start, t_end)] = node_indices_in_segment

print(f"Total segments created: {len(temporal_segments)}")

# --------------------------
# 6. Helper Functions
# --------------------------
def train_unsupervised_with_prior(model, X_features, adj_indices, prev_embeddings=None,
                                  epochs=100, lr=0.01, device='cpu', temporal_weight=0.5):
    """
    Train LLGC model in unsupervised manner with optional temporal regularization.
    - X_features: node features
    - adj_indices: adjacency indices for reconstruction loss
    - prev_embeddings: embeddings from previous time step (optional)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_features = X_features.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        Z = model(X_features)
        row, col = adj_indices
        loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()  # adjacency reconstruction loss
        if prev_embeddings is not None:
            prev_embeddings = prev_embeddings.to(device)
            common_nodes = min(Z.size(0), prev_embeddings.size(0))
            temporal_loss = (Z[:common_nodes] - prev_embeddings[:common_nodes]).pow(2).mean()
            loss += temporal_weight * temporal_loss
        loss.backward()
        optimizer.step()
    return model

def run_anomaly_detection(embeddings, contamination_rate):
    """
    Run IsolationForest to detect anomalies on embeddings.
    Returns anomaly scores and predicted labels (-1 for anomaly, 1 for normal)
    """
    clf = IsolationForest(contamination=contamination_rate, random_state=42)
    clf.fit(embeddings)
    anomalies = clf.predict(embeddings)
    scores = clf.decision_function(embeddings)
    return scores, anomalies

import copy

def inject_synthetic_nodes_from_csv(df, fakes_csv="C:\\Users\\nadir\\FinalProject\\LLGC-main\\fakes.csv"):
    """
    Inject synthetic rows into the dataframe from an existing CSV file.
    Assumes fakes_csv has the same columns as df and includes 'is_synthetic' column.
    """
    df = df.copy()

    # Load synthetic nodes from CSV
    df_fake = pd.read_csv(fakes_csv)

    # Ensure 'is_synthetic' column exists
    if 'is_synthetic' not in df_fake.columns:
        df_fake['is_synthetic'] = True

    # Merge with original dataframe
    df_final = pd.concat([df, df_fake], ignore_index=True)

    print(f"[Injection] Synthetic nodes from {fakes_csv} successfully added.")
    print(f"Final dataframe shape: {df_final.shape}")  # number of rows and columns

    # Save to new CSV
    df_final.to_csv("augmented_dataset.csv", index=False)
    print(df_final.tail())  # prints the last 5 rows by default

    return df_final

# --------------------------
# 7. Parameters & Setup
# --------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K_PROP = 5
ALPHA = 0.1
DROP_OUT = 0.0
USE_BIAS = 1
EPOCHS = 100
LR = 0.01
EMBEDDING_DIM = 256

X_tensor = X_tensor.to(DEVICE)
all_results_rows = []
prev_Z_t = None

print("\n" + "="*70)
print(f"Starting Real Temporal Execution (No Synthetic Injection)")
print("="*70)

# --------------------------
# 8. MAIN LOOP (REAL DATA ONLY)
# --------------------------
for (t_start, t_end), global_indices in temporal_segments.items():
    if not global_indices:
        continue

    X_t = X_tensor[global_indices]
    global_indices_np = np.array(global_indices)
    adj_segment = adj_matrix_sliceable[global_indices_np[:, None], global_indices_np]

    # Normalize adjacency and convert to torch sparse tensor
    adj_norm_real = normalize_adj_sym(adj_segment)
    adj_tensor_real = sparse_to_torch_sparse(adj_norm_real).to(DEVICE)

    # Apply PageRank-based aggregation (SGConv)
    sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
    X_gconv_real, _ = sgconv(X_t, adj_tensor_real._indices(), adj_tensor_real._values())

    # Train LLGC model with optional temporal prior
    model_real = LLGC(X_gconv_real.size(1), EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
    model_real = train_unsupervised_with_prior(
        model_real, X_gconv_real, adj_tensor_real._indices(),
        prev_embeddings=prev_Z_t, epochs=EPOCHS, lr=LR, device=DEVICE
    )

    # Evaluate embeddings and store for next time step
    model_real.eval()
    Z_t_real = model_real(X_gconv_real).cpu().detach()
    prev_Z_t = Z_t_real.clone()

    # Run anomaly detection
    scores_real, pred_real = run_anomaly_detection(Z_t_real.numpy(), contamination_rate=0.01)

    # Save results
    for idx, paper_idx in enumerate(global_indices):
        all_results_rows.append({
            'paper_id': idx_to_id[paper_idx],
            't_start': t_start, 't_end': t_end,
            'anomaly_score': scores_real[idx], 'prediction': pred_real[idx],
            'is_synthetic': False
        })

    print(f"Segment [{t_start}-{t_end}] completed (Real Only)")

# ---------------------------------------------------------
# 8.5 Remove Real Anomalies & Recompute Embeddings
# ---------------------------------------------------------
print("\n" + "="*70)
print("Cleaning real anomalies before injection")
print("="*70)


results_df = pd.DataFrame(all_results_rows)

# Only real nodes
real_results = results_df[results_df["is_synthetic"] == False]

# Nodes flagged as anomalies at least once
anomalous_real_ids = real_results[
    real_results["prediction"] == -1
]["paper_id"].unique()

print(f"Detected {len(anomalous_real_ids)} anomalous real nodes")
df_clean = df[~df["id"].isin(anomalous_real_ids)].copy()
print(f"Clean dataset size: {len(df_clean)} (was {len(df)})")
G_clean = nx.DiGraph()

paper_ids_clean = set(df_clean["id"])

for pid in paper_ids_clean:
    G_clean.add_node(pid)

for _, row in df_clean.iterrows():
    for ref in row["references"]:
        if ref in paper_ids_clean:
            G_clean.add_edge(row["id"], ref)
id_to_idx_clean = {pid: i for i, pid in enumerate(G_clean.nodes())}
idx_to_id_clean = {i: pid for pid, i in id_to_idx_clean.items()}
clean_indices = [id_to_idx[pid] for pid in df_clean["id"]]
X_clean = X_tensor[clean_indices]
edges_clean = [(id_to_idx_clean[u], id_to_idx_clean[v]) for u, v in G_clean.edges()]
rows, cols = zip(*edges_clean) if edges_clean else ([], [])

adj_clean = sp.coo_matrix(
    (np.ones(len(rows)), (rows, cols)),
    shape=(len(df_clean), len(df_clean)),
    dtype=np.float32
)

adj_norm_clean = normalize_adj_sym(adj_clean)
adj_tensor_clean = sparse_to_torch_sparse(adj_norm_clean).to(DEVICE)
sgconv_clean = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv_clean, _ = sgconv_clean(
    X_clean,
    adj_tensor_clean._indices(),
    adj_tensor_clean._values()
)

model_clean = LLGC(
    X_gconv_clean.size(1),
    EMBEDDING_DIM,
    DROP_OUT,
    USE_BIAS
).to(DEVICE)

model_clean = train_unsupervised_with_prior(
    model_clean,
    X_gconv_clean,
    adj_tensor_clean._indices(),
    epochs=EPOCHS,
    lr=LR,
    device=DEVICE
)

model_clean.eval()
Z_clean = model_clean(X_gconv_clean).detach().cpu()
# From now on — work ONLY with cleaned graph
df = df_clean
G = G_clean
X_tensor = X_clean
model_real = model_clean

print("✔ Clean model ready. Proceeding to fake injection.")

# ---------------------------------------------------------
# 9. Inject NEW Synthetic Nodes (Feature + Edge Injection)
# ---------------------------------------------------------
print("\n" + "="*70)
print("Injecting Synthetic Nodes (5% new, 2-7 random connections each)")
print("="*70)

df = inject_synthetic_nodes_from_csv(df, fakes_csv="C:\\Users\\nadir\\FinalProject\\LLGC-main\\fakes.csv")
print(f"New dataframe size after injection: {len(df)} rows")

# ---------------------------------------------------------
# 10. Detect Anomalies on Injected Synthetic Nodes
# ---------------------------------------------------------
print("\n" + "="*70)
print("Running Detection on Injected Synthetic Nodes")
print("="*70)

# Map paper_id → new index
new_idx_map = {pid: i for i, pid in enumerate(df['id'])}
N = len(df)

# Build new feature matrix (TF-IDF only here for simplicity)
texts = df['abstract'].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer(max_features=X_tensor.shape[1])
X_full_new = torch.tensor(vectorizer.fit_transform(texts).todense(), dtype=torch.float32).to(DEVICE)

# Build adjacency from references (bi-directional)
rows, cols = [], []
for i, refs in enumerate(df['references']):
    if isinstance(refs, list):
        for ref in refs:
            if ref in new_idx_map:
                rows += [i, new_idx_map[ref]]
                cols += [new_idx_map[ref], i]

adj_new = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(N, N), dtype=np.float32)

# Normalize adjacency
adj_norm_new = normalize_adj_sym(adj_new)
adj_tensor_new = sparse_to_torch_sparse(adj_norm_new).to(DEVICE)

# Run GCN aggregation
sgconv_new = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv_new, _ = sgconv_new(X_full_new, adj_tensor_new._indices(), adj_tensor_new._values())

model_real.eval()  # do NOT retrain

with torch.no_grad():
    Z_new = model_real(X_gconv_new).cpu().numpy()

# Identify synthetic nodes
fake_mask = df['is_synthetic'].fillna(False).astype(bool).values
fake_indices = np.where(fake_mask)[0]

# Run anomaly detection
contamination = 0.01
scores_new, pred_new = run_anomaly_detection(Z_new, contamination_rate=contamination)

# Save results for synthetic nodes
detected_count = 0
for idx in fake_indices:
    if pred_new[idx] == -1:
        detected_count += 1

    all_results_rows.append({
        'paper_id': df.iloc[idx]['id'],
        't_start': "POST_INJECTION",
        't_end': "POST_INJECTION",
        'anomaly_score': scores_new[idx],
        'prediction': pred_new[idx],
        'is_synthetic': True
    })

print(f"\n🔥 Injected Fake Detection Results: {detected_count}/{len(fake_indices)} "
      f"({detected_count/len(fake_indices):.1%})")

# --------------------------
# 11. Save Results
# --------------------------
out_df = pd.DataFrame(all_results_rows)
out_df.to_csv("temporal_anomaly_results_real_only.csv", index=False)
print("\n✅ CSV saved: temporal_anomaly_results_real_only.csv")


