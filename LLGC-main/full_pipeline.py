"""
run_full_project.py (Version 8 - Print Organization Fix)

- Fixes the temporal vector printing issue: Calculates the chronological offset (gap)
  to correctly align the scores (and N/A padding) with the full list of processed years.
- Renames columns to "Pub Year" and "Score Year" for clarity.
"""

import pandas as pd
import numpy as np
import torch
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import scipy.sparse as sp
import ast
import random

# --- L²GC Imports (from LLGC-main repo) ---
try:
    from model import PageRankAgg, LLGC
    import manifolds
except ImportError as e:
    # MOCK CLASSES for environment compatibility
    class PageRankAgg(torch.nn.Module):
        def __init__(self, K, alpha, **kwargs): super().__init__(); 
        def forward(self, features, edge_index): return features, None
    class LLGC(torch.nn.Module):
        def __init__(self, nfeat, nclass, **kwargs): 
            super().__init__(); 
            self.linear = torch.nn.Linear(nfeat, nclass)
        def forward(self, x): 
            return self.linear(x)
    print("WARNING: GNN models mocked. Ensure original LLGC library is accessible for real results.")

# ===================================================================
#                      CONFIGURATION SETTINGS
# ===================================================================
# --- Project Hyperparameters (from Word Doc) ---
FILE_PATH = "C:\\Users\\nadir\\OneDrive\\Desktop\\final_filtered_by_fos_and_reference.csv"
EMBEDDING_DIM = 64 # Output dimension for L2GC
PAGERANK_ALPHA = 0.15  # Alpha for PageRankAgg
PAGERANK_K = 10  # K (degree) for PageRankAgg

# --- Validation Hyperparameters (Section 5.1/New Logic) ---
INJECTION_PERCENT = 0.05 # "5-10% of the nodes"
CONNECTION_PERCENT = 0.30  # "connect... to approximately 30%"
INJECTION_YEAR = 2025       # Year for the newly injected papers
INJECTION_ID_PREFIX = "NEW_FAKE_PAPER_"

# --- Deterministic Seed ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Running in deterministic mode with SEED = {SEED}")
# ===================================================================

# --- System Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def load_citation_data(filepath):
    """
    Implements Section 4.2.1: Data Preparation.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None, None, None, None, None

    # Handle NaNs and preprocess
    df['abstract'] = df['abstract'].fillna('')
    df['title'] = df['title'].fillna('')
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    
    df['id'] = df['id'].astype(str)
    df = df.set_index('id', drop=False)
    node_ids = df.index.tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    # --- Feature Engineering (Section 4.2.1.1) ---
    print("Generating TF-IDF features (Section 4.2.1.1)...")
    df['text'] = df['title'] + ' ' + df['abstract']
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    features = torch.FloatTensor(tfidf_matrix.toarray()).to(DEVICE)
    feature_dim = features.shape[1]

    # --- Graph Construction (Section 4.1) ---
    print("Building full citation graph (Section 4.1)...")
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(row['id'], year=row['year'])

    for idx, row in df.iterrows():
        try:
            references = ast.literal_eval(row['references'])
            if isinstance(references, list):
                for ref_id in references:
                    ref_id = str(ref_id)
                    if ref_id in node_to_idx and row['id'] in node_to_idx:
                        G.add_edge(row['id'], ref_id)
        except (ValueError, SyntaxError, TypeError):
            pass
            
    print(f"Data loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Feature matrix shape: {features.shape}")
    
    return df, G, features, node_to_idx, feature_dim


def inject_new_anomalous_papers(full_G, features, node_to_idx, df, injection_percent, connection_percent, feature_dim):
    """
    V7 Fix: Creates new papers with random noise features and anomalous connections.
    """
    print(f"--- INJECTING NEW ANOMALOUS PAPERS (NEW VALIDATION STRATEGY) ---")
    
    G_augmented = full_G.copy()
    all_nodes_original = list(full_G.nodes())
    
    num_to_inject = int(len(all_nodes_original) * injection_percent)
    
    new_paper_data = []
    new_features_list = []
    ground_truth_anomalies = set()
    
    for i in range(num_to_inject):
        new_id = f"{INJECTION_ID_PREFIX}{i:03d}"
        ground_truth_anomalies.add(new_id)
        
        new_paper_data.append({
            'id': new_id,
            'title': f"Synthetic Anomalous Paper {i:03d} (100% Fake - Random Feature Outlier)",
            'abstract': f"This paper is a synthetic anomaly used for validation and has random features.",
            'year': INJECTION_YEAR,
            'text': '' 
        })
        
        # --- V7 FIX: Assign random features between -1 and 1 ---
        random_features = torch.FloatTensor(np.random.uniform(low=-1.0, high=1.0, size=(1, feature_dim))).to(DEVICE)
        new_features_list.append(random_features)
        
        G_augmented.add_node(new_id, year=INJECTION_YEAR)
        node_to_idx[new_id] = features.shape[0] + i

    if new_features_list:
        new_features_tensor = torch.cat(new_features_list, dim=0)
        features_augmented = torch.cat((features, new_features_tensor), dim=0)
    else:
        features_augmented = features
        
    # 4. Add anomalous edges
    num_to_connect = int(len(all_nodes_original) * connection_percent)
    
    edges_added = 0
    for anomalous_node in ground_truth_anomalies:
        target_nodes = random.sample(all_nodes_original, num_to_connect)
        for target_node in target_nodes:
            if not G_augmented.has_edge(anomalous_node, target_node):
                G_augmented.add_edge(anomalous_node, target_node)
                edges_added += 1
                
    df_augmented = pd.concat([df, pd.DataFrame(new_paper_data).set_index('id', drop=False)])
                
    print(f"Injected {len(ground_truth_anomalies)} new nodes (Year: {INJECTION_YEAR}) and added {edges_added} synthetic edges.")
    print(f"Augmented Feature matrix shape: {features_augmented.shape}")

    return df_augmented, G_augmented, features_augmented, node_to_idx, ground_truth_anomalies


def get_snapshot_data(full_G, full_features, node_to_idx, year):
    """
    Implements Section 4.2.2: Temporal Segmentation.
    """
    
    nodes_in_snapshot = [
        node for node, data in full_G.nodes(data=True) 
        if data.get('year', float('inf')) <= year
    ]
    
    if len(nodes_in_snapshot) < 2:
        return None, None, None, None

    G_t = full_G.subgraph(nodes_in_snapshot)
    
    snapshot_indices = [node_to_idx[node_id] for node_id in G_t.nodes()]
    features_t = full_features[snapshot_indices, :]

    adj_t_scipy = nx.adjacency_matrix(G_t, nodelist=G_t.nodes())
    adj_t_coo = adj_t_scipy.tocoo()
    
    edge_index_t = torch.tensor(
        np.vstack((adj_t_coo.row, adj_t_coo.col)), 
        dtype=torch.long
    ).to(DEVICE)
    
    return G_t, features_t, edge_index_t, snapshot_indices


def run_temporal_pipeline(G_to_process, features, node_to_idx, min_year, max_year):
    """
    Runs the full L2GC + Isolation Forest pipeline over all temporal snapshots.
    """
    
    euclidean_prop_model = PageRankAgg(K=PAGERANK_K, alpha=PAGERANK_ALPHA, 
                                       add_self_loops=True, normalize=True)
    euclidean_prop_model.to(DEVICE)
    euclidean_prop_model.eval()
    
    all_anomaly_scores = {}
    list_of_processed_years = []

    for year in range(min_year, max_year + 1):
        print(f"\n--- Processing Snapshot for Year: {year} ---")
        
        G_t, features_t, edge_index_t, snapshot_indices = get_snapshot_data(
            G_to_process, features, node_to_idx, year
        )
        
        if G_t is None or G_t.number_of_nodes() < 10 or G_t.number_of_edges() == 0:
            print(f"Skipping year {year}, not enough nodes/edges.")
            continue
            
        list_of_processed_years.append(year)
            
        print(f"Snapshot G_{year}: {G_t.number_of_nodes()} nodes, {G_t.number_of_edges()} edges.")

        with torch.no_grad():
            propagated_features, _ = euclidean_prop_model(features_t, edge_index_t)
        
        n_features = propagated_features.shape[1]
        
        llgc_model = LLGC(
            nfeat=n_features,
            nclass=EMBEDDING_DIM,
            drop_out=0.0,
            use_bias=True
        ).to(DEVICE)
        llgc_model.eval()
        
        with torch.no_grad():
            euclidean_embeddings_zt = llgc_model(propagated_features) 
            
        euclidean_embeddings_zt = euclidean_embeddings_zt.cpu().numpy()

        print(f"Running Isolation Forest on {euclidean_embeddings_zt.shape[0]} nodes...")
        
        clf = IsolationForest(random_state=SEED, n_jobs=-1) 
        clf.fit(euclidean_embeddings_zt)
        
        snapshot_scores = -clf.score_samples(euclidean_embeddings_zt) 
        
        node_ids_t = list(G_t.nodes())
        for i, node_id in enumerate(node_ids_t):
            if node_id not in all_anomaly_scores:
                all_anomaly_scores[node_id] = []
            all_anomaly_scores[node_id].append(snapshot_scores[i])
            
    return all_anomaly_scores, list_of_processed_years


def aggregate_scores(all_anomaly_scores, strategy='max'):
    final_anomaly_scores = {}
    for node_id, scores in all_anomaly_scores.items():
        if scores:
            if strategy == 'max':
                final_anomaly_scores[node_id] = np.max(scores) 
            elif strategy == 'mean':
                final_anomaly_scores[node_id] = np.mean(scores)
                
    return final_anomaly_scores


def calculate_validation_metrics(final_anomaly_scores, ground_truth_set):
    
    k = len(ground_truth_set)
    
    sorted_nodes = sorted(
        final_anomaly_scores.items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    
    predicted_anomalies_set = set([node_id for node_id, score in sorted_nodes[:k]])
    
    true_positives_set = ground_truth_set.intersection(predicted_anomalies_set)
    
    TP = len(true_positives_set)
    FP = len(predicted_anomalies_set - ground_truth_set)
    FN = len(ground_truth_set - predicted_anomalies_set)
    
    print(f"Evaluation Details: TP={TP}, FP={FP}, FN={FN}")
    
    epsilon = 1e-10
    
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return precision, recall, f1_score, predicted_anomalies_set, sorted_nodes


def analyze_validation_results(predicted_anomalies_set, ground_truth_set, final_anomaly_scores, df, top_n_fp=10):
    print("\n" + "="*80)
    print("--- DETAILED VALIDATION ANALYSIS: INJECTED vs. DETECTED (SECTION 5.1) ---")
    
    TP_set = predicted_anomalies_set.intersection(ground_truth_set)
    FN_set = ground_truth_set - predicted_anomalies_set
    FP_set = predicted_anomalies_set - ground_truth_set
    
    # 1. True Positives (Detected Injected)
    print(f"\n[DETECTED INJECTED ANOMALIES - TRUE POSITIVES: {len(TP_set)}/{len(ground_truth_set)}]")
    for node_id in sorted(list(TP_set), key=lambda x: final_anomaly_scores.get(x, 0), reverse=True):
        title = df.loc[node_id]['title'] if node_id in df.index else "Title N/A"
        print(f"  [TP] ID: {node_id} | Score: {final_anomaly_scores[node_id]:.4f} | Title: {title[:70]}...")

    # 2. False Negatives (Undetected Injected)
    print(f"\n[UNDETECTED INJECTED ANOMALIES - FALSE NEGATIVES: {len(FN_set)}]")
    for node_id in sorted(list(FN_set), key=lambda x: final_anomaly_scores.get(x, 0), reverse=True):
        title = df.loc[node_id]['title'] if node_id in df.index else "Title N/A"
        print(f"  [FN] ID: {node_id} | Score: {final_anomaly_scores[node_id]:.4f} | Title: {title[:70]}...")

    # 3. False Positives (Detected Normal - Top {top_n_fp})
    sorted_fp = sorted(list(FP_set), key=lambda x: final_anomaly_scores.get(x, 0), reverse=True)
    print(f"\n[DETECTED NORMAL PAPERS (FALSE POSITIVES) - TOP {min(top_n_fp, len(sorted_fp))}/{len(FP_set)}]")
    for node_id in sorted_fp[:top_n_fp]:
        title = df.loc[node_id]['title'] if node_id in df.index else "Title N/A"
        print(f"  [FP] ID: {node_id} | Score: {final_anomaly_scores[node_id]:.4f} | Title: {title[:70]}...")
    print("="*80)


def print_temporal_vectors(all_anomaly_scores, processed_years, df, top_n=10):
    
    final_anomaly_scores = aggregate_scores(all_anomaly_scores, strategy='max')
    sorted_anomalies = sorted(
        final_anomaly_scores.items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    
    # Calculate the string showing all processed years for the header
    year_list_str = ", ".join([str(y) for y in processed_years])

    print("\n" + "="*150)
    print(f"--- TEMPORAL ANOMALY VECTORS FOR TOP {top_n} DETECTED PAPERS (SECTION 4.4) ---")
    print(f"\nYears Processed ({len(processed_years)} snapshots): {year_list_str}")
    print("-" * 150)
    
    # --- NEW, CLEANED HEADER ---
    print(f"| {'Rank':<4} | {'Max Score':<11} | {'Pub Year':<8} | {'Score Year':<10} | {'Node ID (Title)':<50} | {'Anomaly Scores (Aligned Chronologically)':<60} |")
    print("-" * 150)
    
    for i, (node_id, max_score) in enumerate(sorted_anomalies[:top_n]):
        if node_id not in all_anomaly_scores:
            continue
            
        temporal_vector = all_anomaly_scores[node_id]
        
        node_pub_year = df.loc[node_id]['year'] if node_id in df.index else processed_years[0]
        
        # --- FIX: Robust Logic for Gap Calculation ---
        # 1. The gap size is the number of processed years before scores started.
        gap_size = len(processed_years) - len(temporal_vector)
        
        # 2. Calculate Max Score Year
        if len(temporal_vector) > 0:
            max_score_index = np.argmax(temporal_vector)
            # The actual year of the max score is at processed_years[gap_size + max_score_index]
            score_year_of_max = processed_years[gap_size + max_score_index] 
        else:
            score_year_of_max = "N/A"

        # 3. Create the full temporal vector string with N/A padding at the start.
        n_a_padding = ["N/A" for _ in range(gap_size)]
        score_strings = [f"{s:.3f}" for s in temporal_vector]
        full_temporal_vector_strings = n_a_padding + score_strings
        
        # Format the score string with clean spacing
        temporal_str = " | ".join([f"{s:>5}" for s in full_temporal_vector_strings]) 
        
        try:
            paper_title = df.loc[node_id]['title']
            title_line = f"{node_id} ({paper_title[:40].strip()}...)"
            
            print(f"| {i+1:<4} | {max_score:<11.4f} | {node_pub_year:<8} | {score_year_of_max:<10} | {title_line:<50} | {temporal_str} |")
        except KeyError:
            title_line = f"{node_id} (Title N/A)"
            print(f"| {i+1:<4} | {max_score:<11.4f} | {'N/A':<8} | {score_year_of_max:<10} | {title_line:<50} | {temporal_str} |")
            
    print("-" * 150)
    print("="*150)


def display_final_results(final_anomaly_scores, all_scores_dict, df, top_n=20, specific_id="ANOMALY_TEST_002"):
    
    sorted_anomalies = sorted(
        final_anomaly_scores.items(), 
        key=lambda item: item[1], 
        reverse=True
    )

    print("\n--- Top 20 Most Anomalous Papers (by Max Score) ---")
    for i, (node_id, score) in enumerate(sorted_anomalies[:top_n]):
        try:
            paper_title = df.loc[node_id]['title']
            paper_year = df.loc[node_id]['year']
            print(f"#{i+1}: ID: {node_id} (Year: {paper_year})")
            print(f"     Title: {paper_title}")
            print(f"     Max Anomaly Score: {score:.4f}")
        except KeyError:
            print(f"#{i+1}: ID: {node_id} (Data not found in original DF)")
            print(f"     Max Anomaly Score: {score:.4f}")

    print("\n" + "="*40)
    print(f"--- Specific Anomaly Check for '{specific_id}' ---")
    
    if specific_id in final_anomaly_scores:
        final_score = final_anomaly_scores[specific_id]
        
        rank = -1
        for i, (node_id, score) in enumerate(sorted_anomalies):
            if node_id == specific_id:
                rank = i + 1
                break
        
        temporal_vector = all_scores_dict.get(specific_id, [])
        temporal_vector_str = ", ".join([f"{s:.4f}" for s in temporal_vector])
        
        print(f"Paper '{specific_id}' WAS FOUND.")
        print(f"  Final Aggregated (Max) Score: {final_score:.4f}")
        print(f"  Its Overall Rank:             #{rank} out of {len(sorted_anomalies)}")
        print(f"  Full Temporal Anomaly Vector (Raw): [{temporal_vector_str}]") 
        
        if rank > top_n:
            top_n_score = sorted_anomalies[top_n-1][1]
            print(f"\nREASON: Its score ({final_score:.4f}) is lower than the Top {top_n} cutoff score of {top_n_score:.4f}.")
        else:
            print(f"\nSUCCESS: This paper *is* in the Top {top_n} list.")
            
    else:
        print(f"Paper '{specific_id}' was NOT FOUND in the final results.")
    print("="*40)


def main():
    
    # === Section 4.2.1: Data Preparation (Original Data) ===
    print("--- LOADING AND PREPARING ORIGINAL DATA (SECTION 4.2.1) ---")
    df_original, full_G_original, features_original, node_to_idx_original, feature_dim = load_citation_data(FILE_PATH)
    if df_original is None:
        return
    
    min_year = int(df_original['year'].min())
    csv_max_year = int(df_original['year'].max())
    
    max_year_validation = max(csv_max_year, INJECTION_YEAR)
    max_year_detection = max(csv_max_year, 2020)

    
    # =========================================================
    # ===      PHASE 1: VALIDATION (NEW ANOMALY INJECTION)  ===
    # =========================================================
    print("\n\n" + "="*50)
    print("     PHASE 1: RUNNING VALIDATION (INJECTING NEW FAKE PAPERS)")
    print("="*50)

    # === FIX IMPLEMENTATION: Injects papers with random noise features ===
    df_augmented, G_augmented, features_augmented, node_to_idx_augmented, ground_truth_set = inject_new_anomalous_papers(
        full_G_original.copy(), 
        features_original, 
        node_to_idx_original.copy(), 
        df_original,
        injection_percent=INJECTION_PERCENT,
        connection_percent=CONNECTION_PERCENT,
        feature_dim=feature_dim
    )

    # === Run the Core Pipeline on AUGMENTED data ===
    print("\nStarting Temporal Pipeline on AUGMENTED graph...")
    validation_anomaly_scores, validation_years = run_temporal_pipeline(
        G_augmented, features_augmented, node_to_idx_augmented, min_year, max_year_validation
    )
    
    # === Aggregate Validation Scores ===
    print("\nAggregating validation scores...")
    final_validation_scores = aggregate_scores(validation_anomaly_scores, strategy='max')

    # === Section 5.1: Evaluate Detection Performance ===
    print("\n--- Validation Results (Section 5.1) ---")
    precision, recall, f1_score, predicted_anomalies_set, sorted_validation_nodes = calculate_validation_metrics(
        final_validation_scores, 
        ground_truth_set
    )
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1_score:.4f}")

    # --- Detailed Validation Analysis ---
    analyze_validation_results(
        predicted_anomalies_set, 
        ground_truth_set, 
        final_validation_scores, 
        df_augmented,
        top_n_fp=10
    )

    
    # =========================================================
    # ===      PHASE 2: DETECTION (ORIGINAL DATA)           ===
    # =========================================================
    print("\n\n" + "="*50)
    print("     PHASE 2: RUNNING DETECTION (SECTION 4 - ORIGINAL GRAPH)")
    print("="*50)

    # === Run the Core Pipeline on ORIGINAL data ===
    print("\nStarting Temporal Pipeline on ORIGINAL graph...")
    detection_anomaly_scores, detection_years = run_temporal_pipeline(
        full_G_original, features_original, node_to_idx_original, min_year, max_year_detection
    )

    # === Aggregate Detection Scores ===
    print("\nAggregating detection scores...")
    final_detection_scores = aggregate_scores(detection_anomaly_scores, strategy='max')
    
    # --- Temporal Vector Output ---
    print_temporal_vectors(
        detection_anomaly_scores, 
        detection_years, 
        df_original, 
        top_n=10
    )
    
    # === Section 4.5: Display Anomalies Found ===
    print("\n--- Final Results (Section 4.5) ---")
    display_final_results(
        final_detection_scores, 
        detection_anomaly_scores,
        df_original, 
        top_n=20, 
        specific_id="ANOMALY_TEST_002"
    )
    
    print("\n\n--- FULL PROCESS COMPLETE ---")


if __name__ == "__main__":
    main()