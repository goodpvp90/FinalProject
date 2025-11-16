"""
run_full_project.py

This script implements the *entire* process described in the
"Final Project Ben&Nadir.docx" document in a single, sequential run.

CORRECTION: This version adds global random seeds. The L²GC model
initializes with random weights, so without a seed, the results
are different every time. This seed makes the model's transformation
deterministic and the results reproducible.

NEW IN V4: The final display function now also finds a *specific*
paper ID ("ANOMALY_TEST_002") and prints its scores and
final rank, showing *why* it did not make the Top 20 list.
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
    print(f"ImportError: {e}")
    print("Please run this script from within the 'LLGC-main' directory.")
    print("It requires 'model.py' and 'manifolds'.")
    exit()

# ===================================================================
#                      CONFIGURATION SETTINGS
# ===================================================================
# --- Project Hyperparameters (from Word Doc) ---
FILE_PATH = "C:\\Users\\nadir\\OneDrive\\Desktop\\final_filtered_by_fos_and_reference.csv"
EMBEDDING_DIM = 64      # Output dimension for L2GC
PAGERANK_ALPHA = 0.15   # Alpha for PageRankAgg
PAGERANK_K = 10         # K (degree) for PageRankAgg

# --- Validation Hyperparameters (from Section 5.1) ---
INJECTION_PERCENT = 0.05    # "5-10% of the nodes"
CONNECTION_PERCENT = 0.30   # "connect... to approximately 30%"

# --- Deterministic Seed ---
# This ensures all random operations (torch model
# weights, numpy, and random.sample) are the same every time.
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
    Loads the CSV, creates the full graph, and generates TF-IDF features.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        print(f"Please place '{FILE_PATH}' in the same directory.")
        return None, None, None, None

    # Handle NaNs in text fields for TF-IDF
    df['abstract'] = df['abstract'].fillna('')
    df['title'] = df['title'].fillna('')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year']) # Drop rows with invalid years
    df['year'] = df['year'].astype(int)

    
    # Create a node list and ID-to-index mapping for matrix operations
    df['id'] = df['id'].astype(str)
    df = df.set_index('id', drop=False)
    node_ids = df.index.tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    # --- Feature Engineering (Section 4.2.1.1) ---
    print("Generating TF-IDF features (Section 4.2.1.1)...")
    df['text'] = df['title'] + ' ' + df['abstract']
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    # Convert to dense torch tensor for GNN
    features = torch.FloatTensor(tfidf_matrix.toarray()).to(DEVICE)

    # --- Graph Construction (Section 4.1) ---
    print("Building full citation graph (Section 4.1)...")
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(row['id'], year=row['year'])

    for idx, row in df.iterrows():
        try:
            # Safely evaluate the string representation of the list
            references = ast.literal_eval(row['references'])
            if isinstance(references, list):
                for ref_id in references:
                    ref_id = str(ref_id)
                    # Add edge only if both nodes exist in our dataset
                    if ref_id in node_to_idx and row['id'] in node_to_idx:
                        G.add_edge(row['id'], ref_id)
        except (ValueError, SyntaxError):
            pass
            
    print(f"Data loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Feature matrix shape: {features.shape}")
    
    return df, G, features, node_to_idx


def inject_synthetic_anomalies(full_G, injection_percent, connection_percent):
    """
    Implements Section 5.1: Validation Methodology.
    Creates an augmented graph by injecting synthetic anomalies.
    """
    print(f"Injecting anomalies in {injection_percent*100}% of nodes...")
    G_augmented = full_G.copy()
    all_nodes = list(full_G.nodes())
    
    # Select nodes to be anomalous
    num_to_inject = int(len(all_nodes) * injection_percent)
    ground_truth_anomalies = set(random.sample(all_nodes, num_to_inject))
    
    num_to_connect = int(len(all_nodes) * connection_percent)
    
    edges_added = 0
    for anomalous_node in ground_truth_anomalies:
        # Select target nodes to connect to
        target_nodes = random.sample(all_nodes, num_to_connect)
        for target_node in target_nodes:
            if anomalous_node != target_node and not G_augmented.has_edge(anomalous_node, target_node):
                G_augmented.add_edge(anomalous_node, target_node)
                edges_added += 1
                
    print(f"Added {edges_added} synthetic edges to {len(ground_truth_anomalies)} anomalous nodes.")
    return G_augmented, ground_truth_anomalies


def get_snapshot_data(full_G, full_features, node_to_idx, year):
    """
    Implements Section 4.2.2: Temporal Segmentation.
    Creates the cumulative temporal subgraph G_t for a given year.
    """
    
    # Select nodes present up to the current year
    nodes_in_snapshot = [
        node for node, data in full_G.nodes(data=True) 
        if data.get('year', float('inf')) <= year
    ]
    
    if len(nodes_in_snapshot) < 2: # Need at least 2 nodes for a graph
        return None, None, None, None

    # Create the cumulative subgraph G_t
    G_t = full_G.subgraph(nodes_in_snapshot)
    
    # Get the indices of these nodes in the full feature matrix
    snapshot_indices = [node_to_idx[node_id] for node_id in G_t.nodes()]
    
    # Create the feature matrix for this snapshot
    features_t = full_features[snapshot_indices, :]

    # Create edge_index, which is what PageRankAgg expects
    adj_t_scipy = nx.adjacency_matrix(G_t, nodelist=G_t.nodes())
    adj_t_coo = adj_t_scipy.tocoo()
    
    edge_index_t = torch.tensor(
        np.vstack((adj_t_coo.row, adj_t_coo.col)), 
        dtype=torch.long
    ).to(DEVICE)
    
    return G_t, features_t, edge_index_t, snapshot_indices


def run_temporal_pipeline(G_to_process, features, node_to_idx, min_year, max_year):
    """
    Implements Section 4.3, 4.4, and 4.5.
    Runs the full L2GC + Isolation Forest pipeline over all temporal snapshots.
    """
    
    # === Section 3.2.1: Euclidean Propagation Model ===
    euclidean_prop_model = PageRankAgg(K=PAGERANK_K, alpha=PAGERANK_ALPHA, 
                                       add_self_loops=True, normalize=True)
    euclidean_prop_model.to(DEVICE)
    euclidean_prop_model.eval() # Set to evaluation mode
    
    all_anomaly_scores = {} # Dict to store scores: {node_id: [scores]}

    # === Section 4.2.2: Temporal Segmentation Loop ===
    for year in range(min_year, max_year + 1):
        print(f"\n--- Processing Snapshot for Year: {year} ---")
        
        G_t, features_t, edge_index_t, snapshot_indices = get_snapshot_data(
            G_to_process, features, node_to_idx, year
        )
        
        if G_t is None or G_t.number_of_nodes() < 10 or G_t.number_of_edges() == 0:
            print(f"Skipping year {year}, not enough nodes/edges.")
            continue
            
        print(f"Snapshot G_{year}: {G_t.number_of_nodes()} nodes, {G_t.number_of_edges()} edges.")

        # === Section 4.3: L²GC Model on Temporal Subgraphs ===
        
        # --- Step 1: Euclidean Propagation (Section 3.2.1) ---
        with torch.no_grad(): # No training, just forward pass
            propagated_features, _ = euclidean_prop_model(features_t, edge_index_t)
        
        # --- Step 2: Lorentzian Transformation (Section 3.2.2) ---
        n_features = propagated_features.shape[1]
        
        # This model is randomly initialized. Because we set a
        # torch.manual_seed(), its weights will be the *same* random
        # weights every time, making the run deterministic.
        llgc_model = LLGC(
            nfeat=n_features,
            nclass=EMBEDDING_DIM,
            drop_out=0.0, # No dropout during inference
            use_bias=True
        ).to(DEVICE)
        llgc_model.eval() # Set to evaluation mode
        
        
        # --- Step 3: Get Euclidean Embeddings (Section 4.5, Step 1) ---
        with torch.no_grad():
            euclidean_embeddings_zt = llgc_model(propagated_features)
            
        euclidean_embeddings_zt = euclidean_embeddings_zt.cpu().numpy()

        # === Section 4.5: Anomaly Scoring (Steps 2 & 3) ===
        print(f"Running Isolation Forest on {euclidean_embeddings_zt.shape[0]} nodes...")
        
        # [cite_start]--- Step 2: Apply Isolation Forest [cite: 347-349] ---
        # The random_state=SEED makes the IForest deterministic
        clf = IsolationForest(random_state=SEED, n_jobs=-1) 
        clf.fit(euclidean_embeddings_zt)
        
        # We negate the score: Higher score = MORE ANOMALOUS
        snapshot_scores = -clf.score_samples(euclidean_embeddings_zt) 
        
        # [cite_start]--- Step 3: Store Temporal Anomaly Vector [cite: 351-352] ---
        node_ids_t = list(G_t.nodes())
        for i, node_id in enumerate(node_ids_t):
            if node_id not in all_anomaly_scores:
                all_anomaly_scores[node_id] = []
            all_anomaly_scores[node_id].append(snapshot_scores[i])
            
    return all_anomaly_scores


def aggregate_scores(all_anomaly_scores, strategy='max'):
    """
    Aggregates temporal scores for each node.
    """
    final_anomaly_scores = {}
    for node_id, scores in all_anomaly_scores.items():
        if scores:
            if strategy == 'max':
                final_anomaly_scores[node_id] = np.max(scores) 
            elif strategy == 'mean':
                final_anomaly_scores[node_id] = np.mean(scores)
                
    return final_anomaly_scores


def display_final_results(final_anomaly_scores, all_scores_dict, df, top_n=20, specific_id="ANOMALY_TEST_002"):
    """
    Prints the top N most anomalous nodes, AND
    finds the specific_id to print its scores and rank.
    """
    # Sort nodes by their max anomaly score, descending
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

    # =========================================================
    # ===      NEW: SPECIFIC CHECK FOR YOUR TEST PAPER      ===
    # =========================================================
    print("\n" + "="*40)
    print(f"--- Specific Anomaly Check for '{specific_id}' ---")
    
    # Check if the paper's ID is in the final aggregated scores
    if specific_id in final_anomaly_scores:
        final_score = final_anomaly_scores[specific_id]
        
        # Find its rank
        rank = -1
        for i, (node_id, score) in enumerate(sorted_anomalies):
            if node_id == specific_id:
                rank = i + 1
                break
        
        # Get the full list of scores over time
        temporal_vector = all_scores_dict.get(specific_id, [])
        
        # Format it for readability
        temporal_vector_str = ", ".join([f"{s:.4f}" for s in temporal_vector])
        
        print(f"Paper '{specific_id}' WAS FOUND.")
        print(f"  Final Aggregated (Max) Score: {final_score:.4f}")
        print(f"  Its Overall Rank:             #{rank} out of {len(sorted_anomalies)}")
        print(f"  Full Temporal Anomaly Vector: [{temporal_vector_str}]")
        
        # Explain why it's not in the Top 20
        if rank > top_n:
            top_n_score = sorted_anomalies[top_n-1][1] # Score of the #20 paper
            print(f"\nREASON: Its score ({final_score:.4f}) is lower than the Top {top_n} cutoff score of {top_n_score:.4f}.")
        else:
            print(f"\nSUCCESS: This paper *is* in the Top {top_n} list.")
            
    else:
        print(f"Paper '{specific_id}' was NOT FOUND in the final results.")
        print("  (Did you add it as the last line in the CSV file?)")
        print("  (Is its publication year 2020?)")
    print("="*40)
    # =========================================================


def calculate_validation_metrics(final_anomaly_scores, ground_truth_set):
    """
    Implements Section 5.1: Evaluation Metrics.
    Calculates Precision, Recall, and F1-Score.
    """
    
    # We predict the top N nodes, where N = number of injected anomalies
    k = len(ground_truth_set)
    
    # Sort all nodes by their final anomaly score
    sorted_nodes = sorted(
        final_anomaly_scores.items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    
    # Get the set of predicted anomalies
    predicted_anomalies_set = set([node_id for node_id, score in sorted_nodes[:k]])
    
    # --- Calculate TP, FP, FN ---
    true_positives_set = ground_truth_set.intersection(predicted_anomalies_set)
    
    TP = len(true_positives_set)
    FP = len(predicted_anomalies_set - ground_truth_set)
    FN = len(ground_truth_set - predicted_anomalies_set)
    
    print(f"Evaluation Details: TP={TP}, FP={FP}, FN={FN}")
    
    # [cite_start]--- Calculate Metrics [cite: 371-377] ---
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # [cite_start]Precision [cite: 372-373]
    precision = TP / (TP + FP + epsilon)
    
    # [cite_start]Recall [cite: 374-375]
    recall = TP / (TP + FN + epsilon)
    
    # [cite_start]F1 Score [cite: 376-377]
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return precision, recall, f1_score


def main():
    """
    Main function to run the full, sequential project pipeline.
    """
    
    # === Section 4.2.1: Data Preparation ===
    print("--- LOADING AND PREPARING ORIGINAL DATA (SECTION 4.2.1) ---")
    df, full_G_original, features, node_to_idx = load_citation_data(FILE_PATH)
    if df is None:
        return
    
    min_year = int(df['year'].min())
    csv_max_year = int(df['year'].max())
    
    # Find the max year, accounting for test papers
    test_paper_year = 2020 # Year for "ANOMALY_TEST_002"
    max_year = max(csv_max_year, test_paper_year)

    
    # =========================================================
    # ===      PHASE 1: VALIDATION (SECTION 5)              ===
    # =========================================================
    print("\n\n" + "="*50)
    print("     PHASE 1: RUNNING VALIDATION (SECTION 5)")
    print("="*50)

    # === Section 5.1: Inject Synthetic Anomalies ===
    G_augmented, ground_truth_set = inject_synthetic_anomalies(
        full_G_original, 
        injection_percent=INJECTION_PERCENT,
        connection_percent=CONNECTION_PERCENT
    )

    # === Run the Core Pipeline on AUGMENTED data ===
    print("\nStarting Temporal Pipeline on AUGMENTED graph...")
    validation_anomaly_scores = run_temporal_pipeline(
        G_augmented, features, node_to_idx, min_year, max_year
    )
    
    # === Aggregate Validation Scores ===
    print("\nAggregating validation scores...")
    final_validation_scores = aggregate_scores(validation_anomaly_scores, strategy='max')

    # === Section 5.1: Evaluate Detection Performance ===
    print("\n--- Validation Results (Section 5.1) ---")
    precision, recall, f1_score = calculate_validation_metrics(
        final_validation_scores, 
        ground_truth_set
    )
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1_score:.4f}")
    
    print("\n--- Comparison to Expected Results [cite: 402-404] ---")
    print(f"   Expected Precision: 0.90")
    print(f"   Expected Recall:    0.80")
    print(f"   Expected F1 Score:  0.85")
    print("(Note: Actual scores depend on data and random injection.)")

    
    # =========================================================
    # ===      PHASE 2: DETECTION (SECTION 4)               ===
    # =========================================================
    print("\n\n" + "="*50)
    print("     PHASE 2: RUNNING DETECTION (SECTION 4)")
    print("="*50)

    # === Run the Core Pipeline on ORIGINAL data ===
    print("\nStarting Temporal Pipeline on ORIGINAL graph...")
    detection_anomaly_scores = run_temporal_pipeline(
        full_G_original, features, node_to_idx, min_year, max_year
    )

    # === Aggregate Detection Scores ===
    print("\nAggregating detection scores...")
    final_detection_scores = aggregate_scores(detection_anomaly_scores, strategy='max')
    
    # === Section 4.5: Display Anomalies Found ===
    print("\n--- Final Results (Section 4.5) ---")
    # This function now prints the Top 20 AND your specific paper's info
    display_final_results(
        final_detection_scores, 
        detection_anomaly_scores, # Pass in the full dict
        df, 
        top_n=20, 
        specific_id="ANOMALY_TEST_002"
    )
    
    print("\n\n--- FULL PROCESS COMPLETE ---")


if __name__ == "__main__":
    main()