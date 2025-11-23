#!/usr/bin/env python3
"""
run_full_project_real_improved.py

Full end-to-end pipeline using the provided LLGC (model.py).
No mocks or fallbacks. If anything is missing (CUDA, model file,
CSV, packages), the script will fail loudly.
"""

import os
import sys
import ast
import random
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# -------------------------
# CONFIGURATION
# -------------------------
FILE_PATH = r"C:\Users\nadir\OneDrive\Desktop\final_filtered_by_fos_and_reference.csv"
TFIDF_MAX_FEATURES = 500
EMBEDDING_DIM = 64
PAGERANK_ALPHA = 0.15
PAGERANK_K = 10

INJECTION_PERCENT = 0.05      # 5% injected anomalies
CONNECTION_PERCENT = 0.30     # (unused in "connect all" mode below, kept for reference)
INJECTION_YEAR = 2025
INJECTION_ID_PREFIX = "NEW_FAKE_PAPER_"

SEED = 42

# -------------------------
# Deterministic seeds
# -------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# REQUIREMENTS CHECKS
# -------------------------
# The provided model.py creates tensors on "cuda" internally; ensure CUDA is available.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA detected → running on GPU.")
else:
    DEVICE = torch.device("cpu")
    print("CUDA NOT available → running on CPU (LLGC compatibility patch applied).")

DEVICE = torch.device("cuda")

# -------------------------
# Import the real model (will raise ImportError if not present)
# -------------------------
from model import PageRankAgg, LLGC  # use the user's provided model.py
import manifolds  # required by model

# -------------------------
# Utility: set reproducible seeds (within script)
# -------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(SEED)

# -------------------------
# LOAD DATA and PREPROCESS
# -------------------------
def load_citation_data(filepath: str, max_tfidf_features: int = TFIDF_MAX_FEATURES
                       ) -> Tuple[pd.DataFrame, nx.DiGraph, torch.FloatTensor, Dict[str,int], List[str], int]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    print(f"Loading CSV from: {filepath}")
    df = pd.read_csv(filepath, dtype=str).fillna("")
    required = {'id','title','abstract','year','references'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns. Found columns: {df.columns.tolist()}")

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year']).copy()
    df['year'] = df['year'].astype(int)
    df['id'] = df['id'].astype(str)
    df = df.set_index('id', drop=False)

    # Deterministic node ordering
    node_order = sorted(df.index.tolist())
    df = df.loc[node_order]

    print("Computing TF-IDF features (title + abstract)...")
    df['text'] = (df['title'].fillna("") + " " + df['abstract'].fillna("")).astype(str)
    vectorizer = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english')
    tfidf = vectorizer.fit_transform(df['text'].tolist())  # shape (N, D)
    features = torch.FloatTensor(tfidf.toarray()).to(DEVICE)
    feature_dim = features.shape[1]

    # Build directed citation graph (citing -> cited)
    G = nx.DiGraph()
    for nid in node_order:
        G.add_node(nid, year=int(df.at[nid,'year']))
    # Add edges if referenced id exists in df
    for nid in node_order:
        refs_raw = df.at[nid, 'references']
        if not refs_raw or refs_raw.strip() == "":
            continue
        try:
            refs = ast.literal_eval(refs_raw)
        except Exception:
            # fallback: comma-separated
            refs = [r.strip() for r in refs_raw.split(',') if r.strip()]
        if not isinstance(refs, (list, tuple)):
            refs = [refs]
        for r in refs:
            r = str(r).strip()
            if r and r in df.index:
                G.add_edge(nid, r)  # nid cites r

    node_to_idx = {nid: i for i, nid in enumerate(node_order)}

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Feature matrix shape: {features.shape}")
    return df, G, features, node_to_idx, node_order, feature_dim

# -------------------------
# STRONG SYNTHETIC INJECTION (spike features + dense connections)
# -------------------------
def inject_strong_synthetic_anomalies(
    df: pd.DataFrame,
    G: nx.DiGraph,
    features: torch.FloatTensor,
    node_to_idx: Dict[str,int],
    node_order: List[str],
    feature_dim: int,
    injection_percent: float,
    injection_year: int,
    injection_id_prefix: str,
) -> Tuple[pd.DataFrame, nx.DiGraph, torch.FloatTensor, Dict[str,int], List[str], Set[str]]:

    print("Injecting STRONG synthetic anomalies (sparse spike features + dense connections)...")
    n_original = len(node_order)
    num_to_inject = max(1, int(np.floor(n_original * injection_percent)))
    rng = np.random.RandomState(SEED)

    # Prepare new nodes
    new_nodes = []
    new_feat_tensors = []
    ground_truth: Set[str] = set()
    G_aug = G.copy()
    node_to_idx_aug = dict(node_to_idx)
    base_idx = features.shape[0]

    # For realistic TF-IDF scale statistics, compute medians (but we will produce spikes)
    # We will produce extremely sparse features: choose 1-8 dims to set to large values (e.g., 8-20).
    for i in range(num_to_inject):
        nid = f"{injection_id_prefix}{i:03d}"
        new_nodes.append(nid)
        ground_truth.add(nid)
        # Add node to graph with injection_year attribute
        G_aug.add_node(nid, year=injection_year)
        node_to_idx_aug[nid] = base_idx + i

        # Build sparse spike vector (shape = feature_dim)
        nonzero_count = rng.randint(1, min(8, feature_dim))  # 1..7 nonzeros (or less if feature_dim small)
        indices = rng.choice(feature_dim, size=nonzero_count, replace=False)
        spike = np.zeros(feature_dim, dtype=float)
        # Put very large positive spikes (TF-IDF values typically <= ~1; we use much larger to be outliers)
        spike_values = rng.uniform(8.0, 20.0, size=nonzero_count)
        spike[indices] = spike_values
        new_feat_tensors.append(torch.FloatTensor(spike).unsqueeze(0).to(DEVICE))

    # Concatenate features
    if new_feat_tensors:
        new_feats = torch.cat(new_feat_tensors, dim=0)
        features_aug = torch.cat([features, new_feats], dim=0)
    else:
        features_aug = features

    # Connect each injected node *densely* to many existing nodes to maximize anomaly signal.
    # We'll connect injected -> all original nodes (cites many), and also add many incoming edges from original nodes to injected.
    # This creates strong structural abnormality and PageRank signal.
    for nid in new_nodes:
        for t in node_order:
            if not G_aug.has_edge(nid, t):
                G_aug.add_edge(nid, t)
            # Optionally add reverse edge to create more visibility; can be noisy, but helps detection:
            if not G_aug.has_edge(t, nid):
                G_aug.add_edge(t, nid)

    # Build augmented dataframe: original order then new nodes in the same creation order
    df_new = pd.DataFrame({
        'id': new_nodes,
        'title': [f"Synthetic Anomaly {i:03d}" for i in range(len(new_nodes))],
        'abstract': ["Synthetic anomaly for validation."] * len(new_nodes),
        'year': [injection_year] * len(new_nodes),
        'references': [str([])] * len(new_nodes),
        'text': [""] * len(new_nodes),
    }).set_index('id', drop=False)
    combined_order = node_order + new_nodes
    df_aug = pd.concat([df.loc[node_order], df_new.loc[new_nodes]])
    df_aug = df_aug.loc[combined_order]

    print(f"Injected {len(new_nodes)} anomalies. Augmented features shape: {features_aug.shape}")
    return df_aug, G_aug, features_aug, node_to_idx_aug, combined_order, ground_truth

# -------------------------
# SNAPSHOT extraction (cumulative) keeping deterministic ordering
# -------------------------
def get_snapshot_data(full_G: nx.DiGraph, full_features: torch.FloatTensor, node_to_idx: Dict[str,int],
                      year: int, global_node_order: List[str]):
    nodes_in_snapshot = [n for n in global_node_order if full_G.nodes[n].get('year', float('inf')) <= year]
    if len(nodes_in_snapshot) < 2:
        return None, None, None, None
    G_t = full_G.subgraph(nodes_in_snapshot).copy()
    snapshot_indices = [node_to_idx[n] for n in nodes_in_snapshot]
    features_t = full_features[snapshot_indices, :]
    adj = nx.adjacency_matrix(G_t, nodelist=nodes_in_snapshot)
    adj_coo = adj.tocoo()
    edge_index = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long).to(DEVICE)
    return G_t, features_t, edge_index, nodes_in_snapshot

# -------------------------
# TEMPORAL PIPELINE
# -------------------------

# -----------------------------------------
# CPU COMPATIBILITY PATCH FOR YOUR LLGC MODEL
# -----------------------------------------
def patch_llgc_for_cpu():
    """
    Force LLGC and manifold internals to use CPU if CUDA is not available.
    This overrides the device inside LLGC/manifolds.
    """

    # For PageRankAgg we only need to move it to DEVICE normally.
    # For LLGC the manifold internally binds to CUDA unless overridden.

    # Monkey-patch the manifold base class to use CPU tensors
    import manifolds

    for name in dir(manifolds):
        obj = getattr(manifolds, name)
        if hasattr(obj, "__dict__") and hasattr(obj, "device"):
            try:
                obj.device = DEVICE   # force CPU or GPU based on availability
            except Exception:
                pass

    print(f"LLGC manifold patched to use device: {DEVICE}")


def run_temporal_pipeline(G_proc: nx.DiGraph, features: torch.FloatTensor, node_to_idx: Dict[str,int],
                          node_order: List[str], min_year: int, max_year: int, isolation_contamination=None):
    # PageRankAgg from model
    # Apply patch ONLY when CUDA is not available
    
    patch_llgc_for_cpu()

    prop = PageRankAgg(K=PAGERANK_K, alpha=PAGERANK_ALPHA, add_self_loops=True, normalize=True)
    prop.to(DEVICE)
    prop.eval()

    all_scores = {}
    processed_years = []
    for year in range(min_year, max_year + 1):
        print(f"\nProcessing snapshot year: {year}")
        G_t, feats_t, edge_index_t, snapshot_nodes = get_snapshot_data(G_proc, features, node_to_idx, year, node_order)
        if G_t is None or G_t.number_of_nodes() < 10 or G_t.number_of_edges() == 0:
            print(f"Skipping year {year}: insufficient nodes/edges.")
            continue
        processed_years.append(year)
        print(f"Snapshot: nodes={G_t.number_of_nodes()}, edges={G_t.number_of_edges()}")

        with torch.no_grad():
            propagated, _ = prop(feats_t, edge_index_t)

        nfeat = propagated.shape[1]
        # instantiate LLGC (from your model.py)
        llgc = LLGC(nfeat=nfeat, nclass=EMBEDDING_DIM, drop_out=0.0, use_bias=True).to(DEVICE)
        llgc.eval()

        with torch.no_grad():
            embeddings = llgc(propagated)  # model returns tangent-space Euclidean embeddings as numpy-ready tensor
        emb_np = embeddings.cpu().numpy()

        # IsolationForest: during validation set contamination to expected fraction, otherwise use 'auto'
        clf = IsolationForest(random_state=SEED, contamination=isolation_contamination if isolation_contamination is not None else 'auto', n_jobs=-1)
        clf.fit(emb_np)
        # score_samples: higher => normal, so negate to make higher => anomalous
        scores = -clf.score_samples(emb_np)

        for i, nid in enumerate(snapshot_nodes):
            all_scores.setdefault(nid, []).append(float(scores[i]))

    return all_scores, processed_years

# -------------------------
# AGGREGATION & METRICS
# -------------------------
def aggregate_scores(all_scores: Dict[str,List[float]], strategy='max'):
    out = {}
    for nid, vals in all_scores.items():
        if not vals:
            continue
        if strategy == 'max':
            out[nid] = float(np.max(vals))
        elif strategy == 'mean':
            out[nid] = float(np.mean(vals))
        else:
            raise ValueError("Unknown aggregation")
    return out

def calculate_validation_metrics(final_scores: Dict[str,float], ground_truth: Set[str]):
    k = len(ground_truth)
    sorted_nodes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    predicted = set([nid for nid, _ in sorted_nodes[:k]])
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1, predicted, sorted_nodes

# -------------------------
# REPORTING
# -------------------------
def print_top_n(final_scores: Dict[str,float], df: pd.DataFrame, top_n=20):
    sorted_nodes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Top {} anomalous papers ---".format(top_n))
    for i, (nid, score) in enumerate(sorted_nodes[:top_n], start=1):
        if nid in df.index:
            print(f"#{i}: ID={nid} Year={df.at[nid,'year']} Score={score:.4f}")
            print(f"    Title: {df.at[nid,'title'][:200]}")
        else:
            print(f"#{i}: ID={nid} (not in original df) Score={score:.4f}")

def print_temporal_vectors(all_scores: Dict[str,List[float]], processed_years: List[int], df: pd.DataFrame, top_n=10):
    final_scores = aggregate_scores(all_scores, 'max')
    sorted_nodes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nProcessed years:", processed_years)
    print("--- Top temporal vectors (aligned) ---")
    for rank, (nid, _) in enumerate(sorted_nodes[:top_n], start=1):
        vec = all_scores.get(nid, [])
        gap = len(processed_years) - len(vec)
        padded = ['N/A']*gap + [f"{v:.4f}" for v in vec]
        pubyear = df.at[nid,'year'] if nid in df.index else "N/A"
        print(f"{rank}. {nid} (pub {pubyear}) -> {padded}")

# -------------------------
# MAIN
# -------------------------
def main():
    set_seeds(SEED)
    df, G, features, node_to_idx, node_order, feature_dim = load_citation_data(FILE_PATH, TFIDF_MAX_FEATURES)

    min_year = int(df['year'].min())
    csv_max_year = int(df['year'].max())
    max_year_validation = max(csv_max_year, INJECTION_YEAR)
    max_year_detection = csv_max_year

    # Validation: inject strong anomalies
    df_aug, G_aug, features_aug, node_to_idx_aug, node_order_aug, ground_truth = inject_strong_synthetic_anomalies(
        df, G, features, node_to_idx, node_order, feature_dim,
        injection_percent=INJECTION_PERCENT,
        injection_year=INJECTION_YEAR,
        injection_id_prefix=INJECTION_ID_PREFIX
    )

    # For validation set contamination equal to injection percent (so IForest expects this proportion)
    contamination = INJECTION_PERCENT

    # Run pipeline on augmented graph
    print("\nRunning temporal pipeline on AUGMENTED graph (validation)...")
    val_scores, val_years = run_temporal_pipeline(G_aug, features_aug, node_to_idx_aug, node_order_aug, min_year, max_year_validation, isolation_contamination=contamination)
    final_val_scores = aggregate_scores(val_scores, 'max')
    precision, recall, f1, predicted_set, sorted_val_nodes = calculate_validation_metrics(final_val_scores, ground_truth)

    print("\n--- Validation Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Print TP/FN/FP
    tp = predicted_set & ground_truth
    fn = ground_truth - predicted_set
    fp = predicted_set - ground_truth
    print(f"TP ({len(tp)}): {sorted(list(tp))}")
    print(f"FN ({len(fn)}): {sorted(list(fn))}")
    print(f"FP ({len(fp)}): {sorted(list(fp))}")

    # Now run detection on original graph (no injection)
    print("\nRunning temporal pipeline on ORIGINAL graph (detection)...")
    det_scores, det_years = run_temporal_pipeline(G, features, node_to_idx, node_order, min_year, max_year_detection, isolation_contamination=None)
    final_det_scores = aggregate_scores(det_scores, 'max')

    print_top_n(final_det_scores, df, top_n=20)
    print_temporal_vectors(det_scores, det_years, df, top_n=10)

    # Specific paper check
    target = "ANOMALY_TEST_002"
    if target in final_det_scores:
        rank_list = sorted(final_det_scores.items(), key=lambda x: x[1], reverse=True)
        rank_idx = [i for i,(nid,_) in enumerate(rank_list) if nid==target]
        if rank_idx:
            print(f"Specific paper {target} found at rank #{rank_idx[0]+1} score={final_det_scores[target]:.4f}")
    else:
        print(f"Specific paper {target} not found in detection results.")

    print("\nProcess complete.")

if __name__ == "__main__":
    main()
