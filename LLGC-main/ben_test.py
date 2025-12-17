# ============================================================
# FULL LLGC STRUCTURAL ANOMALY PIPELINE (CLEAN + INJECT + VALIDATE)
# ============================================================

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import ast
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import torch
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from time import perf_counter

from model import LLGC, PageRankAgg

# ============================================================
# 0. Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Load Data
# ============================================================
DATASET = "final_filtered_by_fos_and_reference.csv"
FAKES = "fakes.csv"

print(f"Loading dataset: {DATASET}")
df = pd.read_csv(DATASET)

df["references"] = df["references"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

df["is_synthetic"] = False

# ============================================================
# 2. Build Graph
# ============================================================
def build_graph(dataframe):
    G = nx.DiGraph()
    ids = set(dataframe["id"])
    for pid in ids:
        G.add_node(pid)
    for _, row in dataframe.iterrows():
        for ref in row["references"]:
            if ref in ids:
                G.add_edge(row["id"], ref)
    return G

# ============================================================
# 3. Structural-Only Features (Random Seeds)
# ============================================================
def build_structural_features(G, dim=128):
    nodes = list(G.nodes())
    X = np.random.randn(len(nodes), dim).astype(np.float32)
    return nodes, torch.tensor(X).to(DEVICE)

# ============================================================
# 4. Edge Index
# ============================================================
def build_edge_index(G, nodes):
    id2idx = {n: i for i, n in enumerate(nodes)}
    edges = []
    for u, v in G.edges():
        edges.append([id2idx[u], id2idx[v]])
        edges.append([id2idx[v], id2idx[u]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)

# ============================================================
# 5. Train LLGC
# ============================================================
def train_llgc(X, edge_index, emb_dim=256, epochs=100):
    model = LLGC(
        nfeat=X.size(1),
        nclass=emb_dim,
        drop_out=0.0,
        use_bias=True
    ).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for _ in range(epochs):
        opt.zero_grad()
        Z = model(X)
        r, c = edge_index
        loss = (Z[r] - Z[c]).pow(2).sum(dim=1).mean()
        loss.backward()
        opt.step()

    return model

# ============================================================
# 6. Detect Anomalies
# ============================================================
def detect_anomalies(Z, contamination):
    iso = IsolationForest(contamination=contamination, random_state=SEED)
    pred = iso.fit_predict(Z)
    scores = iso.decision_function(Z)
    return scores, pred

# ============================================================
# 7. Inject Synthetic Nodes from CSV
# ============================================================
def inject_from_csv(df, fake_csv):
    df_fake = pd.read_csv(fake_csv)
    df_fake["references"] = df_fake["references"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df_fake["is_synthetic"] = True
    df_out = pd.concat([df, df_fake], ignore_index=True)
    return df_out

# ============================================================
# 8. INITIAL TRAINING
# ============================================================
print("\n=== INITIAL TRAINING ===")

G = build_graph(df)
nodes, X = build_structural_features(G)
edge_index = build_edge_index(G, nodes)

sgc = PageRankAgg(K=5, alpha=0.1).to(DEVICE)
X_gconv, _ = sgc(X, edge_index)

model = train_llgc(X_gconv, edge_index)

Z = model(X_gconv).detach().cpu().numpy()
scores, pred = detect_anomalies(Z, contamination=0.01)

results = []

for i, pid in enumerate(nodes):
    results.append({
        "paper_id": pid,
        "stage": "initial",
        "anomaly_score": scores[i],
        "prediction": pred[i],
        "is_synthetic": False
    })

anomalous_ids = [nodes[i] for i in range(len(nodes)) if pred[i] == -1]
print(f"Detected initial anomalies: {len(anomalous_ids)}")

# ============================================================
# 9. CLEAN CORE GRAPH
# ============================================================
print("\n=== CLEANING GRAPH ===")

df_core = df[~df["id"].isin(anomalous_ids)].copy()
print(f"Core size: {len(df_core)}")

G_core = build_graph(df_core)
nodes_c, X_c = build_structural_features(G_core)
edge_index_c = build_edge_index(G_core, nodes_c)

Xc_gconv, _ = sgc(X_c, edge_index_c)
model_core = train_llgc(Xc_gconv, edge_index_c)

# ============================================================
# 10. INJECT SYNTHETIC NODES
# ============================================================
print("\n=== INJECTING SYNTHETIC NODES ===")

df_injected = inject_from_csv(df_core, FAKES)
G_injected = build_graph(df_injected)

nodes_i, X_i = build_structural_features(G_injected)
edge_index_i = build_edge_index(G_injected, nodes_i)

Xi_gconv, _ = sgc(X_i, edge_index_i)

Zi = model_core(Xi_gconv).detach().cpu().numpy()

fake_mask = df_injected["is_synthetic"].values
fake_ratio = fake_mask.sum() / len(fake_mask)

scores_i, pred_i = detect_anomalies(Zi, contamination=fake_ratio)

# ============================================================
# 11. VALIDATION & OUTPUT
# ============================================================
detected = 0
for i, pid in enumerate(nodes_i):
    is_fake = pid in set(df_injected[df_injected["is_synthetic"]]["id"])
    if is_fake and pred_i[i] == -1:
        detected += 1

    results.append({
        "paper_id": pid,
        "stage": "post_injection",
        "anomaly_score": scores_i[i],
        "prediction": pred_i[i],
        "is_synthetic": is_fake
    })

print("\n=== VALIDATION RESULTS ===")
print(f"Injected anomalies: {fake_mask.sum()}")
print(f"Detected anomalies: {detected}")
print(f"Detection rate: {detected / fake_mask.sum():.2%}")

# ============================================================
# 12. SAVE RESULTS
# ============================================================
out_df = pd.DataFrame(results)
out_df.to_csv("llgc_full_anomaly_results.csv", index=False)

print("\n✅ Results saved to llgc_full_anomaly_results.csv")
