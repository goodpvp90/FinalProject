# ==========================
# 0. Imports & Setup
# ==========================
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import ast
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from model import LLGC, PageRankAgg

# --------------------------
# Reproducibility
# --------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 1. Load Data
# ==========================
FILE = "final_filtered_by_fos_and_reference.csv"
df = pd.read_csv(FILE)

df['references'] = df['references'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)

df['year'] = df['year'].astype(int)

# ==========================
# 2. Build Citation Graph
# ==========================
G = nx.DiGraph()
paper_ids = set(df['id'])

for _, row in df.iterrows():
    G.add_node(row['id'])

for _, row in df.iterrows():
    for ref in row['references']:
        if ref in paper_ids:
            G.add_edge(row['id'], ref)

id_to_idx = {pid: i for i, pid in enumerate(G.nodes())}
idx_to_id = {i: pid for pid, i in id_to_idx.items()}
N = len(id_to_idx)

# ==========================
# 3. STRUCTURAL FEATURES (NO TEXT)
# ==========================
def build_structural_features(G, id_to_idx):
    deg_in = dict(G.in_degree())
    deg_out = dict(G.out_degree())
    pagerank = nx.pagerank(G, alpha=0.85)
    clustering = nx.clustering(G.to_undirected())

    # 🔴 FIX: remove self-loops for core number
    G_simple = G.to_undirected().copy()
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    core = nx.core_number(G_simple)

    avg_nbr_deg = nx.average_neighbor_degree(G)

    X = np.zeros((len(id_to_idx), 8), dtype=np.float32)

    for node, idx in id_to_idx.items():
        X[idx] = [
            deg_in.get(node, 0),
            deg_out.get(node, 0),
            deg_in.get(node, 0) / (deg_out.get(node, 1) + 1),
            pagerank.get(node, 0),
            clustering.get(node, 0),
            core.get(node, 0),
            avg_nbr_deg.get(node, 0),
            G.degree(node)
        ]

    return StandardScaler().fit_transform(X)


X_static = build_structural_features(G, id_to_idx)
X_tensor = torch.tensor(X_static, dtype=torch.float32).to(DEVICE)

# ==========================
# 4. Adjacency Utilities
# ==========================
def normalize_adj(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    deg = np.array(adj.sum(1)).flatten()
    deg_inv = np.power(deg, -0.5)
    deg_inv[np.isinf(deg_inv)] = 0
    D = sp.diags(deg_inv)
    return D @ adj @ D

def to_torch_sparse(mx):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col))).long()
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(indices, values, mx.shape).to(DEVICE)

edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
rows, cols = zip(*edges)
adj_full = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
adj_full = normalize_adj(adj_full)

# ==========================
# 5. Temporal Segmentation
# ==========================
DELTA = 61
years = df['year']
t_min, t_max = years.min(), years.max()

segments = []
for t in range(t_min, t_max, DELTA):
    nodes = df[df['year'] <= t + DELTA]['id']
    indices = [id_to_idx[n] for n in nodes if n in id_to_idx]
    if indices:
        segments.append((t, t + DELTA, indices))

# ==========================
# 6. Training Function
# ==========================
def train_llgc(model, X, adj_idx, prev_Z=None, epochs=100, lr=0.01):
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        Z = model(X)

        row, col = adj_idx
        recon = (Z[row] - Z[col]).pow(2).sum(1).mean()

        loss = recon

        if prev_Z is not None:
            common = prev_Z.shape[0]
            temporal = (Z[:common] - prev_Z).pow(2).mean()
            loss += 2.0 * temporal

        deg = torch.bincount(row, minlength=Z.size(0)).float().to(DEVICE)
        deg = (deg - deg.mean()) / (deg.std() + 1e-6)
        norm = torch.norm(Z, dim=1)
        loss += 0.2 * torch.mean(torch.abs(norm - deg))

        loss.backward()
        opt.step()

    return model

# ==========================
# 7. Temporal Training (REAL ONLY)
# ==========================
prev_Z = None
real_embeddings = None

for t_start, t_end, idxs in segments:
    idxs = np.array(idxs)

    adj = adj_full[idxs[:, None], idxs]
    adj_t = to_torch_sparse(adj)

    sg = PageRankAgg(K=5, alpha=0.1, add_self_loops=False).to(DEVICE)
    Xg, _ = sg(X_tensor[idxs], adj_t._indices(), adj_t._values())

    model = LLGC(Xg.shape[1], 256, 0.0, True).to(DEVICE)
    model = train_llgc(model, Xg, adj_t._indices(), prev_Z)

    with torch.no_grad():
        Z = model(Xg).cpu()

    prev_Z = Z.clone()
    real_embeddings = Z.numpy()

# ==========================
# 8. Train Anomaly Models (REAL ONLY)
# ==========================
iso = IsolationForest(
    n_estimators=300,
    contamination=0.01,
    random_state=SEED
)
iso.fit(real_embeddings)

svm = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
svm.fit(real_embeddings)

# ==========================
# 9. Inject Synthetic Nodes
# ==========================
df_fake = pd.read_csv("fakes.csv")
df = pd.concat([df, df_fake], ignore_index=True)
df['is_synthetic'] = df['is_synthetic'].fillna(False)

# rebuild graph
G2 = nx.DiGraph()
for _, r in df.iterrows():
    G2.add_node(r['id'])
for _, r in df.iterrows():
    for ref in ast.literal_eval(str(r['references'])):
        if ref in G2:
            G2.add_edge(r['id'], ref)

id_to_idx2 = {pid: i for i, pid in enumerate(G2.nodes())}
X2 = build_structural_features(G2, id_to_idx2)
X2 = torch.tensor(X2, dtype=torch.float32).to(DEVICE)

edges2 = [(id_to_idx2[u], id_to_idx2[v]) for u, v in G2.edges()]
rows, cols = zip(*edges2)
adj2 = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                     shape=(len(G2), len(G2)))
adj2 = normalize_adj(adj2)
adj2 = to_torch_sparse(adj2)

sg = PageRankAgg(K=5, alpha=0.1, add_self_loops=False).to(DEVICE)
Xg2, _ = sg(X2, adj2._indices(), adj2._values())

with torch.no_grad():
    Z2 = model(Xg2).cpu().numpy()

# ==========================
# 10. Detect Fakes
# ==========================
pred_if = iso.predict(Z2)
pred_svm = svm.predict(Z2)

final_pred = np.where((pred_if == -1) & (pred_svm == -1), -1, 1)

fake_mask = df['is_synthetic'].values
detected = np.sum((final_pred == -1) & fake_mask)

print(f"\n🔥 DETECTED {detected}/{fake_mask.sum()} fakes "
      f"({detected/fake_mask.sum():.2%})")

# ==========================
# 11. Save Results
# ==========================
out = pd.DataFrame({
    "paper_id": df['id'],
    "prediction": final_pred,
    "is_synthetic": fake_mask
})

out.to_csv("final_detection_results.csv", index=False)
print("✅ Results saved: final_detection_results.csv")
