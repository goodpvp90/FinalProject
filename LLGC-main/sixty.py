# ======================================================
# 0. Imports, Environment, Safety
# ======================================================
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import ast
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import torch
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from model import LLGC, PageRankAgg

# ======================================================
# 1. Reproducibility
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# 2. Utility Guards
# ======================================================
def ensure_column(df, col, default=False):
    if col not in df.columns:
        df[col] = default
    return df

# ======================================================
# 3. Load Data
# ======================================================
df = pd.read_csv("final_filtered_by_fos_and_reference.csv")

df['references'] = df['references'].apply(safe_parse_references)


df['year'] = df['year'].astype(int)
df = ensure_column(df, 'is_synthetic', False)

# ======================================================
# 4. Build Graph (SAFE)
# ======================================================
G = nx.DiGraph()

for _, r in df.iterrows():
    G.add_node(r['id'])

for _, r in df.iterrows():
    for ref in r['references']:
        if ref in G:
            G.add_edge(r['id'], ref)

# 🔴 CRITICAL FIX
G.remove_edges_from(nx.selfloop_edges(G))

id_to_idx = {pid: i for i, pid in enumerate(G.nodes())}
idx_to_id = {i: pid for pid, i in id_to_idx.items()}
N = len(id_to_idx)

# ======================================================
# 5. STRUCTURAL FEATURES ONLY (ROBUST)
# ======================================================
def build_structural_features(G, id_to_idx):
    deg_in = dict(G.in_degree())
    deg_out = dict(G.out_degree())
    pagerank = nx.pagerank(G, alpha=0.85)
    clustering = nx.clustering(G.to_undirected())

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

X = build_structural_features(G, id_to_idx)
X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

# ======================================================
# 6. Adjacency Utilities
# ======================================================
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
    indices = torch.from_numpy(
        np.vstack((mx.row, mx.col))
    ).long()
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(indices, values, mx.shape).to(DEVICE)

edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
rows, cols = zip(*edges)
adj_full = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
adj_full = normalize_adj(adj_full)

# ======================================================
# 7. Temporal Segmentation
# ======================================================
DELTA = 61
years = df['year']
segments = []

for t in range(years.min(), years.max(), DELTA):
    ids = df[df['year'] <= t + DELTA]['id']
    idxs = [id_to_idx[i] for i in ids if i in id_to_idx]
    if idxs:
        segments.append((t, t + DELTA, idxs))

# ======================================================
# 8. LLGC Training Function (FIXED)
# ======================================================
def train_llgc(model, X, adj_idx, prev_Z=None, epochs=100, lr=0.01):
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        Z = model(X)

        row, col = adj_idx
        recon = (Z[row] - Z[col]).pow(2).sum(1).mean()
        loss = recon

        if prev_Z is not None:
            loss += 2.0 * (Z[:prev_Z.size(0)] - prev_Z).pow(2).mean()

        deg = torch.bincount(row, minlength=Z.size(0)).float().to(DEVICE)
        deg = (deg - deg.mean()) / (deg.std() + 1e-6)
        loss += 0.2 * torch.mean(torch.abs(torch.norm(Z, dim=1) - deg))

        loss.backward()
        opt.step()

    return model

def safe_parse_references(x):
    """
    Safely parse references field.
    Returns a list or empty list.
    """
    if isinstance(x, list):
        return x

    if not isinstance(x, str):
        return []

    x = x.strip()

    # empty or invalid
    if len(x) == 0 or x[0] != '[' or x[-1] != ']':
        return []

    try:
        parsed = ast.literal_eval(x)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

# ======================================================
# 9. Temporal Training (REAL DATA ONLY)
# ======================================================
prev_Z = None
real_embeddings = None

for _, _, idxs in segments:
    idxs = np.array(idxs)

    adj = adj_full[idxs[:, None], idxs]
    adj_t = to_torch_sparse(adj)

    sg = PageRankAgg(K=5, alpha=0.1, add_self_loops=False).to(DEVICE)
    Xg, _ = sg(X[idxs], adj_t._indices(), adj_t._values())

    model = LLGC(Xg.shape[1], 256, 0.0, True).to(DEVICE)
    model = train_llgc(model, Xg, adj_t._indices(), prev_Z)

    with torch.no_grad():
        prev_Z = model(Xg).cpu()

    real_embeddings = prev_Z.numpy()

# ======================================================
# 10. Train Anomaly Models (NO LEAKAGE)
# ======================================================
iso = IsolationForest(
    n_estimators=300,
    contamination=0.01,
    random_state=SEED
)
iso.fit(real_embeddings)

svm = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
svm.fit(real_embeddings)

# ======================================================
# 11. Inject Synthetic Nodes (SAFE)
# ======================================================
df_fake = pd.read_csv("fakes.csv")
df_fake = ensure_column(df_fake, 'is_synthetic', True)

df = pd.concat([df, df_fake], ignore_index=True)

# rebuild graph
G2 = nx.DiGraph()
for _, r in df.iterrows():
    G2.add_node(r['id'])
for _, r in df.iterrows():
    for ref in safe_parse_references(r['references']):
        if ref in G2:
            G2.add_edge(r['id'], ref)

G2.remove_edges_from(nx.selfloop_edges(G2))

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

# ======================================================
# 12. Detect Fakes
# ======================================================
pred_if = iso.predict(Z2)
pred_svm = svm.predict(Z2)
final_pred = np.where((pred_if == -1) & (pred_svm == -1), -1, 1)

fake_mask = df['is_synthetic'].values
detected = np.sum((final_pred == -1) & fake_mask)

print(f"\n🔥 DETECTED {detected}/{fake_mask.sum()} "
      f"({detected/fake_mask.sum():.2%})")

# ======================================================
# 13. Save Results
# ======================================================
out = pd.DataFrame({
    "paper_id": df['id'],
    "prediction": final_pred,
    "is_synthetic": fake_mask
})

out.to_csv("final_detection_results.csv", index=False)
print("✅ Saved: final_detection_results.csv")
