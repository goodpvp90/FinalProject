import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import random

from model import LLGC, PageRankAgg

# =========================================================
# 0. Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 1. Load Data
# =========================================================
df = pd.read_csv("final_filtered_by_fos_and_reference.csv")
df["references"] = df["references"].apply(
    lambda x: eval(x) if isinstance(x, str) else []
)

# =========================================================
# 2. Build Graph (DIRECTED → UNDIRECTED)
# =========================================================
G = nx.Graph()

paper_ids = set(df["id"])

for pid in paper_ids:
    G.add_node(pid)

for _, row in df.iterrows():
    for ref in row["references"]:
        if ref in paper_ids:
            G.add_edge(row["id"], ref)

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# =========================================================
# 3. STRUCTURAL FEATURES (DETERMINISTIC)
# =========================================================
print("Building structural features...")

deg = dict(G.degree())
pagerank = nx.pagerank(G, alpha=0.85)
clustering = nx.clustering(G)

X = []
node_list = list(G.nodes())

for n in node_list:
    X.append([
        deg.get(n, 0),
        pagerank.get(n, 0),
        clustering.get(n, 0)
    ])

X = np.array(X, dtype=np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

print("Feature matrix shape:", X_tensor.shape)

# =========================================================
# 4. Adjacency Matrix (CONSISTENT)
# =========================================================
adj = nx.to_scipy_sparse_array(
    G,
    nodelist=node_list,
    dtype=np.float32
)

# make sure symmetric + self loops
adj = adj + adj.T
adj[adj > 1] = 1
adj = adj + sp.eye(adj.shape[0])

def normalize_adj(adj):
    deg = np.array(adj.sum(1)).flatten()
    deg_inv = np.power(deg, -1)
    deg_inv[np.isinf(deg_inv)] = 0.0
    D_inv = sp.diags(deg_inv)
    return D_inv @ adj

adj_norm = normalize_adj(adj)

def to_torch_sparse(mx):
    mx = mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((mx.row, mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(DEVICE)

adj_tensor = to_torch_sparse(adj_norm)

# =========================================================
# 5. Graph Aggregation (PageRankAgg)
# =========================================================
sgconv = PageRankAgg(
    K=5,
    alpha=0.1,
    add_self_loops=False
).to(DEVICE)

X_gconv, _ = sgconv(
    X_tensor,
    adj_tensor._indices(),
    adj_tensor._values()
)

# =========================================================
# 6. Train LLGC (UNSUPERVISED)
# =========================================================
EMB_DIM = 128
EPOCHS = 100
LR = 0.01

model = LLGC(
    X_gconv.size(1),
    EMB_DIM,
    dropout=0.0,
    bias=True
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    Z = model(X_gconv)

    row, col = adj_tensor._indices()
    loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()

    loss.backward()
    optimizer.step()

print("Initial training completed")

# =========================================================
# 7. Anomaly Detection (CORE EXTRACTION)
# =========================================================
model.eval()
with torch.no_grad():
    Z = model(X_gconv).cpu().numpy()

contamination = max(0.01, 1.0 / len(Z))
iso = IsolationForest(
    contamination=contamination,
    random_state=SEED
)
pred = iso.fit_predict(Z)
scores = iso.decision_function(Z)

anomalous_nodes = [
    node_list[i] for i in range(len(pred)) if pred[i] == -1
]

print(f"Detected anomalies: {len(anomalous_nodes)}")

# =========================================================
# 8. REMOVE ANOMALIES → CORE GRAPH
# =========================================================
df_core = df[~df["id"].isin(anomalous_nodes)].copy()

print(f"Core dataset size: {len(df_core)}")

# rebuild graph
G_core = nx.Graph()
core_ids = set(df_core["id"])

for pid in core_ids:
    G_core.add_node(pid)

for _, row in df_core.iterrows():
    for ref in row["references"]:
        if ref in core_ids:
            G_core.add_edge(row["id"], ref)

# =========================================================
# 9. RE-TRAIN ON CORE
# =========================================================
node_list_core = list(G_core.nodes())

deg = dict(G_core.degree())
pagerank = nx.pagerank(G_core)
clustering = nx.clustering(G_core)

X_core = []
for n in node_list_core:
    X_core.append([
        deg.get(n, 0),
        pagerank.get(n, 0),
        clustering.get(n, 0)
    ])

X_core = scaler.fit_transform(
    np.array(X_core, dtype=np.float32)
)

X_core = torch.tensor(X_core).to(DEVICE)

adj_core = nx.to_scipy_sparse_array(
    G_core,
    nodelist=node_list_core,
    dtype=np.float32
)
adj_core = adj_core + adj_core.T
adj_core[adj_core > 1] = 1
adj_core = adj_core + sp.eye(adj_core.shape[0])

adj_core = normalize_adj(adj_core)
adj_core = to_torch_sparse(adj_core)

X_gconv_core, _ = sgconv(
    X_core,
    adj_core._indices(),
    adj_core._values()
)

model_core = LLGC(
    X_gconv_core.size(1),
    EMB_DIM,
    0.0,
    True
).to(DEVICE)

optimizer = optim.Adam(model_core.parameters(), lr=LR)

model_core.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    Zc = model_core(X_gconv_core)
    r, c = adj_core._indices()
    loss = (Zc[r] - Zc[c]).pow(2).sum(dim=1).mean()
    loss.backward()
    optimizer.step()

print("Core model ready")
