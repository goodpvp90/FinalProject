import pandas as pd
import numpy as np
import networkx as nx
import torch
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
# 2. Build Graph (UNDIRECTED, CLEAN)
# =========================================================
G = nx.Graph()
paper_ids = set(df["id"])

for pid in paper_ids:
    G.add_node(pid)

for _, row in df.iterrows():
    for ref in row["references"]:
        if ref in paper_ids:
            G.add_edge(row["id"], ref)

node_list = list(G.nodes())
id_to_idx = {n: i for i, n in enumerate(node_list)}

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# =========================================================
# 3. STRUCTURAL FEATURES (DETERMINISTIC)
# =========================================================
print("Building structural features...")

deg = dict(G.degree())
pagerank = nx.pagerank(G, alpha=0.85)
clustering = nx.clustering(G)

X = []
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
# 4. EDGE INDEX (PyG FORMAT)
# =========================================================
edges = []
for u, v in G.edges():
    edges.append([id_to_idx[u], id_to_idx[v]])
    edges.append([id_to_idx[v], id_to_idx[u]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)

# =========================================================
# 5. PageRankAgg (NO MANUAL NORMALIZATION)
# =========================================================
sgconv = PageRankAgg(
    K=5,
    alpha=0.1,
    add_self_loops=True,
    normalize=True
).to(DEVICE)

X_gconv, _ = sgconv(X_tensor, edge_index)

# =========================================================
# 6. Train LLGC (UNSUPERVISED)
# =========================================================
EMB_DIM = 128
EPOCHS = 100
LR = 0.01
DROP_OUT = 0.0
USE_BIAS = True

model = LLGC(
    nfeat=X_gconv.size(1),
    nclass=EMB_DIM,
    drop_out=DROP_OUT,
    use_bias=USE_BIAS
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    Z = model(X_gconv)

    row, col = edge_index
    loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()

    loss.backward()
    optimizer.step()

print("Initial training completed")

# =========================================================
# 7. Anomaly Detection
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
# 8. CORE GRAPH (REMOVE ANOMALIES)
# =========================================================
df_core = df[~df["id"].isin(anomalous_nodes)].copy()
print(f"Core dataset size: {len(df_core)}")

# =========================================================
# 9. REBUILD GRAPH + RETRAIN CORE MODEL
# =========================================================
G_core = nx.Graph()
core_ids = set(df_core["id"])

for pid in core_ids:
    G_core.add_node(pid)

for _, row in df_core.iterrows():
    for ref in row["references"]:
        if ref in core_ids:
            G_core.add_edge(row["id"], ref)

node_list_core = list(G_core.nodes())
id_to_idx_core = {n: i for i, n in enumerate(node_list_core)}

deg = dict(G_core.degree())
pagerank = nx.pagerank(G_core)
clustering = nx.clustering(G_core)

Xc = []
for n in node_list_core:
    Xc.append([
        deg.get(n, 0),
        pagerank.get(n, 0),
        clustering.get(n, 0)
    ])

Xc = scaler.fit_transform(np.array(Xc, dtype=np.float32))
Xc = torch.tensor(Xc, dtype=torch.float32).to(DEVICE)

edges_core = []
for u, v in G_core.edges():
    edges_core.append([id_to_idx_core[u], id_to_idx_core[v]])
    edges_core.append([id_to_idx_core[v], id_to_idx_core[u]])

edge_index_core = torch.tensor(edges_core, dtype=torch.long).t().contiguous().to(DEVICE)

Xc_gconv, _ = sgconv(Xc, edge_index_core)

model_core = LLGC(
    nfeat=Xc_gconv.size(1),
    nclass=EMB_DIM,
    drop_out=DROP_OUT,
    use_bias=USE_BIAS
).to(DEVICE)

optimizer = optim.Adam(model_core.parameters(), lr=LR)

model_core.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    Zc = model_core(Xc_gconv)
    r, c = edge_index_core
    loss = (Zc[r] - Zc[c]).pow(2).sum(dim=1).mean()
    loss.backward()
    optimizer.step()

print("✔ Core model ready")
