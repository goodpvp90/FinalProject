import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import random
from copy import deepcopy

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
# 2. Build Graph
# =========================================================
def build_graph(df):
    G = nx.Graph()
    ids = set(df["id"])
    for pid in ids:
        G.add_node(pid)
    for _, row in df.iterrows():
        for ref in row["references"]:
            if ref in ids:
                G.add_edge(row["id"], ref)
    return G

# =========================================================
# 3. Structural Features
# =========================================================
def build_features(G):
    node_list = list(G.nodes())
    deg = dict(G.degree())
    pagerank = nx.pagerank(G)
    clustering = nx.clustering(G)

    X = []
    for n in node_list:
        X.append([
            deg.get(n, 0),
            pagerank.get(n, 0),
            clustering.get(n, 0)
        ])

    X = np.array(X, dtype=np.float32)
    X = StandardScaler().fit_transform(X)
    return node_list, torch.tensor(X, dtype=torch.float32).to(DEVICE)

# =========================================================
# 4. Edge Index
# =========================================================
def build_edge_index(G, node_list):
    id2idx = {n: i for i, n in enumerate(node_list)}
    edges = []
    for u, v in G.edges():
        edges.append([id2idx[u], id2idx[v]])
        edges.append([id2idx[v], id2idx[u]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)

# =========================================================
# 5. Train LLGC
# =========================================================
def train_llgc(X_gconv, edge_index, emb_dim=128, epochs=100):
    model = LLGC(
        nfeat=X_gconv.size(1),
        nclass=emb_dim,
        drop_out=0.0,
        use_bias=True
    ).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for _ in range(epochs):
        opt.zero_grad()
        Z = model(X_gconv)
        r, c = edge_index
        loss = (Z[r] - Z[c]).pow(2).sum(dim=1).mean()
        loss.backward()
        opt.step()

    return model

# =========================================================
# 6. Detect Anomalies
# =========================================================
def detect_anomalies(Z, contamination):
    iso = IsolationForest(
        contamination=contamination,
        random_state=SEED
    )
    pred = iso.fit_predict(Z)
    return pred

# =========================================================
# 7. Inject Synthetic Anomalies
# =========================================================
def inject_anomalies(df, G, ratio=0.05, min_deg=1, max_deg=3):
    """
    Inject structurally anomalous nodes.
    Works with ANY type of node IDs (int / str / mixed).
    """
    df_new = df.copy()
    G_new = G.copy()

    n_fake = int(len(df) * ratio)
    fake_ids = []

    existing_ids = set(df["id"].astype(str))

    nodes = list(G.nodes())

    for i in range(n_fake):
        fake_id = f"FAKE_{i}"
        while fake_id in existing_ids:
            fake_id = f"FAKE_{i}_{random.randint(0,9999)}"

        fake_ids.append(fake_id)
        existing_ids.add(fake_id)

        # low-degree random attachment → structural anomaly
        deg = random.randint(min_deg, max_deg)
        targets = random.sample(nodes, deg)

        G_new.add_node(fake_id)
        for t in targets:
            G_new.add_edge(fake_id, t)

        df_new = pd.concat([
            df_new,
            pd.DataFrame([{
                "id": fake_id,
                "references": targets,
                "is_synthetic": True
            }])
        ], ignore_index=True)

    df_new["is_synthetic"] = df_new.get("is_synthetic", False).fillna(False)
    return df_new, G_new, fake_ids

# =========================================================
# 8. MAIN PIPELINE
# =========================================================
print("\n=== INITIAL TRAINING ===")

G = build_graph(df)
nodes, X = build_features(G)
edge_index = build_edge_index(G, nodes)

sgc = PageRankAgg(K=5, alpha=0.1).to(DEVICE)
X_gconv, _ = sgc(X, edge_index)

model = train_llgc(X_gconv, edge_index)

Z = model(X_gconv).detach().cpu().numpy()
pred = detect_anomalies(Z, contamination=0.01)

anomalous_ids = [nodes[i] for i, p in enumerate(pred) if p == -1]
print(f"Detected initial anomalies: {len(anomalous_ids)}")

# =========================================================
# 9. CLEAN → CORE GRAPH
# =========================================================
df_core = df[~df["id"].isin(anomalous_ids)]
G_core = build_graph(df_core)

print(f"Core size after cleaning: {len(df_core)}")

nodes_c, X_c = build_features(G_core)
edge_index_c = build_edge_index(G_core, nodes_c)
Xc_gconv, _ = sgc(X_c, edge_index_c)

model_core = train_llgc(Xc_gconv, edge_index_c)

# =========================================================
# 10. INJECT SYNTHETIC ANOMALIES
# =========================================================
print("\n=== INJECTING SYNTHETIC ANOMALIES ===")

df_injected, G_injected, fake_ids = inject_anomalies(
    df_core, G_core, ratio=0.05
)

nodes_i, Xi = build_features(G_injected)
edge_index_i = build_edge_index(G_injected, nodes_i)
Xi_gconv, _ = sgc(Xi, edge_index_i)

Zi = model_core(Xi_gconv).detach().cpu().numpy()
pred_i = detect_anomalies(Zi, contamination=len(fake_ids)/len(Zi))

# =========================================================
# 11. VALIDATION
# =========================================================
idx_map = {n: i for i, n in enumerate(nodes_i)}

detected = 0
for fid in fake_ids:
    if pred_i[idx_map[fid]] == -1:
        detected += 1

print("\n=== VALIDATION RESULTS ===")
print(f"Injected anomalies: {len(fake_ids)}")
print(f"Detected anomalies: {detected}")
print(f"Detection rate: {detected / len(fake_ids):.2%}")
