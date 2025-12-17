import pandas as pd
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import random

from model import LLGC, PageRankAgg  # המודל שלך לפי ההגדרות החדשות

# --------------------------
# 0. Set random seeds
# --------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# --------------------------
# 1. Load Data
# --------------------------
file_name = "final_filtered_by_fos_and_reference.csv"
print(f"Loading dataset: {file_name}")
df = pd.read_csv(file_name)

# --------------------------
# 1a. Parse references correctly
# --------------------------
def parse_references(ref):
    if pd.isna(ref):
        return []
    elif isinstance(ref, str):
        ref = ref.strip()
        if ref.startswith("[") and ref.endswith("]"):
            ref = ref[1:-1]
            items = [x.strip().strip("'").strip('"') for x in ref.split(',') if x.strip()]
            return items
        else:
            return [ref.strip()]
    else:
        return []

df['references'] = df['references'].apply(parse_references)

# --------------------------
# 2. Build Graph
# --------------------------
G = nx.DiGraph()
paper_ids = set(df['id'])

for _, row in df.iterrows():
    paper_id = row['id']
    attributes = row.drop(['id','title','authors.name','year','fos.name','n_citation','references','abstract']).to_dict()
    G.add_node(paper_id, **attributes)

for _, row in df.iterrows():
    citing_paper_id = row['id']
    for cited_paper_id in row['references']:
        if cited_paper_id in paper_ids:
            G.add_edge(citing_paper_id, cited_paper_id)

# --------------------------
# 3. Features: STRUCTURE ONLY
# --------------------------
num_nodes = len(df)
structure_dim = 128
X_static = np.random.randn(num_nodes, structure_dim).astype(np.float32)
X_tensor = torch.FloatTensor(X_static)

# --------------------------
# 4. Graph matrices
# --------------------------
idx_to_id = {i: node_id for i, node_id in enumerate(G.nodes())}
id_to_idx = {node_id: i for i, node_id in enumerate(G.nodes())}
edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
N = G.number_of_nodes()
rows = [u for u, v in edges]
cols = [v for u, v in edges]
data = np.ones(len(edges))
adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
adj_matrix_sliceable = adj_matrix.tocsr()

def normalize_adj_sym(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# --------------------------
# 5. Temporal Segmentation
# --------------------------
DELTA_T = 61
years = df['year'].dropna().astype(int)
min_year = years.min()
max_year = years.max()
time_steps = list(range(min_year, max_year + DELTA_T, DELTA_T))
temporal_segments = {}
for i in range(len(time_steps)-1):
    t_start = time_steps[i]
    t_end = time_steps[i+1]
    current_segment_df = df[df['year'] <= t_end]
    node_indices_in_segment = [id_to_idx[id] for id in current_segment_df['id'] if id in id_to_idx]
    temporal_segments[(t_start, t_end)] = node_indices_in_segment

# --------------------------
# 6. Helper functions
# --------------------------
def train_unsupervised_with_prior(model, X_features, adj_indices, prev_embeddings=None,
                                  epochs=100, lr=0.01, device='cpu', temporal_weight=0.5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_features = X_features.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        Z = model(X_features)
        row, col = adj_indices
        loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()
        if prev_embeddings is not None:
            prev_embeddings = prev_embeddings.to(device)
            common_nodes = min(Z.size(0), prev_embeddings.size(0))
            temporal_loss = (Z[:common_nodes] - prev_embeddings[:common_nodes]).pow(2).mean()
            loss += temporal_weight * temporal_loss
        loss.backward()
        optimizer.step()
    return model

def run_anomaly_detection(embeddings, contamination_rate):
    clf = IsolationForest(contamination=contamination_rate, random_state=42)
    clf.fit(embeddings)
    anomalies = clf.predict(embeddings)
    scores = clf.decision_function(embeddings)
    return scores, anomalies

# --------------------------
# 7. Setup
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

# --------------------------
# 8. Main loop: real data only
# --------------------------
for (t_start, t_end), global_indices in temporal_segments.items():
    if not global_indices:
        continue
    X_t = X_tensor[global_indices]
    global_indices_np = np.array(global_indices)
    adj_segment = adj_matrix_sliceable[global_indices_np[:, None], global_indices_np]
    adj_norm_real = normalize_adj_sym(adj_segment)
    adj_tensor_real = sparse_to_torch_sparse(adj_norm_real).to(DEVICE)

    sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
    X_gconv_real, _ = sgconv(X_t, adj_tensor_real._indices(), adj_tensor_real._values())

    model_real = LLGC(X_gconv_real.size(1), EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
    model_real = train_unsupervised_with_prior(
        model_real, X_gconv_real, adj_tensor_real._indices(),
        prev_embeddings=prev_Z_t, epochs=EPOCHS, lr=LR, device=DEVICE
    )

    model_real.eval()
    Z_t_real = model_real(X_gconv_real).cpu().detach()
    prev_Z_t = Z_t_real.clone()

    scores_real, pred_real = run_anomaly_detection(Z_t_real.numpy(), contamination_rate=0.01)

    for idx, paper_idx in enumerate(global_indices):
        all_results_rows.append({
            'paper_id': idx_to_id[paper_idx],
            't_start': t_start, 't_end': t_end,
            'anomaly_score': scores_real[idx],
            'prediction': pred_real[idx],
            'is_synthetic': False
        })

# --------------------------
# 9. Remove real anomalies (core)
# --------------------------
results_df = pd.DataFrame(all_results_rows)
real_results = results_df[results_df["is_synthetic"] == False]
anomalous_real_ids = real_results[real_results["prediction"] == -1]["paper_id"].unique()

df_core = df[~df["id"].isin(anomalous_real_ids)].copy()
print(f"Detected initial anomalies: {len(anomalous_real_ids)}")
print(f"Core size: {len(df_core)}")

# --------------------------
# 10. Inject simple synthetic nodes for validation
# --------------------------
# לדוגמא ניצור 50 nodes עם נושא שונה לגמרי
num_fake = 50
fake_nodes = []
existing_ids = set(df_core["id"])
max_id = max([int(x,16) for x in df_core["id"] if len(x)==24]+[0])
for i in range(num_fake):
    fake_id = hex(max_id+i+1)[2:].rjust(24,'0')
    fake_nodes.append({
        'id': fake_id,
        'title': f'Fake Paper {i}',
        'authors.name': "['Synthetic']",
        'year': 2025,
        'fos.name': 'FakeField',
        'n_citation': 0,
        'references': [],
        'abstract': 'Completely unrelated content',
        'is_synthetic': True
    })

df_core = pd.concat([df_core, pd.DataFrame(fake_nodes)], ignore_index=True)

# --------------------------
# 11. Rebuild features & graph
# --------------------------
num_nodes = len(df_core)
X_static = np.random.randn(num_nodes, structure_dim).astype(np.float32)
X_tensor = torch.FloatTensor(X_static).to(DEVICE)

G = nx.DiGraph()
paper_ids = set(df_core["id"])
for _, row in df_core.iterrows():
    G.add_node(row['id'])
for _, row in df_core.iterrows():
    for ref in row["references"]:
        if ref in paper_ids:
            G.add_edge(row['id'], ref)

idx_to_id = {i: node_id for i, node_id in enumerate(G.nodes())}
id_to_idx = {node_id: i for i, node_id in enumerate(G.nodes())}
edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
rows = [u for u, v in edges]
cols = [v for u, v in edges]
adj_matrix = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes,num_nodes), dtype=np.float32)
adj_norm = normalize_adj_sym(adj_matrix)
adj_tensor = sparse_to_torch_sparse(adj_norm).to(DEVICE)

sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv, _ = sgconv(X_tensor, adj_tensor._indices(), adj_tensor._values())

# --------------------------
# 12. Compute embeddings & detect anomalies
# --------------------------
model_real.eval()
with torch.no_grad():
    Z_final = model_real(X_gconv).cpu().numpy()

fake_mask = df_core['is_synthetic'].fillna(False).astype(bool).values
fake_indices = np.where(fake_mask)[0]
scores, pred = run_anomaly_detection(Z_final, contamination_rate=0.01)

detected_count = sum(pred[fake_indices]==-1)
print(f"Injected anomalies: {len(fake_indices)}")
print(f"Detected anomalies: {detected_count}")
print(f"Detection rate: {detected_count/len(fake_indices)*100:.2f}%")

# --------------------------
# 13. Save results
# --------------------------
for idx in range(len(df_core)):
    all_results_rows.append({
        'paper_id': df_core.iloc[idx]['id'],
        't_start': "POST_INJECTION",
        't_end': "POST_INJECTION",
        'anomaly_score': scores[idx],
        'prediction': pred[idx],
        'is_synthetic': fake_mask[idx]
    })

out_df = pd.DataFrame(all_results_rows)
out_df.to_csv("llgc_full_anomaly_results.csv", index=False)
print("✅ Results saved to llgc_full_anomaly_results.csv")
