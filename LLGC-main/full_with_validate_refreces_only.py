import os
import random
import ast
import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest

# Import your local model classes
from model import LLGC, PageRankAgg

# --------------------------
# 0. STRICT REPRODUCIBILITY SETUP
# --------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# 1. Load & Preprocess Data
# --------------------------
file_name = "final_filtered_by_fos_and_reference.csv"
df = pd.read_csv(file_name)

# CRITICAL: Sort by Hex ID to ensure Row 0 is always the same paper ID
df = df.sort_values('id').reset_index(drop=True)
df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# --------------------------
# 2. Structure-Only Features
# --------------------------
num_nodes = len(df)
structure_dim = 128 
# Row i of X_static matches df.iloc[i] because we sorted df by ID
X_static = np.random.randn(num_nodes, structure_dim).astype(np.float32)
X_tensor = torch.FloatTensor(X_static).to(DEVICE)

# --------------------------
# 3. Global ID Mapping
# --------------------------
all_ids_sorted = sorted(df['id'].tolist())
id_to_idx = {pid: i for i, pid in enumerate(all_ids_sorted)}
idx_to_id = {i: pid for pid, i in id_to_idx.items()}

# Build Adjacency Matrix
edges = []
for i, row in df.iterrows():
    for ref in row['references']:
        if ref in id_to_idx:
            edges.append((i, id_to_idx[ref]))

adj_matrix = sp.coo_matrix((np.ones(len(edges)), ([e[0] for e in edges], [e[1] for e in edges])), 
                           shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()

# --------------------------
# 4. Helper Functions
# --------------------------
def normalize_adj_sym(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return torch.sparse_co_tensor(indices, torch.from_numpy(sparse_mx.data), torch.Size(sparse_mx.shape))

def train_unsupervised(model, X_features, adj_indices, epochs=100, lr=0.01):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        Z = model(X_features)
        row, col = adj_indices
        loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
    return model

# --------------------------
# 5. Clean Real Anomalies
# --------------------------
print("Training on initial graph to find and remove real anomalies...")
adj_norm = normalize_adj_sym(adj_matrix)
adj_t = sparse_to_torch_sparse(adj_norm).to(DEVICE)

# PageRank Aggregation
sgconv = PageRankAgg(K=5, alpha=0.1, add_self_loops=False).to(DEVICE)
X_gconv, _ = sgconv(X_tensor, adj_t._indices(), adj_t._values())

# Lorentzian Model
model = LLGC(structure_dim, 256, 0.0, 1).to(DEVICE)
model = train_unsupervised(model, X_gconv, adj_t._indices())

model.eval()
Z = model(X_gconv).detach().cpu().numpy()
clf = IsolationForest(contamination=0.01, random_state=SEED)
preds = clf.fit_predict(Z)

# Keep only clean real nodes
clean_mask = (preds == 1)
df_clean = df[clean_mask].copy()
X_clean_np = X_static[clean_mask]

print(f"Removed {np.sum(~clean_mask)} initial anomalies. Remaining: {len(df_clean)}")

# --------------------------
# 6. Deterministic Synthetic Injection (FIXED)
# --------------------------
print("\nInjecting Synthetic Nodes (Fixed at 5 edges each)...")

# Generate 5% synthetic nodes
num_fakes = int(0.05 * len(df_clean))
fake_rows = []
# Create new unique IDs for fakes
fake_ids = [f"FAKE_{i:04d}" for i in range(num_fakes)]

# Pool of available clean targets (Sorted for deterministic sampling)
target_pool = sorted(df_clean['id'].tolist())

# Fix seed right before sampling
random.seed(SEED)
for fid in fake_ids:
    # 1. FIXED Edge count: Always 5 random connections
    sampled_refs = random.sample(target_pool, 5)
    fake_rows.append({'id': fid, 'references': sampled_refs, 'is_synthetic': True})

df_fake = pd.DataFrame(fake_rows)
df_final = pd.concat([df_clean, df_fake], ignore_index=True).sort_values('id').reset_index(drop=True)

# 2. FIXED Feature consistency: Synthetic nodes get Gaussian noise, Real nodes keep theirs
X_final_np = np.zeros((len(df_final), structure_dim), dtype=np.float32)
new_id_to_idx = {pid: i for i, pid in enumerate(df_final['id'])}

# Seed for the new synthetic features
np.random.seed(SEED)
for i, pid in enumerate(df_final['id']):
    if "FAKE_" in pid:
        X_final_np[i] = np.random.randn(structure_dim)
    else:
        # Match original feature from the clean set
        orig_idx = id_to_idx[pid]
        X_final_np[i] = X_static[orig_idx]

X_final_tensor = torch.FloatTensor(X_final_np).to(DEVICE)

# --------------------------
# 7. Final Detection
# --------------------------
print("Running final detection on augmented graph...")
final_edges = []
for i, row in df_final.iterrows():
    for ref in row['references']:
        if ref in new_id_to_idx:
            final_edges.append((i, new_id_to_idx[ref]))

adj_final = sp.coo_matrix((np.ones(len(final_edges)), ([e[0] for e in final_edges], [e[1] for e in final_edges])), 
                          shape=(len(df_final), len(df_final)), dtype=np.float32)
adj_final_t = sparse_to_torch_sparse(normalize_adj_sym(adj_final)).to(DEVICE)

X_gconv_final, _ = sgconv(X_final_tensor, adj_final_t._indices(), adj_final_t._values())
model_final = LLGC(structure_dim, 256, 0.0, 1).to(DEVICE)
model_final = train_unsupervised(model_final, X_gconv_final, adj_final_t._indices())

model_final.eval()
Z_final = model_final(X_gconv_final).detach().cpu().numpy()
clf_final = IsolationForest(contamination=0.01, random_state=SEED)
final_preds = clf_final.fit_predict(Z_final)

# Result analysis
df_final['prediction'] = final_preds
fakes = df_final[df_final['is_synthetic'] == True]
detected = fakes[fakes['prediction'] == -1]

print(f"\nResults: Detected {len(detected)} out of {len(fakes)} synthetic anomalies.")
df_final[['id', 'prediction', 'is_synthetic']].to_csv("reproducible_anomaly_results.csv", index=False)
print("✅ Output saved to reproducible_anomaly_results.csv")