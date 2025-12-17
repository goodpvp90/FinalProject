import os
import random
import ast
import copy
import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# Import your local model classes
from model import LLGC, PageRankAgg

# --------------------------
# 0. STRICT REPRODUCIBILITY SETUP
# --------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Critical for GPU determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------
# 1. Load & Preprocess Data
# --------------------------
file_name = "final_filtered_by_fos_and_reference.csv"
print(f"Loading data from: {file_name}")
df = pd.read_csv(file_name)

# FIX 1: Sort by Hex ID immediately. This ensures that 'Index 0' 
# always refers to the same paper, regardless of the CSV line order.
df = df.sort_values('id').reset_index(drop=True)

df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

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
# 3. Feature Extraction (STRUCTURE ONLY)
# --------------------------
print("Building STRUCTURE-ONLY features...")
num_nodes = len(df)
structure_dim = 128 

# Use the fixed seed for the random feature matrix
X_static = np.random.randn(num_nodes, structure_dim).astype(np.float32)
print(f"Feature matrix shape: {X_static.shape}")

# --------------------------
# 4. Graph Matrices & Tensor Setup
# --------------------------
# FIX 2: Ensure ID mappings are derived from a sorted order
all_ids_sorted = sorted(list(G.nodes()))
id_to_idx = {node_id: i for i, node_id in enumerate(all_ids_sorted)}
idx_to_id = {i: node_id for i, node_id in enumerate(all_ids_sorted)}

edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
N = G.number_of_nodes()
rows = [u for u, v in edges]
cols = [v for u, v in edges]
edge_data = np.ones(len(edges))

adj_matrix = sp.coo_matrix((edge_data, (rows, cols)), shape=(N, N), dtype=np.float32)
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

X_tensor = torch.FloatTensor(X_static)

# --------------------------
# 5. Temporal Segmentation
# --------------------------
DELTA_T = 61
years = df['year'].dropna().astype(int)
time_steps = list(range(years.min(), years.max() + DELTA_T, DELTA_T))

temporal_segments = {}
for i in range(len(time_steps)-1):
    t_start, t_end = time_steps[i], time_steps[i+1]
    current_segment_df = df[df['year'] <= t_end]
    # FIX 3: Ensure indices are extracted in a sorted manner
    node_indices_in_segment = sorted([id_to_idx[pid] for pid in current_segment_df['id'] if pid in id_to_idx])
    temporal_segments[(t_start, t_end)] = node_indices_in_segment

# --------------------------
# 6. Helper Functions
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
    # FIXED: random_state is set to SEED
    clf = IsolationForest(contamination=contamination_rate, random_state=SEED)
    clf.fit(embeddings)
    anomalies = clf.predict(embeddings)
    scores = clf.decision_function(embeddings)
    return scores, anomalies

def inject_synthetic_nodes_from_csv(df_orig, fakes_csv="fakes.csv"):
    df_fake = pd.read_csv(fakes_csv)
    if 'is_synthetic' not in df_fake.columns:
        df_fake['is_synthetic'] = True
    df_final = pd.concat([df_orig, df_fake], ignore_index=True)
    # FIX 4: Sort final DF by ID again to keep ordering stable
    df_final = df_final.sort_values('id').reset_index(drop=True)
    return df_final

# --------------------------
# 7. Parameters & Setup
# --------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_PROP, ALPHA, DROP_OUT, USE_BIAS = 5, 0.1, 0.0, 1
EPOCHS, LR, EMBEDDING_DIM = 100, 0.01, 256

X_tensor = X_tensor.to(DEVICE)
all_results_rows = []
prev_Z_t = None

# --------------------------
# 8. MAIN LOOP (REAL DATA)
# --------------------------
for (t_start, t_end), global_indices in temporal_segments.items():
    if not global_indices: continue

    X_t = X_tensor[global_indices]
    global_indices_np = np.array(global_indices)
    adj_segment = adj_matrix_sliceable[global_indices_np[:, None], global_indices_np]

    adj_norm_real = normalize_adj_sym(adj_segment)
    adj_tensor_real = sparse_to_torch_sparse(adj_norm_real).to(DEVICE)

    sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
    X_gconv_real, _ = sgconv(X_t, adj_tensor_real._indices(), adj_tensor_real._values())

    model_real = LLGC(X_gconv_real.size(1), EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
    model_real = train_unsupervised_with_prior(model_real, X_gconv_real, adj_tensor_real._indices(),
                                              prev_embeddings=prev_Z_t, epochs=EPOCHS, lr=LR, device=DEVICE)

    model_real.eval()
    Z_t_real = model_real(X_gconv_real).cpu().detach()
    prev_Z_t = Z_t_real.clone()

    scores_real, pred_real = run_anomaly_detection(Z_t_real.numpy(), contamination_rate=0.01)

    for idx, paper_idx in enumerate(global_indices):
        all_results_rows.append({
            'paper_id': idx_to_id[paper_idx],
            't_start': t_start, 't_end': t_end,
            'anomaly_score': scores_real[idx], 'prediction': pred_real[idx],
            'is_synthetic': False
        })

# ---------------------------------------------------------
# 8.5 Remove Real Anomalies & Recompute Embeddings
# ---------------------------------------------------------
results_df = pd.DataFrame(all_results_rows)
anomalous_real_ids = results_df[(results_df["is_synthetic"] == False) & (results_df["prediction"] == -1)]["paper_id"].unique()

df_clean = df[~df["id"].isin(anomalous_real_ids)].copy()
# FIX 5: Use sorted list for clean IDs to avoid set iteration randomness
paper_ids_clean = sorted(list(df_clean["id"].unique()))

G_clean = nx.DiGraph()
for pid in paper_ids_clean: G_clean.add_node(pid)
for _, row in df_clean.iterrows():
    for ref in row["references"]:
        if ref in paper_ids_clean: G_clean.add_edge(row["id"], ref)

id_to_idx_clean = {pid: i for i, pid in enumerate(paper_ids_clean)}
clean_indices = [id_to_idx[pid] for pid in paper_ids_clean] # Use original id_to_idx to get correct features
X_clean = X_tensor[clean_indices]

edges_clean = [(id_to_idx_clean[u], id_to_idx_clean[v]) for u, v in G_clean.edges()]
rows_c, cols_c = zip(*edges_clean) if edges_clean else ([], [])
adj_clean = sp.coo_matrix((np.ones(len(rows_c)), (rows_c, cols_c)), shape=(len(df_clean), len(df_clean)), dtype=np.float32)
adj_tensor_clean = sparse_to_torch_sparse(normalize_adj_sym(adj_clean)).to(DEVICE)

sgconv_clean = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv_clean, _ = sgconv_clean(X_clean, adj_tensor_clean._indices(), adj_tensor_clean._values())

model_clean = LLGC(X_gconv_clean.size(1), EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
model_clean = train_unsupervised_with_prior(model_clean, X_gconv_clean, adj_tensor_clean._indices(), epochs=EPOCHS, lr=LR, device=DEVICE)

# ---------------------------------------------------------
# 9 & 10. Inject and Detect
# ---------------------------------------------------------
df = inject_synthetic_nodes_from_csv(df_clean, fakes_csv="fakes.csv")
# FIX 6: Map is built from sorted final dataframe
new_idx_map = {pid: i for i, pid in enumerate(df['id'])}
N_new = len(df)

# FIX 7: TF-IDF texts must follow the EXACT order of the IDs in the map
texts = [df[df['id'] == pid]['abstract'].fillna("").values[0] for pid in df['id']]
vectorizer = TfidfVectorizer(max_features=X_tensor.shape[1])
X_full_new = torch.tensor(vectorizer.fit_transform(texts).toarray().astype(np.float32)).to(DEVICE)

rows_n, cols_n = [], []
for i, row in df.iterrows():
    for ref in row['references']:
        if ref in new_idx_map:
            rows_n += [i, new_idx_map[ref]]
            cols_n += [new_idx_map[ref], i]

adj_new = sp.coo_matrix((np.ones(len(rows_n)), (rows_n, cols_n)), shape=(N_new, N_new), dtype=np.float32)
adj_tensor_new = sparse_to_torch_sparse(normalize_adj_sym(adj_new)).to(DEVICE)

X_gconv_new, _ = sgconv_clean(X_full_new, adj_tensor_new._indices(), adj_tensor_new._values())
model_clean.eval()
with torch.no_grad():
    Z_new = model_clean(X_gconv_new).cpu().numpy()

scores_new, pred_new = run_anomaly_detection(Z_new, contamination_rate=0.01)

# Result assembly
final_rows = []
for idx, row in df.iterrows():
    final_rows.append({
        'paper_id': row['id'],
        'anomaly_score': scores_new[idx],
        'prediction': pred_new[idx],
        'is_synthetic': row.get('is_synthetic', False)
    })

pd.DataFrame(final_rows).to_csv("reproducible_results.csv", index=False)
print("\n✅ Results saved to reproducible_results.csv")