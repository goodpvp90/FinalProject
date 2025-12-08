import pandas as pd
import networkx as nx
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import random

from model import LLGC, PageRankAgg 

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# --------------------------
# 1. Load & Preprocess Data
# --------------------------
file_name = "C:\\Users\\nadir\\FinalProject\\LLGC-main\\final_filtered_by_fos_and_reference.csv"
print(f"Loading data from: {file_name}")
df = pd.read_csv(file_name)

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
# 3. Feature Extraction
# --------------------------
df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(df['text_combined']).toarray()

numerical_features = df[['n_citation', 'year']].copy()
scaler = StandardScaler()
X_numerical = scaler.fit_transform(numerical_features)

fos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_fos = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))

df['num_authors'] = df['authors.name'].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) and x != '[]' else 1)
X_authors = scaler.fit_transform(df[['num_authors']])

X_static = np.hstack((X_text, X_numerical, X_fos, X_authors))
print(f"Final static feature matrix shape: {X_static.shape}")

# --------------------------
# 4. Graph Matrices & Tensor Setup
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

X_tensor = torch.FloatTensor(X_static)

# --------------------------
# 5. Temporal Segmentation
# --------------------------
DELTA_T = 5
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

print(f"Total segments created: {len(temporal_segments)}")

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
    clf = IsolationForest(contamination=contamination_rate, random_state=42)
    clf.fit(embeddings)
    anomalies = clf.predict(embeddings)
    scores = clf.decision_function(embeddings)
    return scores, anomalies

import copy

def inject_synthetic_nodes(df, percent_new=0.05, connection_percent=0.30, id_prefix="FAKE_"):
    """
    Inject synthetic rows into the dataframe.
    - percent_new: proportion of new nodes to add (default 5%).
    - connection_percent: proportion of real nodes each fake node connects to (default 30%).
    Returns new dataframe with an 'is_synthetic' boolean column and list-valued 'references'.
    """
    df = df.copy()  # avoid mutating caller's object unexpectedly
    num_original = len(df)
    num_fake = max(1, int(num_original * percent_new))

    print(f"[Injection] Creating {num_fake} synthetic nodes...")

    # Extract valid values from real data
    years = df["year"].dropna().tolist()
    fos_names = df["fos.name"].dropna().tolist()
    citations = df["n_citation"].dropna().tolist()

    if not years:       years = [2000]
    if not fos_names:   fos_names = ["Unknown"]
    if not citations:   citations = [0]

    fake_rows = []

    # ---- Create the fake rows ----
    for i in range(num_fake):
        fake_rows.append({
            "id": f"{id_prefix}{i}",
            "title": "Synthetic Paper Node",
            "authors.name": "[]",
            "year": int(random.choice(years)),
            "fos.name": random.choice(fos_names),
            "n_citation": int(random.choice(citations)),
            "references": [],   # will fill with real ids (list)
            "abstract": "Synthetic auto-generated node",
            "is_synthetic": True
        })

    # Ensure originals are marked
    if 'is_synthetic' not in df.columns:
        df['is_synthetic'] = False

    # ---- Build connections ----
    all_real_ids = df["id"].tolist()
    connections_per_fake = max(1, int(num_original * connection_percent))
    print(f"[Injection] Each fake node will reference {connections_per_fake} real nodes.")

    for fake_row in fake_rows:
        sampled_ids = random.sample(all_real_ids, connections_per_fake)
        # store as list (not string)
        fake_row['references'] = sampled_ids

    df_fake = pd.DataFrame(fake_rows)

    # ---- Merge with original ----
    df_final = pd.concat([df, df_fake], ignore_index=True)

    print("[Injection] Synthetic nodes successfully added.")
    return df_final



# --------------------------
# 7. Parameters & Setup
# --------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K_PROP = 5
ALPHA = 0.1
DROP_OUT = 0.0
USE_BIAS = 1
EPOCHS = 100
LR = 0.01
EMBEDDING_DIM = 64

X_tensor = X_tensor.to(DEVICE)
all_results_rows = [] 
prev_Z_t = None

print("\n" + "="*70)
print(f"Starting Real Temporal Execution (No Synthetic Injection)")
print("="*70)

# --------------------------
# 8. MAIN LOOP (REAL DATA ONLY)
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
            'anomaly_score': scores_real[idx], 'prediction': pred_real[idx],
            'is_synthetic': False
        })

    print(f"Segment [{t_start}-{t_end}] completed (Real Only)")
"""
# ---------------------------------------------------------
# 9. Inject NEW Synthetic Nodes (Feature + Edge Injection)
# ---------------------------------------------------------
print("\n" + "="*70)
print("Injecting Synthetic Nodes (5% new, 30% connections)")
print("="*70)

df = inject_synthetic_nodes(df, percent_new=0.02, connection_percent=0.0 5)

print(f"New dataframe size after injection: {len(df)} rows")

# ---------------------------------------------------------
# 10. Detect Anomalies on Injected Synthetic Nodes
# ---------------------------------------------------------
print("\n" + "="*70)
print("Running Detection on Injected Synthetic Nodes")
print("="*70)

# Rebuild adjacency matrix & features from updated df

# Map paper_id → new index
new_idx_map = {pid: i for i, pid in enumerate(df['id'])}
N = len(df)

# Build feature matrix (TF-IDF or whatever X_tensor source you used)
# IMPORTANT: rebuild from the updated df
texts = df['abstract'].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer(max_features=X_tensor.shape[1])
X_full_new = torch.tensor(vectorizer.fit_transform(texts).todense(), dtype=torch.float32).to(DEVICE)

# Build adjacency from references
rows = []
cols = []

for i, refs in enumerate(df['references']):
    if isinstance(refs, str) and refs.startswith("["):
        try:
            parsed = eval(refs)
            for ref in parsed:
                if ref in new_idx_map:
                    rows.append(i)
                    cols.append(new_idx_map[ref])
                    rows.append(new_idx_map[ref])
                    cols.append(i)
        except:
            pass

adj_new = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(N, N), dtype=np.float32)

# Normalize adjacency
adj_norm_new = normalize_adj_sym(adj_new)
adj_tensor_new = sparse_to_torch_sparse(adj_norm_new).to(DEVICE)

# Run GCN aggregation
sgconv_new = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv_new, _ = sgconv_new(X_full_new, adj_tensor_new._indices(), adj_tensor_new._values())

# Train LLGC from scratch for full graph
model_new = LLGC(X_gconv_new.size(1), EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
model_new = train_unsupervised_with_prior(
    model_new,
    X_gconv_new,
    adj_tensor_new._indices(),
    prev_embeddings=None,
    epochs=EPOCHS,
    lr=LR,
    device=DEVICE
)

model_new.eval()
Z_new = model_new(X_gconv_new).cpu().detach().numpy()

# Identify synthetic nodes
fake_mask = df['is_synthetic'].astype(bool).values
fake_indices = np.where(fake_mask)[0]

# Run anomaly detection with realistic contamination
contamination = max(0.01, len(fake_indices) / len(df))
scores_new, pred_new = run_anomaly_detection(Z_new, contamination_rate=contamination)

# Save results ONLY for injected fakes
detected_count = 0
for idx in fake_indices:
    if pred_new[idx] == -1:
        detected_count += 1

    all_results_rows.append({
        'paper_id': df.iloc[idx]['id'],
        't_start': "POST_INJECTION",
        't_end': "POST_INJECTION",
        'anomaly_score': scores_new[idx],
        'prediction': pred_new[idx],
        'is_synthetic': True
    })

print(f"\n🔥 Injected Fake Detection Results: {detected_count}/{len(fake_indices)} "
      f"({detected_count/len(fake_indices):.1%})")
"""


# --------------------------
# 10. Save Results
# --------------------------
out_df = pd.DataFrame(all_results_rows)
out_df.to_csv("temporal_anomaly_results_real_only.csv", index=False)

print("\n✅ CSV saved: temporal_anomaly_results_real_only.csv")
