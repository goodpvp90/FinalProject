import pandas as pd
import networkx as nx
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import random
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# Import your custom modules
from model import LLGC, PageRankAgg

# --------------------------
# 0. System & Seeds Setup
# --------------------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# 1. Load Data
# --------------------------
file_name = "final_filtered_by_fos_and_reference.csv"
print(f"Loading data from: {file_name}")
df = pd.read_csv(file_name)

# Helper to parse references
df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# --------------------------
# 2. Global Feature Engineering (FIX: Pre-fit everything)
# --------------------------
print("Building consistent feature pipeline (Text + Metadata)...")

# A. Text Features (Title + Abstract)
df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
# max_features=512 provides a good signal-to-noise ratio for hyperbolic embeddings
vectorizer = TfidfVectorizer(stop_words='english', max_features=512)
X_text_all = vectorizer.fit_transform(df['text_combined']).toarray()

# B. Numerical Features (FIX: Clip citations to prevent hyperbolic explosion)
df['n_citation_clipped'] = df['n_citation'].clip(upper=500)
scaler = StandardScaler()
X_num_all = scaler.fit_transform(df[['n_citation_clipped', 'year']])

# C. Categorical Features (Field of Study)
fos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_fos_all = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))

# D. Merge into global feature matrix
X_static = np.hstack((X_text_all, X_num_all, X_fos_all))
X_tensor = torch.FloatTensor(X_static)
print(f"Final feature matrix shape: {X_static.shape}")

# --------------------------
# 3. Graph Construction
# --------------------------
G = nx.DiGraph()
paper_ids = set(df['id'])

for _, row in df.iterrows():
    paper_id = row['id']
    G.add_node(paper_id)

for _, row in df.iterrows():
    citing_id = row['id']
    for cited_id in row['references']:
        if cited_id in paper_ids:
            G.add_edge(citing_id, cited_id)

idx_to_id = {i: node_id for i, node_id in enumerate(G.nodes())}
id_to_idx = {node_id: i for i, node_id in enumerate(G.nodes())}

# Sparse Adjacency Setup
edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
N = G.number_of_nodes()
rows, cols = zip(*edges) if edges else ([], [])
adj_matrix = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N), dtype=np.float32)
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
# 4. Temporal Segmentation
# --------------------------
DELTA_T = 61
years = df['year'].dropna().astype(int)
time_steps = list(range(years.min(), years.max() + DELTA_T, DELTA_T))

temporal_segments = {}
for i in range(len(time_steps)-1):
    t_end = time_steps[i+1]
    current_ids = df[df['year'] <= t_end]['id']
    temporal_segments[(time_steps[i], t_end)] = [id_to_idx[pid] for pid in current_ids if pid in id_to_idx]

# --------------------------
# 5. Training Functions
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
        # Simple contrastive-style loss: keep connected nodes together
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
    anomalies = clf.fit_predict(embeddings)
    scores = clf.decision_function(embeddings)
    return scores, anomalies

# --------------------------
# 6. Main Temporal Loop (Real Data)
# --------------------------
K_PROP, ALPHA, EMBEDDING_DIM = 5, 0.1, 256
all_results_rows = []
prev_Z_t = None

print("\n" + "="*50)
print("Starting Temporal Training Phase")
print("="*50)

for (t_start, t_end), global_indices in temporal_segments.items():
    if not global_indices: continue
    
    X_t = X_tensor[global_indices].to(DEVICE)
    adj_segment = adj_matrix_sliceable[np.array(global_indices)[:, None], np.array(global_indices)]
    adj_tensor = sparse_to_torch_sparse(normalize_adj_sym(adj_segment)).to(DEVICE)

    sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
    X_gconv, _ = sgconv(X_t, adj_tensor._indices(), adj_tensor._values())

    model = LLGC(X_gconv.size(1), EMBEDDING_DIM, 0.0, 1).to(DEVICE)
    model = train_unsupervised_with_prior(model, X_gconv, adj_tensor._indices(), 
                                          prev_embeddings=prev_Z_t, epochs=100, lr=0.01, device=DEVICE)
    
    model.eval()
    Z_t = model(X_gconv).cpu().detach()
    prev_Z_t = Z_t.clone()

    scores, pred = run_anomaly_detection(Z_t.numpy(), contamination_rate=0.01)

    for idx, paper_idx in enumerate(global_indices):
        all_results_rows.append({
            "paper_id": idx_to_id[paper_idx],
            "t_start": t_start, "t_end": t_end,
            "anomaly_score": scores[idx], "prediction": pred[idx], "is_synthetic": False
        })
    print(f"Segment [{t_start}-{t_end}] completed.")

# ---------------------------------------------------------
# 7. Synthetic Injection & Final Detection (FIX: Consistent Pipeline)
# ---------------------------------------------------------
print("\n" + "="*50)
print("Injecting & Detecting Synthetic Nodes")
print("="*50)

# Load fakes and merge
df_fake = pd.read_csv("fakes.csv")
df_fake['is_synthetic'] = True
df_augmented = pd.concat([df, df_fake], ignore_index=True)

# Build augmented features USING THE PRE-FITTED TRANSFORMERS
df_augmented['text_combined'] = df_augmented['title'].fillna('') + ' ' + df_augmented['abstract'].fillna('')
df_augmented['n_citation_clipped'] = df_augmented['n_citation'].clip(upper=500)

X_text_new = vectorizer.transform(df_augmented['text_combined']).toarray()
X_num_new = scaler.transform(df_augmented[['n_citation_clipped', 'year']])
X_fos_new = fos_encoder.transform(df_augmented[['fos.name']].fillna('Unknown'))

X_full_new = torch.tensor(np.hstack((X_text_new, X_num_new, X_fos_new)), dtype=torch.float32).to(DEVICE)

# Adjacency for augmented graph
new_idx_map = {pid: i for i, pid in enumerate(df_augmented['id'])}
rows_aug, cols_aug = [], []
for i, refs in enumerate(df_augmented['references']):
    if isinstance(refs, list):
        for ref in refs:
            if ref in new_idx_map:
                rows_aug += [i, new_idx_map[ref]]
                cols_aug += [new_idx_map[ref], i]

adj_aug = sp.coo_matrix((np.ones(len(rows_aug)), (rows_aug, cols_aug)), 
                        shape=(len(df_augmented), len(df_augmented)), dtype=np.float32)
adj_tensor_aug = sparse_to_torch_sparse(normalize_adj_sym(adj_aug)).to(DEVICE)

# Run prediction
sgconv_final = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv_aug, _ = sgconv_final(X_full_new, adj_tensor_aug._indices(), adj_tensor_aug._values())

model.eval()
with torch.no_grad():
    Z_final = model(X_gconv_aug).cpu().numpy()

# FIX: Set contamination to match (5% injected + 1% baseline)
contamination = 0.06
scores_final, pred_final = run_anomaly_detection(Z_final, contamination_rate=contamination)

# Analyze Synthetic Results
fake_mask = df_augmented['is_synthetic'].fillna(False).values
fake_indices = np.where(fake_mask)[0]
detected = sum(1 for idx in fake_indices if pred_final[idx] == -1)

for idx in fake_indices:
    all_results_rows.append({
        'paper_id': df_augmented.iloc[idx]['id'],
        't_start': "POST_INJECTION", 't_end': "POST_INJECTION",
        'anomaly_score': scores_final[idx], 'prediction': pred_final[idx], 'is_synthetic': True
    })

print(f"\n🔥 Detection Results: {detected}/{len(fake_indices)} ({detected/len(fake_indices):.1%})")
pd.DataFrame(all_results_rows).to_csv("temporal_anomaly_results_final.csv", index=False)