import pandas as pd
import networkx as nx
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from sklearn.ensemble import IsolationForest
from model import LLGC, PageRankAgg  # Import your LLGC model files

# --------------------------
# Load CSV
# --------------------------
file_name = "C://Users//benzac//Desktop//Project//FinalProject//LLGC-main//final_filtered_by_fos_and_reference.csv"
df = pd.read_csv(file_name)

# --------------------------
# Preprocess references
# --------------------------
df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# --------------------------
# Build graph
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
# Feature extraction
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
# Node mapping
# --------------------------
idx_to_id = {i: node_id for i, node_id in enumerate(G.nodes())}
id_to_idx = {node_id: i for i, node_id in enumerate(G.nodes())}

# --------------------------
# Adjacency matrix
# --------------------------
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

adj_normalized = normalize_adj_sym(adj_matrix)

def sparse_to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

X_tensor = torch.FloatTensor(X_static)
adj_tensor = sparse_to_torch_sparse(adj_normalized)

# --------------------------
# Temporal segmentation
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
# Unsupervised LLGC with temporal prior
# --------------------------
def train_unsupervised_with_prior(model, X_features, adj_indices, adj_values, prev_embeddings=None,
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

# --------------------------
# Anomaly detection
# --------------------------
CONTAMINATION_RATE = 0.01
def run_anomaly_detection(embeddings):
    clf = IsolationForest(contamination=CONTAMINATION_RATE, random_state=42)
    clf.fit(embeddings)
    anomalies = clf.predict(embeddings)
    scores = clf.decision_function(embeddings)
    return scores, anomalies

# --------------------------
# Parameters
# --------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_PROP = 20
ALPHA = 0.1
DROP_OUT = 0.0
USE_BIAS = 1
EPOCHS = 100
LR = 0.01
EMBEDDING_DIM = 64

# --------------------------
# Execution loop
# --------------------------
X_tensor = X_tensor.to(DEVICE)
anomaly_results = {}
prev_Z_t = None

for (t_start, t_end), global_indices in temporal_segments.items():
    if not global_indices:
        continue

    X_t = X_tensor[global_indices]
    N_t = len(global_indices)
    
    global_indices_np = np.array(global_indices)
    adj_segment = adj_matrix_sliceable[global_indices_np[:, None], global_indices_np]
    adj_segment_norm = normalize_adj_sym(adj_segment)
    adj_segment_tensor = sparse_to_torch_sparse(adj_segment_norm).to(DEVICE)
    
    sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
    X_gconv, _ = sgconv(X_t, adj_segment_tensor._indices(), adj_segment_tensor._values())
    
    in_feat = X_gconv.size(1)
    model = LLGC(in_feat, EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
    
    model_trained = train_unsupervised_with_prior(
        model, X_gconv, adj_segment_tensor._indices(),
        adj_segment_tensor._values(), prev_embeddings=prev_Z_t,
        epochs=EPOCHS, lr=LR, device=DEVICE
    )
    
    model_trained.eval()
    Z_t = model_trained(X_gconv).cpu().detach()
    scores, anomalies = run_anomaly_detection(Z_t.numpy())
    
    prev_Z_t = Z_t.clone()
    
    anomaly_results[(t_start, t_end)] = {
        'n_nodes': N_t,
        'n_anomalies': (anomalies == -1).sum(),
        'scores': scores,
        'predictions': anomalies,
        'embeddings': Z_t.numpy(),
        'indices': global_indices
    }
    
    print(f"Segment [{t_start}-{t_end}]: Nodes={N_t}, Anomalies={(anomalies==-1).sum()}")

# --------------------------
# Final summary
# --------------------------
print("\nLLGC Embedding and Anomaly Detection Complete.")
for key, res in anomaly_results.items():
    anomaly_ratio = res['n_anomalies'] / res['n_nodes'] if res['n_nodes'] > 0 else 0
    print(f"[{key[0]}-{key[1]}]: Nodes={res['n_nodes']}, Anomalies={res['n_anomalies']} ({anomaly_ratio:.2%})")


rows_out = []
for (t_start, t_end), res in anomaly_results.items():
    for idx, paper_idx in enumerate(res['indices']):
        paper_id = idx_to_id[paper_idx]
        rows_out.append({
            'paper_id': paper_id,
            't_start': t_start,
            't_end': t_end,
            'anomaly_score': res['scores'][idx],
            'prediction': res['predictions'][idx]
        })

# Create DataFrame and save as CSV
out_df = pd.DataFrame(rows_out)
out_df.to_csv("temporal_anomaly_results.csv", index=False)
print("✅ CSV file saved: temporal_anomaly_results.csv")