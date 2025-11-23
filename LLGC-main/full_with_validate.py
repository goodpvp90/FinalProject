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
import random

# Import your LLGC model files
# Ensure LLGC-main/model.py is in the same directory or python path
from model import LLGC, PageRankAgg 

# --------------------------
# 1. Load & Preprocess Data
# --------------------------
# Using the specific path you provided
file_name = "C://Users//benzac//Desktop//Project//FinalProject//LLGC-main//final_filtered_by_fos_and_reference.csv"
print(f"Loading data from: {file_name}")
df = pd.read_csv(file_name)

# Convert string representation of lists to actual lists
df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# --------------------------
# 2. Build Graph
# --------------------------
G = nx.DiGraph()
paper_ids = set(df['id'])

# Add Nodes
for _, row in df.iterrows():
    paper_id = row['id']
    # Drop columns that are not attributes or processed later
    attributes = row.drop(['id','title','authors.name','year','fos.name','n_citation','references','abstract']).to_dict()
    G.add_node(paper_id, **attributes)

# Add Edges
for _, row in df.iterrows():
    citing_paper_id = row['id']
    for cited_paper_id in row['references']:
        if cited_paper_id in paper_ids:
            G.add_edge(citing_paper_id, cited_paper_id)

# --------------------------
# 3. Feature Extraction
# --------------------------
# A. Text (Title + Abstract)
df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(df['text_combined']).toarray()

# B. Numerical (Citation + Year)
numerical_features = df[['n_citation', 'year']].copy()
scaler = StandardScaler()
X_numerical = scaler.fit_transform(numerical_features)

# C. Categorical (FOS)
fos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_fos = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))

# D. Authorship (Count)
df['num_authors'] = df['authors.name'].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) and x != '[]' else 1)
X_authors = scaler.fit_transform(df[['num_authors']])

# Combine all
X_static = np.hstack((X_text, X_numerical, X_fos, X_authors))
print(f"Final static feature matrix shape: {X_static.shape}")

# --------------------------
# 4. Graph Matrices & Tensor Setup
# --------------------------
idx_to_id = {i: node_id for i, node_id in enumerate(G.nodes())}
id_to_idx = {node_id: i for i, node_id in enumerate(G.nodes())}

# Create Adjacency Matrix
edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
N = G.number_of_nodes()
rows = [u for u, v in edges]
cols = [v for u, v in edges]
data = np.ones(len(edges))

adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
# Convert to CSR for efficient slicing in the loop
adj_matrix_sliceable = adj_matrix.tocsr()

def normalize_adj_sym(adj):
    """Symmetrically normalize adjacency matrix (D^-0.5 * A * D^-0.5)."""
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_torch_sparse(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
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
    # Cumulative: all papers up to t_end
    current_segment_df = df[df['year'] <= t_end]
    node_indices_in_segment = [id_to_idx[id] for id in current_segment_df['id'] if id in id_to_idx]
    temporal_segments[(t_start, t_end)] = node_indices_in_segment

print(f"Total segments created: {len(temporal_segments)}")

# --------------------------
# 6. Helper Functions
# --------------------------

def train_unsupervised_with_prior(model, X_features, adj_indices, prev_embeddings=None,
                                  epochs=100, lr=0.01, device='cpu', temporal_weight=0.5):
    """
    Trains LLGC with Graph Loss (neighbors) and Temporal Loss (history).
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_features = X_features.to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass: Euclidean -> Hyperbolic -> Tangent Space (Z)
        Z = model(X_features)
        
        # 1. Graph Loss: Minimize distance between connected nodes
        row, col = adj_indices
        loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()
        
        # 2. Temporal Loss: Minimize distance from previous time step (smoothness)
        if prev_embeddings is not None:
            prev_embeddings = prev_embeddings.to(device)
            # Only compare nodes that exist in both (the common set)
            common_nodes = min(Z.size(0), prev_embeddings.size(0))
            temporal_loss = (Z[:common_nodes] - prev_embeddings[:common_nodes]).pow(2).mean()
            loss += temporal_weight * temporal_loss
            
        loss.backward()
        optimizer.step()
    return model

def run_anomaly_detection(embeddings, contamination_rate):
    """Runs Isolation Forest with a specific contamination rate."""
    clf = IsolationForest(contamination=contamination_rate, random_state=42)
    clf.fit(embeddings)
    anomalies = clf.predict(embeddings) # -1: Anomaly, 1: Normal
    scores = clf.decision_function(embeddings) # Lower score = More anomalous
    return scores, anomalies

def inject_synthetic_anomalies(X, adj, injection_ratio=0.05, connection_ratio=0.30):
    """
    Injects synthetic anomalies strictly following Project Document Section 5.1.
    1. Adds 'injection_ratio' (5%) extra nodes.
    2. Connects each new node to 'connection_ratio' (30%) of existing nodes.
    """
    num_real_nodes, num_features = X.shape
    
    # 1. Calculate number of fakes (at least 1)
    n_injected = max(1, int(num_real_nodes * injection_ratio))
    
    # 2. Generate Noisy Features (Gaussian noise, shifted to be distinct)
    noise_level = 3.0
    X_noise = torch.randn(n_injected, num_features) * noise_level + noise_level
    X_new = torch.cat([X, X_noise], dim=0)
    
    # 3. Generate Edges (Connect to 30% of existing nodes)
    n_neighbors = max(1, int(num_real_nodes * connection_ratio))
    
    new_rows = []
    new_cols = []
    fake_indices = list(range(num_real_nodes, num_real_nodes + n_injected))
    
    for fake_idx in fake_indices:
        # Sample random real nodes to connect to
        targets = random.sample(range(num_real_nodes), min(num_real_nodes, n_neighbors))
        for t in targets:
            # Bidirectional
            new_rows.append(fake_idx); new_cols.append(t) 
            new_rows.append(t); new_cols.append(fake_idx)
            
    # Merge edges
    new_data = np.ones(len(new_rows))
    adj_new_edges = sp.coo_matrix((new_data, (new_rows, new_cols)), 
                                  shape=(num_real_nodes + n_injected, num_real_nodes + n_injected), 
                                  dtype=np.float32)
    
    # Resize original and add
    adj.resize((num_real_nodes + n_injected, num_real_nodes + n_injected))
    adj_new = adj.tocsr() + adj_new_edges.tocsr()
    
    return X_new, adj_new, fake_indices, n_injected

# --------------------------
# 7. Parameters & Setup
# --------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Hyperparameters
K_PROP = 20           # Propagation steps
ALPHA = 0.1           # Teleport probability
DROP_OUT = 0.0
USE_BIAS = 1
EPOCHS = 100
LR = 0.01
EMBEDDING_DIM = 64

# Validation Settings (from Document)
INJECTION_RATIO = 0.05   # 5%
CONNECTION_RATIO = 0.30  # 30%

# Prepare Execution
X_tensor = X_tensor.to(DEVICE)
all_results_rows = [] 
prev_Z_t = None

print("\n" + "="*70)
print(f"Starting Dual Execution: Real Data + Synthetic Validation")
print(f"Validation Config: {INJECTION_RATIO*100}% Injection, {CONNECTION_RATIO*100}% Connectivity")
print("="*70)

# --------------------------
# 8. Main Execution Loop
# --------------------------
for (t_start, t_end), global_indices in temporal_segments.items():
    if not global_indices:
        continue

    # --- Common Data Prep ---
    X_t = X_tensor[global_indices]
    N_real = len(global_indices)
    global_indices_np = np.array(global_indices)
    # Slice Adjacency
    adj_segment = adj_matrix_sliceable[global_indices_np[:, None], global_indices_np]

    # ============================================================
    # PASS 1: REAL DATA (The "Clean" Run)
    # ============================================================
    
    # Prepare Real Graph
    adj_norm_real = normalize_adj_sym(adj_segment)
    adj_tensor_real = sparse_to_torch_sparse(adj_norm_real).to(DEVICE)
    
    # PageRank Aggregation (SGC step)
    sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
    X_gconv_real, _ = sgconv(X_t, adj_tensor_real._indices(), adj_tensor_real._values())
    
    # Initialize & Train Model
    model_real = LLGC(X_gconv_real.size(1), EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
    model_real = train_unsupervised_with_prior(
        model_real, X_gconv_real, adj_tensor_real._indices(), 
        prev_embeddings=prev_Z_t, epochs=EPOCHS, lr=LR, device=DEVICE
    )
    
    # Extract Embeddings
    model_real.eval()
    Z_t_real = model_real(X_gconv_real).cpu().detach()
    
    # Save Clean History for next segment
    prev_Z_t = Z_t_real.clone()
    
    # Detect on Real Data (Assume ~1% anomalies for real data)
    scores_real, pred_real = run_anomaly_detection(Z_t_real.numpy(), contamination_rate=0.01)
    
    # Store Real Results
    for idx, paper_idx in enumerate(global_indices):
        all_results_rows.append({
            'paper_id': idx_to_id[paper_idx],
            't_start': t_start, 't_end': t_end,
            'anomaly_score': scores_real[idx], 'prediction': pred_real[idx],
            'is_synthetic': False
        })

    # ============================================================
    # PASS 2: SYNTHETIC VALIDATION (The "Dirty" Run)
    # ============================================================
    
    # Inject Fakes based on Document Ratios
    X_t_fake, adj_fake, fake_indices, n_injected = inject_synthetic_anomalies(
        X_t.cpu(), adj_segment.copy(), 
        injection_ratio=INJECTION_RATIO, 
        connection_ratio=CONNECTION_RATIO
    )
    X_t_fake = X_t_fake.to(DEVICE)
    
    # Prepare Fake Graph
    adj_norm_fake = normalize_adj_sym(adj_fake)
    adj_tensor_fake = sparse_to_torch_sparse(adj_norm_fake).to(DEVICE)
    
    # PageRank & Model (New Instance)
    sgconv_fake = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
    X_gconv_fake, _ = sgconv_fake(X_t_fake, adj_tensor_fake._indices(), adj_tensor_fake._values())
    
    model_fake = LLGC(X_gconv_fake.size(1), EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
    
    # Train using SAME history (prev_Z_t) so fakes don't pollute timeline
    model_fake = train_unsupervised_with_prior(
        model_fake, X_gconv_fake, adj_tensor_fake._indices(), 
        prev_embeddings=prev_Z_t, epochs=EPOCHS, lr=LR, device=DEVICE
    )
    
    model_fake.eval()
    Z_t_fake = model_fake(X_gconv_fake).cpu().detach()
    
    # --- DYNAMIC CONTAMINATION ---
    # Calculate exact ratio: Fakes / (Real + Fakes)
    # e.g., if we injected 5%, we look for ~5% anomalies.
    actual_contamination = n_injected / (N_real + n_injected)
    valid_contamination = min(0.5, actual_contamination) # Cap at 0.5
    
    # Run detection with dynamic contamination
    Z_numpy_fake = Z_t_fake.numpy()
    scores_fake, pred_fake = run_anomaly_detection(Z_numpy_fake, contamination_rate=valid_contamination)
    
    # Calculate Stats
    fake_preds = pred_fake[fake_indices] # Only check indices of fakes
    detected_count = (fake_preds == -1).sum()
    
    # Store Fake Results (optional, for analysis)
    for i, fake_idx in enumerate(fake_indices):
        all_results_rows.append({
            'paper_id': f"FAKE_{t_start}_{i}",
            't_start': t_start, 't_end': t_end,
            'anomaly_score': scores_fake[fake_idx], 'prediction': pred_fake[fake_idx],
            'is_synthetic': True
        })

    print(f"Segment [{t_start}-{t_end}]: Real={N_real}, Injected={n_injected} | "
          f"Detection: {detected_count}/{n_injected} ({(detected_count/n_injected):.1%})")

# --------------------------
# 9. Save Results
# --------------------------
out_df = pd.DataFrame(all_results_rows)
out_df.to_csv("temporal_anomaly_results.csv", index=False)
print("\n✅ CSV saved: temporal_anomaly_results.csv")