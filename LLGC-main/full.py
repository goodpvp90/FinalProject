import pandas as pd
import networkx as nx
import ast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from sklearn.ensemble import IsolationForest
# The LLGC model and PageRankAgg are imported from your existing files
from model import LLGC, PageRankAgg


# --- New Helper Function for Anomaly Detection ---

def run_anomaly_detection(embeddings):
    """
    Fits an IsolationForest model on the embeddings to identify anomalies.
    
    Returns:
        anomaly_scores: The anomaly score for each node (lower is more anomalous).
        anomalies: Binary array where -1 indicates an anomaly and 1 indicates an inlier.
    """
    # Initialize Isolation Forest. contamination='auto' estimates the contamination
    # (proportion of outliers) automatically.
    # We use a random_state for reproducibility.
    # Note: Isolation Forest is run on the CPU (NumPy array) for stability.
    clf = IsolationForest(contamination='auto', random_state=42)
    
    # Fit the model and predict outliers
    clf.fit(embeddings)
    anomalies = clf.predict(embeddings)
    
    # Get the decision function (anomaly score)
    anomaly_scores = clf.decision_function(embeddings)
    
    return anomaly_scores, anomalies

# --- Modified Training Function (for Dummy Unsupervised Embedding) ---

def train(model, train_features, dummy_labels, epochs=50, lr=0.0002):
    """
    Simplified training to learn a hyperbolic projection (embedding).
    It uses cross-entropy loss with dummy labels to train the LorentzLinear layer.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # The output dimension of the LLGC model is now the Embedding Dimension
        output = model(train_features)
        
        # Cross-entropy loss with dummy labels (all zero) forces the model 
        # to find a stable/meaningful low-dimensional projection.
        loss_train = F.cross_entropy(output, dummy_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t
    return model, train_time

# --- Helper Functions required from previous steps (re-included for clarity) ---

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).float()


def normalize_adj_sym(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# Name of the uploaded CSV file
file_name = "final_filtered_by_fos_and_reference.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_name)

# Display the shape (rows, columns) and the first 5 rows of the DataFrame
print(f"DataFrame shape: {df.shape}")
print("\nFirst 5 rows of the DataFrame:")
print(df.head())
# This block assumes 'df' DataFrame is already loaded from the CSV.

# 1. Pre-process the 'references' column
# The 'references' column must be converted from a string to a Python list
df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# 2. Create the Directed Graph (DiGraph)
G = nx.DiGraph()

# 3. Add Nodes and their Attributes
paper_ids = set(df['id'])
for _, row in df.iterrows():
    paper_id = row['id']
    attributes = row.drop(['id','title'	,'authors.name','year','fos.name','n_citation','references','abstract']).to_dict()
    G.add_node(paper_id, **attributes)

# 4. Add Edges (Citations)
# Edges: Citing Paper -> Cited Paper (Source -> Target)
for _, row in df.iterrows():
    citing_paper_id = row['id']
    cited_papers_ids = row['references']
    
    for cited_paper_id in cited_papers_ids:
        # Only add an edge if the cited paper is present in our dataset
        if cited_paper_id in paper_ids:
            G.add_edge(citing_paper_id, cited_paper_id) 

# The variable 'G' is now your graph input.


# --- A. Text Feature Extraction (from title and abstract) ---

# 1. Combine the title and abstract text fields
# These features are vectorized together
df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# 2. Apply TF-IDF Vectorization to convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(df['text_combined']).toarray()


# --- B. Metadata Feature Extraction (from year, n_citation, fos.name, author.name) ---

# 1. Numerical Features: n_citation and year
numerical_features = df[['n_citation', 'year']].copy()
scaler = StandardScaler()
X_numerical = scaler.fit_transform(numerical_features)

# 2. Categorical Features: fos.name
fos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_fos = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))

# 3. Authorship Feature: author.name is transformed into "Number of authors"
# We calculate the number of authors and scale it
df['num_authors'] = df['authors.name'].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) and x != '[]' else 1)
X_authors = scaler.fit_transform(df[['num_authors']])


# --- C. Feature Combination (The final static feature matrix) ---

# Stack all derived feature matrices horizontally
X_static = np.hstack((X_text, X_numerical, X_fos, X_authors))

print("\n---")
print("Step 5 (4.2.1.1): Final Static Feature Vector Creation")
print(f"Shape of text features (title + abstract): {X_text.shape}")
print(f"Shape of metadata features (year, n_citation, fos.name, author.name): {X_numerical.shape[1] + X_fos.shape[1] + X_authors.shape[1]} features")
print(f"Shape of the final static feature matrix (X_static): {X_static.shape}")


# --- A. Node ID to Index Mapping ---
# The adjacency matrix and feature matrix are indexed from 0 to N-1, 
# where N is the number of nodes. We need a mapping from the original string IDs to these indices.
idx_to_id = {i: node_id for i, node_id in enumerate(G.nodes())}
id_to_idx = {node_id: i for i, node_id in enumerate(G.nodes())}


# --- B. Create Adjacency Matrix (A) ---
# The adjacency matrix is the graph structure input required by the model.

# 1. Get the edge list (citing_id, cited_id)
# We only care about the internal citations within our dataset
edges = []
for u, v in G.edges():
    # Convert string IDs back to 0-based integer indices
    edges.append((id_to_idx[u], id_to_idx[v]))

# 2. Convert edge list to a sparse matrix (COO format)
N = G.number_of_nodes()
# The adjacency matrix for a directed graph is not necessarily symmetric.
# (row indices, column indices), data, shape
rows = [u for u, v in edges]
cols = [v for u, v in edges]
data = np.ones(len(edges))
adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)


# --- C. Normalization (Crucial for GNNs like LLGC) ---
# The LLGC model likely requires a normalized, symmetric adjacency matrix (or similar).
# We will use the standard symmetric normalization (D^-0.5 * A * D^-0.5) common in GCNs
# as a robust default, which often works with models derived from GCN principles.
def normalize_adj_sym(adj):
    """Symmetrically normalize adjacency matrix."""
    # Convert to symmetric matrix: A + A_T
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # Add self-loops: A_hat = A_sym + I
    adj_hat = adj + sp.eye(adj.shape[0])
    # Degree matrix: D
    rowsum = np.array(adj_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # Normalization: D^-0.5 * A_hat * D^-0.5
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

adj_normalized = normalize_adj_sym(adj_matrix)


# --- D. Convert to PyTorch Tensors ---

# 1. Features (X_static)
X_tensor = torch.FloatTensor(X_static) 

# 2. Adjacency matrix (sparse representation)
# PyTorch Geometric models often use an "edge_index" representation (2xE tensor)
# but for the LLGC model's normalization utility functions, it often prefers 
# the sparse matrix format, which we convert to a PyTorch sparse tensor.
def sparse_to_torch_sparse(sparse_mx):
    """Converts a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

adj_tensor = sparse_to_torch_sparse(adj_normalized)


print("\n---")
print("Step 6: PyTorch Data Preparation")
print(f"Feature Tensor Shape (X_tensor): {X_tensor.shape}")
print(f"Normalized Adjacency Tensor Shape (adj_tensor): {adj_tensor.shape}")
print(f"Number of non-zero entries in Adjacency Tensor (edges + self-loops): {adj_tensor._values().size(0)}")
print("Data is now ready for the LLGC model initialization.")

# --- Temporal Segmentation (4.2.2) ---

# Define the window size (delta t) for each temporal snapshot
DELTA_T = 5

# Get the relevant years from the DataFrame
years = df['year'].dropna().astype(int)

# Determine the start and end year for segmentation
min_year = years.min()
max_year = years.max()

# Create a sequence of time steps (t1, t2, ...)
time_steps = list(range(min_year, max_year + DELTA_T, DELTA_T))

# Map original paper IDs to their indices (0 to N-1) for subsetting tensors
id_to_idx = {node_id: i for i, node_id in enumerate(G.nodes())}


# Generate the list of temporal segments
temporal_segments = {}
for i in range(len(time_steps) - 1):
    t_start = time_steps[i]
    t_end = time_steps[i+1]
    
    # Select papers published up to the end of the current time step (t_end)
    # The segment includes all papers with year <= t_end.
    current_segment_df = df[df['year'] <= t_end]
    
    # Get the 0-based indices corresponding to these papers
    node_indices_in_segment = [id_to_idx[id] for id in current_segment_df['id'] if id in id_to_idx]
    
    # Store the indices for this time window [t_start, t_end]
    temporal_segments[(t_start, t_end)] = node_indices_in_segment
    
    # Optional: Break if the segment does not add new nodes (i.e., we passed the max year)
    if t_end > max_year and len(node_indices_in_segment) == len(temporal_segments[list(temporal_segments.keys())[-2]]):
         break

print("\n---")
print("Step 8 (4.2.2): Temporal Segmentation")
print(f"Time Range: {min_year} - {max_year}")
print(f"Window Size ($\Delta t$): {DELTA_T} years")
print(f"Total Segments Created: {len(temporal_segments)}")
# Show the size of the first and last segments
first_key = list(temporal_segments.keys())[0]
last_key = list(temporal_segments.keys())[-1]
print(f"First Segment {first_key}: {len(temporal_segments[first_key])} nodes")
print(f"Last Segment {last_key}: {len(temporal_segments[last_key])} nodes")


# --- Anomaly Detection Parameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_PROP = 20           # K: number of propagation steps.
ALPHA = 0.1         # alpha: PageRankAgg alpha
DROP_OUT = 0.0      # Hyperbolic linear drop_out
USE_BIAS = 1        # Hyperbolic linear bias (1 or 0)

# Critical change: The output dimension of LLGC is now the embedding size
EMBEDDING_DIM = 2 # Setting to 2 for a minimal projection, could be 64 for a richer embedding.
DUMMY_N_CLASSES = 2 # The target dimension for the Lorentz Linear layer is now the embedding dim

print("\n---")
print(f"Embedding dimension for anomaly detection: {EMBEDDING_DIM}")
print(f"Training device: {DEVICE}")
print("---")


# --- Anomaly Detection Loop (Replaces the Classification Loop) ---

anomaly_results = {}
# Ensure X_tensor is on the correct device
X_tensor = X_tensor.to(DEVICE)

print("\nStep 8 (4.3): Extracting Embeddings and Running Anomaly Detection on Temporal Subgraphs")

# Iterate over each time window [t_start, t_end]
for (t_start, t_end), global_indices in temporal_segments.items():
    if not global_indices:
        print(f"Skipping segment [{t_start}, {t_end}] - No nodes.")
        continue

    # --- 1. Subsetting Data for the Segment ---
    X_t = X_tensor[global_indices]
    N_t = len(global_indices)
    
    # --- 2. Sub-Adjacency Matrix Generation and Normalization ---
    global_indices_np = np.array(global_indices)
    adj_segment_unnorm = adj_matrix[global_indices_np[:, None], global_indices_np]
    adj_segment_norm = normalize_adj_sym(adj_segment_unnorm)
    adj_segment_tensor = sparse_mx_to_torch_sparse_tensor(adj_segment_norm).to(DEVICE)

    # --- 3. LLGC Pre-computation (PageRankAgg) ---
    sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
    adj_indices = adj_segment_tensor._indices()
    adj_values = adj_segment_tensor._values()
    X_gconv, precompute_time = sgconv(X_t, adj_indices, adj_values) 

    # --- 4. LLGC Model Initialization and Dummy Training ---
    in_feat = X_gconv.size(1)
    # Initialize LLGC model with the desired EMBEDDING_DIM as the output size
    model = LLGC(in_feat, EMBEDDING_DIM, DROP_OUT, USE_BIAS).to(DEVICE)

    # Create dummy labels (all 0) for the simplified training
    dummy_labels = torch.zeros(N_t).long().to(DEVICE)
    
    # Train the model with the dummy labels to learn the hyperbolic projection
    model_trained, train_time = train(model, X_gconv, dummy_labels)
    
    # --- 5. Extract Final Embeddings ---
    # The output of the model is the Lorentzian embedding (z_t)
    model_trained.eval()
    Z_t = model_trained(X_gconv).cpu().detach().numpy() # Move to CPU for Sklearn
    
    # --- 6. Anomaly Detection ---
    anomaly_scores, anomalies = run_anomaly_detection(Z_t)
    
    # Calculate simple stats on the anomaly prediction
    n_anomalies = np.sum(anomalies == -1)
    
    # Store and print results
    anomaly_results[(t_start, t_end)] = {
        'n_nodes': N_t,
        'n_anomalies': n_anomalies,
        'anomaly_ratio': n_anomalies / N_t,
        'embeddings': Z_t,
        'scores': anomaly_scores,
        'predictions': anomalies,
    }
    
    print(f"Segment [{t_start}, {t_end}]: Nodes={N_t}, Anomalies Detected: {n_anomalies} ({anomaly_results[(t_start, t_end)]['anomaly_ratio']:.2%}), Pre-compute Time: {precompute_time:.4f}s, Train Time: {train_time:.4f}s")


print("\n---")
print("LLGC Embedding and Anomaly Detection Complete for all temporal segments.")
print("Summary of Anomaly Detection per Segment:")
for key, res in anomaly_results.items():
    print(f"[{key[0]}-{key[1]}]: Nodes={res['n_nodes']}, Anomalies={res['n_anomalies']} ({res['anomaly_ratio']:.2%})")



