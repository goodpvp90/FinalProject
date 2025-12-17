import pandas as pd
import networkx as nx
import ast
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import torch.nn as nn
import torch.optim as optim
import random

from model import LLGC, PageRankAgg

# --------------------------
# 0. Reproducibility
# --------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# 1. Load dataset
# --------------------------
file_name = "final_filtered_by_fos_and_reference.csv"
print(f"Loading dataset: {file_name}")
df = pd.read_csv(file_name)

# Convert references to list of strings
df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# --------------------------
# 2. Graph creation
# --------------------------
G = nx.DiGraph()
paper_ids = set(df['id'])

for _, row in df.iterrows():
    G.add_node(row['id'])

for _, row in df.iterrows():
    for ref in row['references']:
        if ref in paper_ids:
            G.add_edge(row['id'], ref)

# --------------------------
# 3. Feature creation
# --------------------------
# Combine text features
df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_text = vectorizer.fit_transform(df['text_combined']).toarray()

# Numerical
num_features = df[['n_citation', 'year']].copy()
scaler = StandardScaler()
X_num = scaler.fit_transform(num_features)

# One-hot FOS
fos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_fos = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))

# Combine
X_static = np.hstack((X_text, X_num, X_fos))
X_tensor = torch.FloatTensor(X_static).to(DEVICE)
print(f"Feature matrix shape: {X_tensor.shape}")

# --------------------------
# 4. Graph adjacency
# --------------------------
idx_to_id = {i: nid for i, nid in enumerate(G.nodes())}
id_to_idx = {nid: i for i, nid in enumerate(G.nodes())}
edges = [(id_to_idx[u], id_to_idx[v]) for u,v in G.edges()]
N = len(G.nodes())
rows = [u for u,v in edges]
cols = [v for u,v in edges]
adj_matrix = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N), dtype=np.float32)

def normalize_adj(adj):
    adj = adj + adj.T.multiply(adj.T>adj) - adj.multiply(adj.T>adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_torch(adj):
    adj = adj.tocoo()
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape)).to(DEVICE)

adj_norm = normalize_adj(adj_matrix)
adj_tensor = sparse_to_torch(adj_norm)

# --------------------------
# 5. Helper functions
# --------------------------
def train_LLGC(model, X, adj_indices, prev_embeddings=None, epochs=50, lr=0.01, temporal_weight=0.5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    row, col = adj_indices
    for _ in range(epochs):
        optimizer.zero_grad()
        Z = model(X)
        loss = (Z[row]-Z[col]).pow(2).sum(dim=1).mean()
        if prev_embeddings is not None:
            prev_embeddings = prev_embeddings.to(DEVICE)
            common = min(Z.size(0), prev_embeddings.size(0))
            loss += temporal_weight*((Z[:common]-prev_embeddings[:common]).pow(2).mean())
        loss.backward()
        optimizer.step()
    return model

def detect_anomalies(embeddings, contamination=0.05):
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(embeddings)
    pred = clf.predict(embeddings)
    scores = clf.decision_function(embeddings)
    return pred, scores

def inject_synthetic(df, fraction=0.05, min_edges=2, max_edges=5):
    n_fake = max(1, int(len(df)*fraction))
    df_fake = []
    existing_ids = set(df['id'])
    for i in range(n_fake):
        new_id = f"FAKE_{i}"
        # text very different
        title = f"Fake paper {i} about unicorns and cake"
        abstract = f"This fake abstract {i} contains bizarre keywords alien to dataset"
        year = df['year'].max()+1
        n_citation = 0
        fos = "SyntheticFOS"
        df_fake.append({
            'id': new_id,
            'title': title,
            'authors.name': "['FakeAuthor']",
            'year': year,
            'fos.name': fos,
            'n_citation': n_citation,
            'references': [],
            'abstract': abstract,
            'is_synthetic': True
        })
    df_fake = pd.DataFrame(df_fake)
    df_fake['references'] = df_fake['references'].apply(lambda x: x if isinstance(x,list) else [])
    df_new = pd.concat([df, df_fake], ignore_index=True)
    return df_new, df_fake['id'].tolist()

# --------------------------
# 6. Initial training and anomaly detection
# --------------------------
model = LLGC(X_tensor.size(1), 256, drop_out=0.0, use_bias=1).to(DEVICE)
model = train_LLGC(model, X_tensor, adj_tensor._indices())
model.eval()
with torch.no_grad():
    Z = model(X_tensor).cpu().numpy()

pred, scores = detect_anomalies(Z, contamination=0.05)
anomalous_ids = [idx_to_id[i] for i, p in enumerate(pred) if p==-1]
print(f"Detected initial anomalies: {len(anomalous_ids)}")

# Remove initial anomalies
df_core = df[~df['id'].isin(anomalous_ids)].copy()
core_indices = [id_to_idx[i] for i in df_core['id']]
X_core = X_tensor[core_indices]

print(f"Core size: {len(df_core)}")

# --------------------------
# 7. Inject synthetic anomalies
# --------------------------
df_core, fake_ids = inject_synthetic(df_core, fraction=0.05)
print(f"Injected anomalies: {len(fake_ids)}")

# rebuild features
df_core['text_combined'] = df_core['title'].fillna('') + ' ' + df_core['abstract'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_text = vectorizer.fit_transform(df_core['text_combined']).toarray()

num_features = df_core[['n_citation', 'year']].copy()
X_num = scaler.fit_transform(num_features)

X_fos = fos_encoder.fit_transform(df_core[['fos.name']].fillna('Unknown'))

X_static = np.hstack((X_text, X_num, X_fos))
X_tensor = torch.FloatTensor(X_static).to(DEVICE)

# rebuild graph
G = nx.DiGraph()
for _, row in df_core.iterrows():
    G.add_node(row['id'])
for _, row in df_core.iterrows():
    # connect fake nodes randomly
    refs = row['references']
    if row.get('is_synthetic', False):
        real_ids = [i for i in df_core['id'] if not str(i).startswith("FAKE_")]
        n_refs = random.randint(2,5)
        refs = random.sample(real_ids, min(n_refs, len(real_ids)))
        row['references'] = refs
    for ref in refs:
        if ref in df_core['id'].values:
            G.add_edge(row['id'], ref)

idx_to_id = {i: nid for i, nid in enumerate(G.nodes())}
id_to_idx = {nid: i for i, nid in enumerate(G.nodes())}
edges = [(id_to_idx[u], id_to_idx[v]) for u,v in G.edges()]
rows = [u for u,v in edges]
cols = [v for u,v in edges]
adj_matrix = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(G),len(G)), dtype=np.float32)
adj_norm = normalize_adj(adj_matrix)
adj_tensor = sparse_to_torch(adj_norm)

# --------------------------
# 8. Recompute embeddings on core + injected
# --------------------------
model = LLGC(X_tensor.size(1), 256, drop_out=0.0, use_bias=1).to(DEVICE)
model = train_LLGC(model, X_tensor, adj_tensor._indices())
model.eval()
with torch.no_grad():
    Z = model(X_tensor).cpu().numpy()

# Detect anomalies
pred, scores = detect_anomalies(Z, contamination=0.1)
fake_mask = df_core['is_synthetic'].fillna(False).astype(bool)
detected_fakes = sum((pred==-1) & fake_mask.values)

print(f"Injected anomalies: {len(fake_ids)}")
print(f"Detected anomalies: {detected_fakes}")
print(f"Detection rate: {detected_fakes/len(fake_ids)*100:.2f}%")

# --------------------------
# 9. Save results
# --------------------------
results = pd.DataFrame({
    'paper_id': df_core['id'],
    'is_synthetic': df_core.get('is_synthetic', False),
    'anomaly_score': scores,
    'prediction': pred
})
results.to_csv("llgc_full_anomaly_results.csv", index=False)
print("✅ Results saved to llgc_full_anomaly_results.csv")
