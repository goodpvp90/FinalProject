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
from model import LLGC, PageRankAgg
import random

# --------------------------
# 0. Reproducibility
# --------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# 1. Load Data
# --------------------------
FILE_REAL = "final_filtered_by_fos_and_reference.csv"
FILE_FAKE = "fakes.csv"

df_real = pd.read_csv(FILE_REAL)
df_fake = pd.read_csv(FILE_FAKE)

# Convert references
for df in [df_real, df_fake]:
    df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# --------------------------
# 2. Build Graph
# --------------------------
G = nx.DiGraph()
for _, row in df_real.iterrows():
    G.add_node(row['id'])
for _, row in df_real.iterrows():
    for ref in row['references']:
        if ref in G:
            G.add_edge(row['id'], ref)

# Map id<->index
id_to_idx = {nid:i for i,nid in enumerate(G.nodes())}
idx_to_id = {i:nid for nid,i in id_to_idx.items()}

# --------------------------
# 3. Feature Construction
# --------------------------
# Textual: TF-IDF of title + abstract
df_real['text'] = df_real['title'].fillna('') + ' ' + df_real['abstract'].fillna('')
vectorizer = TfidfVectorizer(max_features=128)
X_text = vectorizer.fit_transform(df_real['text']).toarray()

# Numerical: citations, year, num_authors
df_real['num_authors'] = df_real['authors.name'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x,str) else 1)
X_num = df_real[['n_citation','year','num_authors']].values
X_num = StandardScaler().fit_transform(X_num)

# Categorical: fos.name
X_fos = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(df_real[['fos.name']].fillna('Unknown'))

# Combine all features
X_features = np.hstack([X_text, X_num, X_fos])
X_tensor = torch.FloatTensor(X_features).to(DEVICE)

# --------------------------
# 4. Adjacency Matrix
# --------------------------
edges = [(id_to_idx[u], id_to_idx[v]) for u,v in G.edges()]
rows = [u for u,v in edges]
cols = [v for u,v in edges]
adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(G), len(G)), dtype=np.float32)

def normalize_adj(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    d_inv_sqrt = np.power(np.array(adj_hat.sum(1)), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

adj_norm = normalize_adj(adj)
adj_tensor = sparse_to_tensor(adj_norm).to(DEVICE)

# --------------------------
# 5. Train LLGC on real data
# --------------------------
K_PROP = 5
ALPHA = 0.1
EMBED_DIM = 128
EPOCHS = 50
LR = 0.01

sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv, _ = sgconv(X_tensor, adj_tensor._indices(), adj_tensor._values())

model = LLGC(X_gconv.size(1), EMBED_DIM, drop_out=0.0, use_bias=True).to(DEVICE)

def train_unsupervised(model, X, adj_idx, epochs=50, lr=0.01):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    row, col = adj_idx
    for _ in range(epochs):
        optimizer.zero_grad()
        Z = model(X)
        loss = ((Z[row]-Z[col])**2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
    return model

model = train_unsupervised(model, X_gconv, adj_tensor._indices(), epochs=EPOCHS, lr=LR)
model.eval()

with torch.no_grad():
    Z_real = model(X_gconv).cpu().numpy()

# --------------------------
# 6. Detect initial anomalies
# --------------------------
clf = IsolationForest(contamination=0.01, random_state=SEED)
clf.fit(Z_real)
pred_real = clf.predict(Z_real)
anomalous_real_ids = np.array(df_real['id'])[pred_real==-1]
print(f"Detected initial anomalies: {len(anomalous_real_ids)}")

# --------------------------
# 7. Remove detected anomalies (core)
# --------------------------
df_core = df_real[~df_real['id'].isin(anomalous_real_ids)].copy()
G_core = nx.DiGraph()
for nid in df_core['id']: G_core.add_node(nid)
for _, row in df_core.iterrows():
    for ref in row['references']:
        if ref in G_core: G_core.add_edge(row['id'], ref)

print(f"Core size: {len(df_core)}")

# --------------------------
# 8. Inject synthetic anomalies
# --------------------------
# Assign unique IDs to fakes
next_id_int = int(df_core['id'].apply(lambda x: int(x[:8],16)).max()) + 1
for i,row in df_fake.iterrows():
    row_id = f"{next_id_int:08x}fake{i}"
    df_fake.at[i,'id'] = row_id
    next_id_int += 1
    row_refs = []  # optionally connect to 0-3 real nodes randomly
    df_fake.at[i,'references'] = row_refs
    df_fake.at[i,'is_synthetic'] = True

df_injected = pd.concat([df_core, df_fake], ignore_index=True)

# --------------------------
# 9. Recompute features for injected graph
# --------------------------
df_injected['text'] = df_injected['title'].fillna('') + ' ' + df_injected['abstract'].fillna('')
X_text_new = vectorizer.transform(df_injected['text']).toarray()
df_injected['num_authors'] = df_injected['authors.name'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x,str) else 1)
X_num_new = StandardScaler().fit_transform(df_injected[['n_citation','year','num_authors']].values)
X_fos_new = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(df_injected[['fos.name']].fillna('Unknown'))
X_all_new = np.hstack([X_text_new, X_num_new, X_fos_new])
X_tensor_new = torch.FloatTensor(X_all_new).to(DEVICE)

# Rebuild graph
G_new = nx.DiGraph()
for nid in df_injected['id']: G_new.add_node(nid)
for _, row in df_injected.iterrows():
    for ref in row['references']:
        if ref in df_injected['id']:
            G_new.add_edge(row['id'], ref)

id_to_idx_new = {nid:i for i,nid in enumerate(G_new.nodes())}
edges_new = [(id_to_idx_new[u], id_to_idx_new[v]) for u,v in G_new.edges()]
rows_new = [u for u,v in edges_new]
cols_new = [v for u,v in edges_new]
adj_new = sp.coo_matrix((np.ones(len(rows_new)), (rows_new, cols_new)), shape=(len(G_new), len(G_new)), dtype=np.float32)
adj_norm_new = normalize_adj(adj_new)
adj_tensor_new = sparse_to_tensor(adj_norm_new).to(DEVICE)

# --------------------------
# 10. GCN aggregation + embeddings
# --------------------------
sgconv_new = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv_new, _ = sgconv_new(X_tensor_new, adj_tensor_new._indices(), adj_tensor_new._values())

with torch.no_grad():
    Z_new = model(X_gconv_new).cpu().numpy()

# --------------------------
# 11. Detect anomalies on injected nodes
# --------------------------
clf_new = IsolationForest(contamination=0.01, random_state=SEED)
clf_new.fit(Z_new)
pred_new = clf_new.predict(Z_new)

fake_mask = df_injected.get('is_synthetic', pd.Series(False)).values.astype(bool)
detected_count = np.sum(pred_new[fake_mask]==-1)
print(f"Injected anomalies: {fake_mask.sum()}")
print(f"Detected anomalies: {detected_count}")
print(f"Detection rate: {detected_count/fake_mask.sum():.2%}")

# --------------------------
# 12. Save results
# --------------------------
results = pd.DataFrame({
    'paper_id': df_injected['id'],
    'anomaly_score': clf_new.decision_function(Z_new),
    'prediction': pred_new,
    'is_synthetic': fake_mask
})
results.to_csv("llgc_full_anomaly_results.csv", index=False)
print("✅ Results saved to llgc_full_anomaly_results.csv")
