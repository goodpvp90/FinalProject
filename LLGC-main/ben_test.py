import pandas as pd
import networkx as nx
import ast
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import random
from model import LLGC, PageRankAgg

# -------------------------- 0. Setup --------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------- 1. Load & preprocess --------------------------
file_name = "final_filtered_by_fos_and_reference.csv"
print(f"Loading dataset: {file_name}")
df = pd.read_csv(file_name)

# Ensure references is a list
def parse_references(x):
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        return []
    except:
        return []

df['references'] = df['references'].apply(parse_references)

# Compute number of authors
def count_authors(x):
    try:
        if pd.isna(x) or x=='[]':
            return 1
        return len(ast.literal_eval(x))
    except:
        return 1
df['num_authors'] = df['authors.name'].apply(count_authors)

# -------------------------- 2. Build Graph --------------------------
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_node(row['id'])
for _, row in df.iterrows():
    for ref in row['references']:
        if ref in df['id'].values:
            G.add_edge(row['id'], ref)

# -------------------------- 3. Build Features --------------------------
# Textual features
df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_text = vectorizer.fit_transform(df['text_combined']).toarray()

# Numerical features
num_features = df[['n_citation', 'year', 'num_authors']].copy()
scaler = StandardScaler()
X_num = scaler.fit_transform(num_features)

# One-hot fos
fos_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_fos = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))

# Combine all features
X_all = np.hstack([X_text, X_num, X_fos])
X_tensor = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)
print(f"Feature matrix shape: {X_tensor.shape}")

# -------------------------- 4. Adjacency --------------------------
id_to_idx = {pid:i for i,pid in enumerate(df['id'])}
idx_to_id = {i:pid for pid,i in id_to_idx.items()}

edges = [(id_to_idx[u], id_to_idx[v]) for u,v in G.edges()]
rows = [u for u,v in edges]
cols = [v for u,v in edges]
adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(df)), dtype=np.float32)

# Symmetric normalization
def normalize_adj(adj):
    adj = adj + adj.T.multiply(adj.T>adj) - adj.multiply(adj.T>adj)
    adj_hat = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

adj_norm = normalize_adj(adj)
def sparse_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse_coo_tensor(indices, values, sparse_mx.shape)

adj_tensor = sparse_to_tensor(adj_norm).to(DEVICE)

# -------------------------- 5. Train LLGC --------------------------
K_PROP, ALPHA = 5, 0.1
DROP_OUT, USE_BIAS = 0.0, 1
EMBED_DIM = 256
EPOCHS, LR = 100, 0.01

# PageRank aggregation
sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv, _ = sgconv(X_tensor, adj_tensor._indices(), adj_tensor._values())

# Train LLGC on all nodes
model = LLGC(X_gconv.size(1), EMBED_DIM, DROP_OUT, USE_BIAS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    Z = model(X_gconv)
    row, col = adj_tensor._indices()
    loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    Z_final = model(X_gconv).cpu().numpy()

# -------------------------- 6. Initial Anomaly Detection --------------------------
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(Z_final)
pred = clf.predict(Z_final)  # -1 = anomaly
scores = clf.decision_function(Z_final)

anomalous_ids = df['id'].iloc[np.where(pred==-1)[0]].tolist()
print(f"Detected initial anomalies: {len(anomalous_ids)}")

# Remove real anomalies
df_core = df[~df['id'].isin(anomalous_ids)].reset_index(drop=True)
X_tensor_core = X_tensor[[id_to_idx[pid] for pid in df_core['id']]]
print(f"Core size: {len(df_core)}")

# -------------------------- 7. Inject Synthetic Anomalies --------------------------
# Generate fake nodes directly
N_fake = max(1, int(0.05*len(df_core)))  # 5% new nodes
fake_rows = []
for i in range(N_fake):
    fake_id = f"FAKE{i:05d}"
    fake_rows.append({
        'id': fake_id,
        'title': f"Fake paper {i}",
        'authors.name': "['Fake Author']",
        'year': 2023,
        'fos.name': "Unknown",
        'n_citation': 0,
        'references': [],
        'abstract': f"This is fake abstract {i}",
        'num_authors': 1,
        'is_synthetic': True
    })

df_core['is_synthetic'] = False
df_injected = pd.concat([df_core, pd.DataFrame(fake_rows)], ignore_index=True)

# -------------------------- 8. Features & Embeddings for injected --------------------------
# TF-IDF for all abstracts (same vectorizer)
texts = df_injected['title'].fillna('') + ' ' + df_injected['abstract'].fillna('')
X_text_inj = vectorizer.transform(texts).toarray()

num_features_inj = df_injected[['n_citation', 'year', 'num_authors']].copy()
X_num_inj = scaler.transform(num_features_inj)

X_fos_inj = fos_encoder.transform(df_injected[['fos.name']].fillna('Unknown'))

X_all_inj = np.hstack([X_text_inj, X_num_inj, X_fos_inj])
X_tensor_inj = torch.tensor(X_all_inj, dtype=torch.float32).to(DEVICE)

# Rebuild adjacency
id_to_idx_inj = {pid:i for i,pid in enumerate(df_injected['id'])}
rows, cols = [], []
for i, refs in enumerate(df_injected['references']):
    if isinstance(refs, list):
        for ref in refs:
            if ref in id_to_idx_inj:
                rows += [i, id_to_idx_inj[ref]]
                cols += [id_to_idx_inj[ref], i]
adj_inj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df_injected), len(df_injected)), dtype=np.float32)
adj_norm_inj = normalize_adj(adj_inj)
adj_tensor_inj = sparse_to_tensor(adj_norm_inj).to(DEVICE)

sgconv_inj = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv_inj, _ = sgconv_inj(X_tensor_inj, adj_tensor_inj._indices(), adj_tensor_inj._values())

# -------------------------- 9. Compute embeddings & detect injected anomalies --------------------------
model.eval()
with torch.no_grad():
    Z_inj = model(X_gconv_inj).cpu().numpy()

fake_mask = df_injected['is_synthetic'].fillna(False).astype(bool).values
clf_inj = IsolationForest(contamination=0.01, random_state=42)
clf_inj.fit(Z_inj)
pred_inj = clf_inj.predict(Z_inj)
scores_inj = clf_inj.decision_function(Z_inj)

detected_count = (pred_inj[fake_mask]==-1).sum()
print(f"Injected anomalies: {fake_mask.sum()}")
print(f"Detected anomalies: {detected_count}")
print(f"Detection rate: {detected_count/fake_mask.sum():.2%}")

# -------------------------- 10. Save results --------------------------
results = []
for i, pid in enumerate(df_injected['id']):
    results.append({
        'paper_id': pid,
        'anomaly_score': scores_inj[i],
        'prediction': pred_inj[i],
        'is_synthetic': df_injected.loc[i,'is_synthetic']
    })

results_df = pd.DataFrame(results)
results_df.to_csv("llgc_full_anomaly_results.csv", index=False)
print("✅ Results saved to llgc_full_anomaly_results.csv")
