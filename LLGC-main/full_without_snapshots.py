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
import random

from model import LLGC, PageRankAgg

# --------------------------
# 0. Set Seeds
# --------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# --------------------------
# 1. Load Data
# --------------------------
file_name = "C:\\Users\\nadir\\FinalProject\\LLGC-main\\final_filtered_by_fos_and_reference.csv"
print(f"Loading data from: {file_name}")
df = pd.read_csv(file_name)

df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df['is_synthetic'] = False

# --------------------------
# 2. Synthetic Injection
# --------------------------
def inject_synthetic_nodes(df, percent_new=0.05, min_connections=2, max_connections=7, id_prefix="FAKE_"):
    df = df.copy()
    n_real = len(df)
    n_fake = max(1, int(n_real * percent_new))

    years = df["year"].dropna().tolist() or [2000]
    fos_list = df["fos.name"].dropna().tolist() or ["Unknown"]
    citations = df["n_citation"].dropna().tolist() or [0]

    fake_rows = []
    print(f"[Injection] Creating {n_fake} synthetic nodes...")

    for i in range(n_fake):
        fake_rows.append({
            "id": f"{id_prefix}{i}",
            "title": "Synthetic Node",
            "authors.name": "[]",
            "year": int(random.choice(years)),
            "fos.name": random.choice(fos_list),
            "n_citation": int(random.choice(citations)),
            "references": [],
            "abstract": "Synthetic auto-generated text.",
            "is_synthetic": True
        })

    real_ids = df["id"].tolist()

    for fake in fake_rows:
        k = random.randint(min_connections, max_connections)
        fake["references"] = random.sample(real_ids, min(k, len(real_ids)))

    df_fake = pd.DataFrame(fake_rows)
    return pd.concat([df, df_fake], ignore_index=True)


print("\nInjecting synthetic nodes...")
df = inject_synthetic_nodes(df, percent_new=0.05)
print(f"New size: {len(df)} rows")

# --------------------------
# 3. Build Graph
# --------------------------
G = nx.DiGraph()
paper_ids = set(df["id"])

for _, row in df.iterrows():
    G.add_node(row["id"])

for _, row in df.iterrows():
    for ref in row["references"]:
        if ref in paper_ids:
            G.add_edge(row["id"], ref)

# --------------------------
# 4. Feature Extraction
# --------------------------
df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_text = vectorizer.fit_transform(df["text"]).toarray()

num_features = df[["n_citation", "year"]].fillna(0)
scaler = StandardScaler()
X_num = scaler.fit_transform(num_features)

df["num_authors"] = df["authors.name"].apply(
    lambda x: len(ast.literal_eval(x)) if pd.notna(x) and x != "[]" else 1
)
X_auth = scaler.fit_transform(df[["num_authors"]])

fos_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_fos = fos_encoder.fit_transform(df[["fos.name"]].fillna("Unknown"))

X = np.hstack((X_text, X_num, X_auth, X_fos))
X_tensor = torch.tensor(X, dtype=torch.float32)

# --------------------------
# 5. Build Adjacency
# --------------------------
id_to_idx = {pid: i for i, pid in enumerate(df["id"])}
rows, cols = [], []

for u, v in G.edges():
    rows.append(id_to_idx[u])
    cols.append(id_to_idx[v])

adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(df)))

def normalize_adj_sym(A):
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    A_hat = A + sp.eye(A.shape[0])
    rowsum = np.array(A_hat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D = sp.diags(d_inv_sqrt)
    return A_hat.dot(D).transpose().dot(D).tocoo()

def sparse_to_torch(A):
    A = A.tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack((A.row, A.col)), dtype=torch.int64)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, A.shape)

adj_norm = normalize_adj_sym(adj)
adj_tensor = sparse_to_torch(adj_norm)

# --------------------------
# 6. LLGC + SGConv
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_PROP = 5
ALPHA = 0.1
EMBEDDING_DIM = 256
EPOCHS = 100
LR = 0.01

X_tensor = X_tensor.to(DEVICE)
adj_tensor = adj_tensor.to(DEVICE)

sgconv = PageRankAgg(K=K_PROP, alpha=ALPHA, add_self_loops=False).to(DEVICE)
X_gconv, _ = sgconv(X_tensor, adj_tensor._indices(), adj_tensor._values())

model = LLGC(X_gconv.size(1), EMBEDDING_DIM, 0.0, True).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()

    Z = model(X_gconv)
    row, col = adj_tensor._indices()[0], adj_tensor._indices()[1]
    loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()

    loss.backward()
    opt.step()

# --------------------------
# 7. Anomaly Detection
# --------------------------
model.eval()
embeddings = model(X_gconv).cpu().detach().numpy()

contamination = max(0.01, df["is_synthetic"].mean())
clf = IsolationForest(contamination=contamination, random_state=42)
clf.fit(embeddings)

scores = clf.decision_function(embeddings)
pred = clf.predict(embeddings)

# --------------------------
# 8. Save One Row Per Paper
# --------------------------
out = pd.DataFrame({
    "paper_id": df["id"],
    "anomaly_score": scores,
    "prediction": pred,
    "is_synthetic": df["is_synthetic"]
})

out.to_csv("anomaly_results_no_snapshots.csv", index=False)
print("\nSaved: anomaly_results_no_snapshots.csv\n")
