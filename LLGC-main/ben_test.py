import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul
import scipy.sparse as sp
from sklearn.ensemble import IsolationForest

# --------------------------
# 0. Load dataset
# --------------------------
DATA_FILE = "final_filtered_by_fos_and_reference.csv"
print(f"Loading dataset: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# ensure 'references' column is a list
def parse_references(x):
    if isinstance(x, str):
        # remove quotes if they exist
        x_clean = x.strip().replace("'", '"')
        try:
            return list(eval(x_clean))
        except:
            return []
    elif isinstance(x, list):
        return x
    else:
        return []
df['references'] = df['references'].apply(parse_references)

# --------------------------
# 1. Build Graph
# --------------------------
G = nx.DiGraph()
paper_ids = set(df['id'])
for _, row in df.iterrows():
    G.add_node(row['id'])

for _, row in df.iterrows():
    for ref in row['references']:
        if ref in paper_ids:
            G.add_edge(row['id'], ref)

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --------------------------
# 2. Initial Features (degree only)
# --------------------------
deg_in = np.array([G.in_degree(n) for n in G.nodes()], dtype=np.float32)
deg_out = np.array([G.out_degree(n) for n in G.nodes()], dtype=np.float32)
X = np.stack([deg_in, deg_out], axis=1)
X_tensor = torch.FloatTensor(X)

# --------------------------
# 3. Sparse adjacency
# --------------------------
edges = list(G.edges())
rows = [list(G.nodes()).index(u) for u, v in edges]
cols = [list(G.nodes()).index(v) for u, v in edges]
data = np.ones(len(edges))
adj = sp.coo_matrix((data, (rows, cols)), shape=(len(G), len(G)), dtype=np.float32)

def sparse_to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

adj_tensor = sparse_to_torch_sparse(adj)

# --------------------------
# 4. PageRank-based aggregator
# --------------------------
class PageRankAgg(MessagePassing):
    def __init__(self, K=5, alpha=0.1):
        super().__init__(aggr='add')
        self.K = K
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for _ in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            x = x * (1 - self.alpha) + self.alpha * h
        return x

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return x_j * edge_weight.view(-1, 1)

agg = PageRankAgg(K=5, alpha=0.1)
embeddings = agg(X_tensor, adj_tensor._indices(), adj_tensor._values())

# --------------------------
# 5. Anomaly detection
# --------------------------
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(embeddings.detach().numpy())
pred = clf.predict(embeddings.detach().numpy())  # -1 = anomaly, 1 = normal
num_anomalies = (pred == -1).sum()
print(f"Detected anomalies: {num_anomalies}/{len(G)} ({num_anomalies/len(G):.2%})")

# --------------------------
# 6. Save results
# --------------------------
results_df = pd.DataFrame({
    'paper_id': list(G.nodes()),
    'anomaly_prediction': pred
})
results_df.to_csv("llgc_graph_only_anomaly_results.csv", index=False)
print("✅ Results saved to llgc_graph_only_anomaly_results.csv")
