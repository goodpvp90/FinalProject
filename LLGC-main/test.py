import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import scipy.sparse as sp

from utils import load_citation
from model import LLGC, PageRankAgg


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def find_column(df, candidates):
    cols = df.columns
    lc = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in lc:
            return cols[lc.index(cand.lower())]
    for cand in candidates:
        for i, c in enumerate(lc):
            if cand.lower() in c:
                return cols[i]
    return None


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.argmax(dim=1)
    return (preds == labels).float().mean().item()


def lorentz_to_euclidean(emb):
    if emb.shape[1] <= 1:
        return emb
    x = emb[:, 1:]
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-9)


# ---------------------------------------------------------
# Load dataset (single full graph)
# ---------------------------------------------------------
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # find needed columns
    id_col     = find_column(df, ['id'])
    title_col  = find_column(df, ['title'])
    auth_col   = find_column(df, ['authors.name'])
    fos_col    = find_column(df, ['fos.name'])
    refs_col   = find_column(df, ['references'])

    print("Using:", id_col, title_col, auth_col, fos_col, refs_col)

    # ---------- Build features ----------
    combined = (
        df[title_col].astype(str) + " "
        + df[auth_col].astype(str) + " "
        + df[fos_col].astype(str)
    )
    vectorizer = TfidfVectorizer(max_features=1000)
    feats = vectorizer.fit_transform(combined)
    feats = torch.FloatTensor(np.array(feats.todense()))

    # ---------- Labels ----------
    encoder = LabelEncoder()
    labels = torch.LongTensor(encoder.fit_transform(df[fos_col]))

    # ---------- Build adjacency ----------
    ids = df[id_col].astype(str).tolist()
    id_to_idx = {x: i for i, x in enumerate(ids)}

    G = nx.DiGraph()
    G.add_nodes_from(range(len(ids)))

    for i, r in enumerate(df[refs_col]):
        try:
            lst = eval(r)
            for ref in lst:
                if ref in id_to_idx:
                    G.add_edge(i, id_to_idx[ref])
        except:
            pass

    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # ---------- GPU ----------
    if torch.cuda.is_available():
        feats = feats.cuda()
        labels = labels.cuda()
        adj = adj.cuda()

    # train/val/test split
    n = len(df)
    idx_train = torch.arange(int(0.6*n))
    idx_val   = torch.arange(int(0.6*n), int(0.8*n))
    idx_test  = torch.arange(int(0.8*n), n)

    if torch.cuda.is_available():
        idx_train = idx_train.cuda()
        idx_val   = idx_val.cuda()
        idx_test  = idx_test.cuda()

    return df, adj, feats, labels, idx_train, idx_val, idx_test


# ---------------------------------------------------------
# Train LLGC (one time only)
# ---------------------------------------------------------
def train_llgc(features, adj, labels, idx_train, idx_val, idx_test):
    args = {
        "lr": 0.6,
        "weight_decay": 3e-5,
        "epochs": 200,
        "K": 20,
        "alpha": 0.1,
        "drop_out": 0.0,
        "use_bias": 1,
    }

    # Precompute PageRank embedding
    sgconv = PageRankAgg(K=args["K"], alpha=args["alpha"]).cuda()
    x_gconv, pre_time = sgconv(features, adj._indices(), None)

    model = LLGC(
        x_gconv.size(1),
        labels.max().item() + 1,
        args["drop_out"],
        args["use_bias"]
    ).cuda()

    opt = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    for epoch in range(args["epochs"]):
        model.train()
        opt.zero_grad()
        out = model(x_gconv[idx_train])
        loss = F.cross_entropy(out, labels[idx_train])
        loss.backward()
        opt.step()

    model.eval()
    acc = accuracy(model(x_gconv[idx_test]), labels[idx_test])
    print("Test Accuracy:", acc)

    return x_gconv.detach().cpu().numpy()


# ---------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------
def run_anomaly_detection(emb):
    emb_euc = lorentz_to_euclidean(emb)
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(emb_euc)
    scores = -clf.decision_function(emb_euc)
    return scores


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main(file_path):
    df, adj, feats, labels, idx_tr, idx_v, idx_te = load_dataset(file_path)

    print("\nTraining LLGC (single run)...")
    emb = train_llgc(feats, adj, labels, idx_tr, idx_v, idx_te)

    print("\nRunning anomaly detection...")
    scores = run_anomaly_detection(emb)
    df["anomaly_score"] = scores

    print("\nTop 10 anomalies:")
    print(df.nlargest(10, "anomaly_score")[["title", "anomaly_score"]])


if __name__ == "__main__":
    FILE = "C:\\Users\\nadir\\OneDrive\\Desktop\\final_filtered_by_fos_and_reference (1).csv"
    main(FILE)
