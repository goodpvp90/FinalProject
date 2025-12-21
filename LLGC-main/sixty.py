import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import random
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# ייבוא הרכיבים מהקבצים הקיימים ב-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# --------------------------
# 1. פונקציית עזר לניקוי ה-References (פותר את ה-SyntaxError)
# --------------------------
def safe_parse_references(x):
    if not isinstance(x, str) or x.strip() in ["[]", ""]:
        return []
    # הסרת סוגריים וגרשיים, ופיצול לפי פסיק
    cleaned = x.strip('[]').replace("'", "").replace('"', "")
    return [ref.strip() for ref in cleaned.split(',') if ref.strip()]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1, help='Crucial for smoothing real nodes')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# 2. טעינה והכנת נתונים
# --------------------------
def load_and_prepare_data():
    print("Loading data...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_synthetic'] = False
    df_fakes['is_synthetic'] = True
    
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # שימוש בתיקון החדש במקום ast.literal_eval
    df['references'] = df['references'].apply(safe_parse_references)
    
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for i, row in df.iterrows():
        for ref in row['references']:
            if str(ref) in node_to_idx:
                G.add_edge(row['id'], str(ref))
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return df, G, node_ids

# --------------------------
# 3. הנדסת תכונות (Hybrid)
# --------------------------
def extract_hybrid_features(df, G, node_ids):
    print("Extracting hybrid features...")
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    # הגדלת max_features ושימוש ב-sublinear_tf לנטרול חריגות קיצוניות
    vectorizer = TfidfVectorizer(max_features=2500, stop_words='english', sublinear_tf=True)
    X_text = vectorizer.fit_transform(df['text']).toarray()
    
    # שימוש ב-RobustScaler כדי שמאמרי סקר (Hubs) לא יחשבו לאנומליות
    degrees = np.array([G.degree(n) for n in node_ids]).reshape(-1, 1)
    scaler = RobustScaler()
    X_struct = scaler.fit_transform(np.log1p(degrees))
    
    # איחוד עם נרמול שורות
    X_combined = np.hstack([X_text * 2.0, X_struct])
    row_norms = np.linalg.norm(X_combined, axis=1, keepdims=True)
    X_combined = X_combined / (row_norms + 1e-9)
    
    return torch.FloatTensor(X_combined).to(device)

def train_model(model, features, adj_indices, epochs=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        Z = model(features)
        row, col = adj_indices
        loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
    return model

# --------------------------
# 4. הרצה מרכזית
# --------------------------
def main():
    df, G, node_ids = load_and_prepare_data()
    X_tensor = extract_hybrid_features(df, G, node_ids)
    
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj_norm, _ = preprocess_citation(adj, X_tensor.cpu().numpy(), normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
    
    # החלקה (Smoothing) - alpha=0.1 מבטיח שצמתים רגילים יתמזגו
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    X_smooth, _ = aggregator(X_tensor, adj_tensor._indices())
    
    model = LLGC(nfeat=X_tensor.size(1), nclass=args.embedding_dim, 
                  drop_out=0.0, use_bias=True).to(device)
    
    model = train_model(model, X_smooth, adj_tensor._indices(), epochs=args.epochs)
    
    model.eval()
    with torch.no_grad():
        embeddings = model(X_smooth).cpu().numpy()
    
    # זיהוי עם כיול מדויק של 5% (אחוז הזיופים בפועל)
    clf = IsolationForest(contamination=0.05, random_state=args.seed, n_estimators=200)
    clf.fit(embeddings)
    
    df['anomaly_score'] = -clf.decision_function(embeddings)
    df['prediction'] = clf.predict(embeddings) 
    
    fakes_mask = df['is_synthetic'] == True
    detected_fakes = df[fakes_mask & (df['prediction'] == -1)].shape[0]
    
    print("\n" + "="*50)
    print(f"FINAL RESULTS")
    print(f"Detected {detected_fakes} out of {fakes_mask.sum()} fakes.")
    print(f"Precision: {detected_fakes / fakes_mask.sum():.4f}")
    print("="*50)

if __name__ == "__main__":
    main()