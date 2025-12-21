import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import torch.optim as optim

# ייבוא הרכיבים מהקבצים הקיימים ב-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# --------------------------
# 1. הגדרות וקיבוע זרע (Reproducibility)
# --------------------------
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
parser.add_argument('--alpha', type=float, default=0.1, help='Low alpha for better smoothing of real nodes')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

set_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# 2. טעינה והזרקת אנומליות בצורה חכמה
# --------------------------
def load_and_prepare_data():
    print("Loading original and fake data...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_synthetic'] = False
    df_fakes['is_synthetic'] = True
    
    # איחוד דאטה-סטים
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # בניית הגרף
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for i, row in df.iterrows():
        for ref in row['references']:
            if str(ref) in node_to_idx:
                G.add_edge(row['id'], str(ref))
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return df, G, node_ids

# --------------------------
# 3. הנדסת תכונות היברידית (Text + Structure)
# --------------------------
def extract_hybrid_features(df, G, node_ids):
    print("Extracting hybrid features (TF-IDF + Structural Priors)...")
    
    # א. טקסט: TF-IDF עם הגדרות לניטרול אורך (Sublinear TF)
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', sublinear_tf=True)
    X_text = vectorizer.fit_transform(df['text']).toarray()
    
    # ב. מבנה: Log-Degree עם Robust Scaling לנטרול Hubs
    degrees = np.array([G.degree(n) for n in node_ids]).reshape(-1, 1)
    scaler = RobustScaler()
    X_struct = scaler.fit_transform(np.log1p(degrees))
    
    # ג. איחוד וקיבוע נרמול שורות
    X_combined = np.hstack([X_text * 2.0, X_struct]) # משקל גבוה יותר לטקסט
    row_norms = np.linalg.norm(X_combined, axis=1, keepdims=True)
    X_combined = X_combined / (row_norms + 1e-9)
    
    return torch.FloatTensor(X_combined).to(device)

# --------------------------
# 4. פונקציית אימון (Unsupervised)
# --------------------------
def train_model(model, features, adj_indices, epochs=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        Z = model(features)
        # Adjacency Reconstruction Loss
        row, col = adj_indices
        loss = (Z[row] - Z[col]).pow(2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    return model

# --------------------------
# 5. הרצה מרכזית
# --------------------------
def main():
    df, G, node_ids = load_and_prepare_data()
    X_tensor = extract_hybrid_features(df, G, node_ids)
    
    # הכנת מטריצת שכנות
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj_norm, _ = preprocess_citation(adj, X_tensor.cpu().numpy(), normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
    
    # אגריגציה (Smoothing) - Alpha נמוך הוא המפתח!
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    X_smooth, _ = aggregator(X_tensor, adj_tensor._indices())
    
    # מודל לורנציאני
    model = LLGC(nfeat=X_tensor.size(1), nclass=args.embedding_dim, 
                  drop_out=0.0, use_bias=True).to(device)
    
    # אימון על כל הגרף (כולל הזיופים, בצורה לא מונחית)
    model = train_model(model, X_smooth, adj_tensor._indices(), epochs=args.epochs)
    
    model.eval()
    with torch.no_grad():
        embeddings = model(X_smooth).cpu().numpy()
    
    # זיהוי אנומליות - כיול לפי 5% (שיעור ה-fakes המשוער)
    contamination = 0.05 
    clf = IsolationForest(contamination=contamination, random_state=args.seed, n_estimators=200)
    clf.fit(embeddings)
    
    df['anomaly_score'] = -clf.decision_function(embeddings)
    df['prediction'] = clf.predict(embeddings) # -1 לאנומליה
    
    # הערכה
    fakes_mask = df['is_synthetic'] == True
    num_fakes = fakes_mask.sum()
    detected_fakes = df[fakes_mask & (df['prediction'] == -1)].shape[0]
    
    print("\n" + "="*50)
    print(f"DETECTION RESULTS")
    print(f"Detected {detected_fakes} out of {num_fakes} fakes.")
    print(f"Precision@{num_fakes}: {detected_fakes / num_fakes:.4f}")
    print("="*50)
    
    # הצגת הטופ 10
    print("\nTop 10 Most Anomalous Papers Found:")
    top_10 = df.sort_values(by='anomaly_score', ascending=False).head(10)
    for i, (idx, row) in enumerate(top_10.iterrows()):
        label = "[FAKE]" if row['is_synthetic'] else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | Title: {row['title'][:70]}...")

if __name__ == "__main__":
    main()