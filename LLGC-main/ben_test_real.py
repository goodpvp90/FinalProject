import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ייבוא הרכיבים מהקבצים ב-Repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--K', type=int, default=10)
# העלאה ל-0.5 כדי לשמור על ה-FOS המקורי של המאמר
parser.add_argument('--alpha', type=float, default=0.5) 
# משקל אגרסיבי ל-FOS
parser.add_argument('--fos_weight', type=float, default=20.0) 
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data_high_fos():
    print(f"Loading data with high FOS focus (Weight: {args.fos_weight})...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. טקסט - נרמול הפיצ'רים לרמה של 0 עד 1
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    text_features = vectorizer.fit_transform(df['text']).toarray()
    
    # 2. FOS - קידוד ומתן משקל גבוה
    fos_encoder = OneHotEncoder(sparse_output=False)
    fos_features = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))
    weighted_fos = fos_features * args.fos_weight
    
    # 3. איחוד
    combined_features = np.hstack([text_features, weighted_fos])
    features_tensor = torch.FloatTensor(combined_features).to(device)
    
    # בניית הגרף
    node_ids = df['id'].tolist()
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx: G.add_edge(row['id'], str(ref))
        except: continue

    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, combined_features, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features_tensor, adj_tensor, df

def main():
    features, adj, df = load_data_high_fos()
    
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    num_fakes = df['is_anomaly'].sum()
    top_detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Results Summary (Alpha={args.alpha}, FOS Weight={args.fos_weight}):")
    print(f"Precision@{num_fakes}: {top_detected / num_fakes:.4f} ({top_detected}/{num_fakes})")
    
    print("\nTop 10 Anomalies:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        src = "[INJECTED]" if row['is_anomaly'] == 1 else "[NATURAL]"
        print(f"#{i+1} {src} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:50]}...")
    print("="*50)

if __name__ == "__main__":
    main()