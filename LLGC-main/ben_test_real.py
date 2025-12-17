import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

# ייבוא הרכיבים מהקבצים הקיימים ב-Repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.2) 
# פרמטר חדש: כמה משקל לתת ל-FOS (למשל פי 5 יותר מהטקסט)
parser.add_argument('--fos_weight', type=float, default=5.0) 
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data_with_fos_weight():
    print("Loading data and encoding FOS features...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. הפקת פיצ'רים טקסטואליים (TF-IDF)
    df['combined_text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    text_features = vectorizer.fit_transform(df['combined_text']).toarray()
    
    # 2. קידוד FOS (One-Hot Encoding)
    # אנחנו הופכים את הקטגוריה של ה-FOS לוקטור מספרי
    fos_encoder = OneHotEncoder(sparse_output=False)
    fos_features = fos_encoder.fit_transform(df[['fos.name']].fillna('Unknown'))
    
    # 3. שילוב ומתן משקל (Weighting)
    # אנחנו מכפילים את וקטור ה-FOS ב-Weight שבחרנו כדי להגדיל את הדומיננטיות שלו
    weighted_fos = fos_features * args.fos_weight
    
    # איחוד הפיצ'רים: טקסט + FOS משוקלל
    combined_features = np.hstack([text_features, weighted_fos])
    print(f"Feature shape: {combined_features.shape} (Text: 500, FOS: {fos_features.shape[1]})")
    
    features_tensor = torch.FloatTensor(combined_features).to(device)
    
    # בניית הגרף
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx:
                    G.add_edge(row['id'], str(ref))
        except: continue

    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, combined_features, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features_tensor, adj_tensor, df

def main():
    features, adj, df = load_data_with_fos_weight()
    
    # PageRank Aggregation
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    # Lorentzian Projection
    model = LLGC(nfeat=features.size(1), nclass=256, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    # זיהוי אנומליות
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    # תוצאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    num_fakes = df['is_anomaly'].sum()
    top_detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Results with FOS Weight ({args.fos_weight}):")
    print(f"Detected {top_detected} / {num_fakes} fakes in Top {num_fakes}")
    print(f"Precision: {top_detected / num_fakes:.4f}")
    
    print("\nTop 5 Anomalies:")
    for i, (idx, row) in enumerate(df_sorted.head(5).iterrows()):
        source = "[INJECTED]" if row['is_anomaly'] == 1 else "[NATURAL]"
        print(f"#{i+1} {source} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:50]}...")
    print("="*50)

if __name__ == "__main__":
    main()