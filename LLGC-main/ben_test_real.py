import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, normalize

from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=128) # הגדלת ממד
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.7) # Alpha גבוה לשימור FOS
parser.add_argument('--fos_weight', type=float, default=50.0) # משקל אגרסיבי מאוד
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data_ultra_fos():
    print(f"Loading data with Ultra FOS focus (Weight: {args.fos_weight})...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. טקסט + נרמול L2 (חשוב מאוד!)
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    text_features = vectorizer.fit_transform(df['text']).toarray()
    # נרמול כל שורה כך שסכום הריבועים יהיה 1 - מונע מטקסט ארוך להשתלט
    text_features = normalize(text_features, norm='l2')
    
    # 2. FOS משוקלל
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
    features, adj, df = load_data_ultra_fos()
    
    # PageRank עם Alpha גבוה לשמירה על הפיצ'רים המשוקללים
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    # מודל לורנציאני בממד גבוה
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    # זיהוי אנומליות
    clf = IsolationForest(contamination='auto', random_state=args.seed, n_jobs=-1)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    num_fakes = df['is_anomaly'].sum()
    top_detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print("\n" + "="*60)
    print(f"Results (Alpha={args.alpha}, FOS Weight={args.fos_weight}, Dim={args.embedding_dim}):")
    print(f"Detected {top_detected} / {num_fakes} fakes in Top {num_fakes} (Precision: {top_detected/num_fakes:.4f})")
    
    print("\nTop 15 Anomalies:")
    for i, (idx, row) in enumerate(df_sorted.head(15).iterrows()):
        src = "[INJECTED]" if row['is_anomaly'] == 1 else "[NATURAL]"
        print(f"#{i+1:<2} {src} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']:<25} | Title: {row['title'][:45]}...")
    print("="*60)

if __name__ == "__main__":
    main()