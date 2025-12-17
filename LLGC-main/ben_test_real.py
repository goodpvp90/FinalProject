import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים מהקבצים הקיימים ב-Repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("Loading and preparing data...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # בניית הגרף
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx:
                    G.add_edge(row['id'], str(ref))
        except: continue

    # פיצ'רים טקסטואליים והזרקת רעש
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text']).toarray()
    
    for i in range(len(df)):
        if df.iloc[i]['is_anomaly'] == 1:
            tfidf_matrix[i] = np.random.uniform(-1.0, 1.0, 500)

    features = torch.FloatTensor(tfidf_matrix).to(device)
    
    # נורמליזציה של הגרף
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, tfidf_matrix, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    # הרצת המודל
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj_tensor._indices())
    
    model = LLGC(nfeat=500, nclass=args.embedding_dim, drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    # זיהוי אנומליות
    print("Detecting anomalies...")
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    # סידור התוצאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    # --- הפלט לקובץ CSV ---
    output_filename = "full_anomaly_detection_results.csv"
    # אנחנו שומרים רק את העמודות הרלוונטיות כדי שהקובץ יהיה קריא
    columns_to_save = ['id', 'title', 'year', 'is_anomaly', 'anomaly_score']
    df_sorted[columns_to_save].to_csv(output_filename, index=False)
    
    print(f"\nSuccess! Full results for all {len(df)} nodes saved to: {output_filename}")
    
    # הצגת סיכום קצר למסך
    num_fakes = len(df_fakes)
    detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    print(f"Precision@{num_fakes}: {detected / num_fakes:.4f}")

if __name__ == "__main__":
    main()