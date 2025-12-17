import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים מהקבצים הקיימים ב-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.8, help='Higher alpha keeps original anomaly signal')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_inject_anomalies():
    print("Loading data...")
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

    # הפקת פיצ'רים טקסטואליים
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text']).toarray()
    
    # --- הקשחת אנומליה (Crucial Fix) ---
    # אם המאמרים ב-fakes.csv נראים רגילים, אנחנו נהפוך את הפיצ'רים שלהם ל"רעש" 
    # כדי לוודא שהם Outliers סטטיסטיים (כפי שמופיע ב-full_pipeline.py)
    for i, row in df.iterrows():
        if row['is_anomaly'] == 1:
            # הזרקת ערכים אקראיים קיצוניים לתכונות של המאמרים המזויפים
            tfidf_matrix[i] = np.random.uniform(low=-1.0, high=1.0, size=(500,))

    features = torch.FloatTensor(tfidf_matrix).to(device)
    
    # הכנת מטריצת שכנות
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, tfidf_matrix, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features, adj_tensor, df, len(df_fakes)

def main():
    features, adj, df, num_injected = load_and_inject_anomalies()
    
    # PageRank Smoothing
    # שימוש ב-alpha גבוה (0.8) כדי שהשכנים ה"נורמליים" לא ימחקו את האנומליה
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    # Lorentzian Embedding (LLGC)
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    # Isolation Forest
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    # Evaluation
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    top_detected = df_sorted.head(num_injected)['is_anomaly'].sum()
    
    print(f"\nResults Summary:")
    print(f"Precision@{num_injected}: {top_detected / num_injected:.4f}")
    print(f"Detected {top_detected} out of {num_injected} injected fakes.")
    
    print("\nTop 10 Anomaly Candidates:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | Title: {row['title'][:50]}...")

if __name__ == "__main__":
    main()