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
# אלפא נמוך הופך את הגילוי ליותר מבני (פחות מלאכותי)
parser.add_argument('--alpha', type=float, default=0.2) 
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_realistic_data():
    print("Loading data for realistic detection...")
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
    print("Building citation graph...")
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx:
                    G.add_edge(row['id'], str(ref))
        except: continue

    # הפקת פיצ'רים טקסטואליים אמיתיים (ללא הזרקת רעש!)
    print("Generating TF-IDF features from actual text...")
    df['combined_text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    # הפיצ'רים של ה-fakes יהיו מבוססים על הטקסט שכתוב ב-fakes.csv
    tfidf_matrix = vectorizer.fit_transform(df['combined_text']).toarray()
    
    features = torch.FloatTensor(tfidf_matrix).to(device)
    
    # הכנת מטריצת שכנות
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, tfidf_matrix, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features, adj_tensor, df, len(df_fakes)

def main():
    # 1. טעינה
    features, adj, df, num_injected = load_realistic_data()
    
    # 2. PageRank Aggregation
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    # 3. Lorentzian Projection (LLGC)
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    # 4. זיהוי אנומליות
    print("Detecting anomalies (Realistic mode)...")
    # contamination=auto יאפשר למודל למצוא כמה אנומליות שהוא "מרגיש"
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    # 5. שמירת תוצאות מלאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    output_file = "realistic_anomaly_results.csv"
    df_sorted[['id', 'title', 'is_anomaly', 'anomaly_score']].to_csv(output_file, index=False)
    
    # 6. הצגת סיכום
    top_candidates = df_sorted.head(num_injected)
    detected_fakes = top_candidates['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Realistic Detection Results:")
    print(f"Fakes detected in Top {num_injected}: {detected_fakes}/{num_injected}")
    print(f"Precision@{num_injected}: {detected_fakes / num_injected:.4f}")
    print(f"Full results saved to: {output_file}")
    
    print("\nTop 10 Overall Anomalies (Natural + Injected):")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        source = "[INJECTED]" if row['is_anomaly'] == 1 else "[NATURAL]"
        print(f"#{i+1} {source} Score: {row['anomaly_score']:.4f} | Title: {row['title'][:60]}...")
    print("="*50)

if __name__ == "__main__":
    main()