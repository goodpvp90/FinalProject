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
parser.add_argument('--alpha', type=float, default=0.15, help='Lower alpha for more smoothing')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_inject_anomalies():
    print("Loading data...")
    # טעינת המאמרים האמיתיים והמזויפים
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # 1. בניית הגרף (זיהוי קשרים)
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx:
                    G.add_edge(row['id'], str(ref))
        except: continue

    # 2. הפקת פיצ'רים טקסטואליים (TF-IDF)
    # אנחנו לא דוחפים ערכים מוגזמים, אלא נותנים ל-TF-IDF לעשות את שלו
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text']).toarray()
    
    # 3. הוספת פיצ'ר מבני: דרגת הצומת (Degree) - קריטי לזיהוי אנומליה ללא רעש
    # צמתים מזויפים הם לרוב מבודדים בגרף הציטוטים
    degrees = np.array([G.degree(n) for n in node_ids]).reshape(-1, 1)
    # נרמול הדרגה כדי שתהיה בטווח של הפיצ'רים האחרים (0 עד 1)
    if degrees.max() > 0:
        degrees = degrees / degrees.max()
    
    # שילוב הדרגה עם ה-TF-IDF
    combined_features = np.hstack([tfidf_matrix, degrees])
    
    features_tensor = torch.FloatTensor(combined_features).to(device)
    
    # הכנת מטריצת שכנות לנרמול (AugNormAdj)
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, combined_features, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features_tensor, adj_tensor, df, len(df_fakes)

def main():
    features, adj, df, num_injected = load_and_inject_anomalies()
    
    # 4. PageRank Smoothing (Aggregation)
    # צמתים רגילים "יתערבבו" עם השכנים שלהם ויהפכו לממוצע הקבוצה.
    # צמתים מבודדים (אנומליות) יישארו עם הפיצ'רים המקוריים שלהם, מה שיבליט אותם.
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    # 5. Lorentzian Embedding (LLGC)
    # המודל משליך את הנתונים למרחב היפרבולי, שטוב במיוחד לזיהוי היררכיות וחריגות.
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                  drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        # ההטמעה (Embedding) תדגיש את המרחק של האנומליות מהמרכז
        embeddings = model(x_smooth).cpu().numpy()
        
    # 6. Isolation Forest - זיהוי ה-Outliers במרחב ההטמעה
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    # הערכת ביצועים
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    top_detected = df_sorted.head(num_injected)['is_anomaly'].sum()
    
    print(f"\n--- Detection Results (Without Noise Injection) ---")
    print(f"Precision@{num_injected}: {top_detected / num_injected:.4f}")
    print(f"Detected {top_detected} out of {num_injected} injected fakes.")
    
    print("\nTop 10 Anomaly Candidates Found:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | Degree: {row['references']} | Title: {row['title'][:50]}...")

if __name__ == "__main__":
    main()