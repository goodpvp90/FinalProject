import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים מה-Repo
from model import PageRankAgg
from manifolds.lorentzian import Lorentzian
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# הגדרות
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--K', type=int, default=10, help='מספר צעדי הפרופגציה')
parser.add_argument('--alpha', type=float, default=0.15, help='Teleport probability')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifold = Lorentzian()
CURVATURE = 1.0

def build_dataset():
    print("Loading datasets...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. עיבוד טקסט (TF-IDF)
    print("Vectorizing text...")
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    x_text = tfidf.fit_transform(df['text']).toarray()
    
    # 2. עיבוד FOS (One-Hot) - המפתח לזיהוי חוסר התאמה
    print("Encoding FOS categories...")
    fos_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_fos = fos_enc.fit_transform(df[['fos.name']].fillna('Other'))
    
    # איחוד תכונות
    x_all = np.hstack([x_text, x_fos])
    features = torch.FloatTensor(x_all).to(device)
    
    # 3. בניית גרף
    print("Building citation graph...")
    node_ids = df['id'].tolist()
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
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
    adj, _ = preprocess_citation(adj, x_all, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features, adj_tensor, df

def main():
    # טעינה
    features, adj, df = build_dataset()
    
    # שלב א': הטלה למרחב לורנץ (Mapping to Manifold)
    # אנחנו מעבירים את הפיצ'רים הגולמיים למרחב ההיפרבולי
    x_lorentz = manifold.normalize_input(features, CURVATURE)
    
    # שלב ב': החלקה גרפית (Context Aggregation)
    # PageRankAgg מבצע low-pass filter - הוא מוצא איך הצומת "אמור" להיראות לפי השכנים שלו
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    # שימוש ב-lorentz_centroid כפי שמופיע במחלקה לטובת אגרגציה נכונה במרחב לורנץ
    # (הערה: PageRankAgg המקורי עובד במישור המשיק או אוקלידי, כאן נשתמש בגרסה המשופרת)
    x_smooth = aggregator(x_lorentz, adj._indices())[0]
    
    # שלב ג': חישוב "ציון ההפתעה" (Lorentzian Mismatch)
    # אנומליה היא צומת שרחוק מאוד מהייצוג הממוצע של השכנים שלו
    with torch.no_grad():
        # חישוב מרחק לורנציאני בין המקור לבין המוחלק
        dists = manifold.induced_distance(x_lorentz, x_smooth, CURVATURE)
        dists = dists.cpu().numpy().flatten()
    
    # שלב ד': Isolation Forest על השילוב של ה-Embeddings והמרחקים
    print("Running Isolation Forest on Lorentzian mismatch features...")
    # אנחנו נותנים ל-IF את ה-embeddings המוחלקים ביחד עם המרחק (residual)
    combined_emb = np.hstack([x_smooth.cpu().numpy(), dists.reshape(-1, 1)])
    
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    clf.fit(combined_emb)
    
    # שילוב של ציון ה-IF ומרחק המיסמאץ' לציון סופי
    if_scores = -clf.decision_function(combined_emb)
    df['anomaly_score'] = if_scores + (dists / dists.max()) # נרמול ושילוב
    
    # שלב ה': פלט וסיכום
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    output_file = "lorentzian_mismatch_results.csv"
    df_sorted[['id', 'title', 'fos.name', 'is_anomaly', 'anomaly_score']].to_csv(output_file, index=False)
    
    num_fakes = df['is_anomaly'].sum()
    top_detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Lorentzian Contextual Detection Results:")
    print(f"Precision@{num_fakes}: {top_detected / num_fakes:.4f} ({top_detected}/{num_fakes})")
    print(f"Full results saved to: {output_file}")
    
    print("\nTop 10 Anomalies Found:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:50]}...")
    print("="*50)

if __name__ == "__main__":
    main()