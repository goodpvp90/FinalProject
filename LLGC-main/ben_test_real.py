import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

# ייבוא הרכיבים המקוריים מה-Repo שלך
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
from manifolds.lorentzian import Lorentzian

# הגדרת פרמטרים (בדומה ל-run_cora.py)
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=16, help='ממד נמוך עוזר בזיהוי אנומליות מבניות')
parser.add_argument('--K', type=int, default=10, help='מספר צעדי הפרופגציה')
parser.add_argument('--alpha', type=float, default=0.5, help='איזון בין התכונות העצמיות לגרף')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_custom_data():
    print("Loading datasets...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    # סימון Ground Truth
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. תכונות טקסטואליות (TF-IDF)
    print("Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=400, stop_words='english')
    x_text = tfidf.fit_transform(df['title'].fillna('') + " " + df['abstract'].fillna('')).toarray()
    
    # 2. תכונות קטגוריות (FOS) - המפתח לזיהוי ה-Fakes
    # מכיוון של-Fakes יש FOS ייחודי (כמו Cooking), קידוד One-Hot יצור סיגנל חזק
    print("Encoding FOS...")
    fos_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_fos = fos_enc.fit_transform(df[['fos.name']].fillna('Unknown'))
    
    # איחוד תכונות: טקסט + FOS
    x_all = np.hstack([x_text, x_fos])
    features = torch.FloatTensor(x_all).to(device)
    
    # 3. בניית גרף הציטוטים
    print("Building citation graph...")
    node_ids = df['id'].tolist()
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                ref_str = str(ref)
                if ref_str in node_to_idx:
                    G.add_edge(row['id'], ref_str)
        except: continue
        
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, x_all, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features, adj_tensor, df

def main():
    # שלב א': הכנת נתונים
    features, adj, df = prepare_custom_data()
    
    # שלב ב': הרצת מודל LLGC (בדומה ל-run_cora)
    # 1. PageRank Aggregation (Smoothing)
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    # x_smooth הוא הייצוג של המאמר "בקונטקסט" של השכנים שלו
    x_smooth, _ = aggregator(features, adj._indices())
    
    # 2. Lorentzian Transformation
    # המרת התכונות המוחלקות למרחב לורנץ בעזרת שכבת ה-LLGC
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        # הפקת Embeddings לורנציאניים
        embeddings = model(x_smooth).cpu().numpy()
        
    # שלב ג': זיהוי אנומליות בעזרת Isolation Forest על ה-Embeddings
    # ה-IF יחפש נקודות שנמצאות ב"קצוות" או באזורים דלילים של המרחב ההיפרבולי
    print("Running Isolation Forest on Lorentzian embeddings...")
    clf = IsolationForest(contamination='auto', random_state=args.seed, n_jobs=-1)
    
    # אימון וחישוב ציון (הפיכת סימן כך שציון גבוה = אנומלי יותר)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    # שלב ד': ניתוח תוצאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    # חישוב דיוק (כמה מזויפים נכנסו ל-Top N)
    num_fakes = df['is_anomaly'].sum()
    top_candidates = df_sorted.head(num_fakes)
    detected = top_candidates['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Results Summary:")
    print(f"Detected {detected} out of {num_fakes} injected fakes.")
    print(f"Precision@{num_fakes}: {detected / num_fakes:.4f}")
    
    # הצגת ה-Top 10 לזיהוי אנומליות טבעיות ומזויפות
    print("\nTop 10 Anomaly Candidates:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:50]}...")
    print("="*50)
    
    # שמירה ל-CSV
    df_sorted[['id', 'title', 'fos.name', 'is_anomaly', 'anomaly_score']].to_csv("anomaly_results.csv", index=False)
    print("Full results saved to anomaly_results.csv")

if __name__ == "__main__":
    main()