import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים מהקבצים הקיימים
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
from manifolds.lorentzian import Lorentzian

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifold = Lorentzian()
CURVATURE = torch.tensor([1.0]).to(device)

def load_and_analyze():
    print("Loading data and analyzing graph structure...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. ניתוח מבני (Structural Features)
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
    
    # חישוב דרגה (Degree) וצפיפות מקומית - אנומליות הן לרוב בדרגה נמוכה
    degrees = dict(G.degree())
    df['degree'] = df['id'].map(degrees).fillna(0)
    df['log_degree'] = np.log1p(df['degree'])
    
    # 2. הנדסת תכונות עמוקה (Deep Features)
    print("Extracting textual and categorical features...")
    # טקסט (TF-IDF)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    x_text = tfidf.fit_transform(df['title'].fillna('') + " " + df['abstract'].fillna('')).toarray()
    
    # FOS (One-Hot) - קריטי לזיהוי הזרקות (למשל Cooking vs Computer Science)
    fos_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_fos = fos_enc.fit_transform(df[['fos.name']].fillna('Other'))
    
    # שילוב וקטורים: נותנים משקל יחסי גבוה ל-FOS ולמבנה
    # (ללא ניפוח ערכים קיצוניים, רק שקלול חשיבות)
    x_combined = np.hstack([x_text, x_fos * 2.0, df[['log_degree']].values * 3.0])
    
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x_combined)
    features = torch.FloatTensor(x_normalized).to(device)
    
    # הכנת מטריצת שכנות
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, x_normalized, normalization="AugNormAdj")
    edge_index = sparse_mx_to_torch_sparse_tensor(adj)._indices().to(device)
    
    return features, edge_index, df, adj

def main():
    features, edge_index, df, adj_mx = load_and_analyze()
    
    # שלב א': הטלה למרחב לורנץ בממד 256 (בדומה ל-run_cora)
    # אנחנו משתמשים במודל ה-LLGC שביקשת
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    # שלב ב': אגרגציה מבוססת קונטקסט (PageRank Smoothing)
    # המטרה: לראות איך המאמר "נראה" לעומת הסביבה שלו
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, edge_index)
    
    with torch.no_grad():
        # 1. הפקת Embeddings (במישור המשיק)
        embeddings = model(x_smooth)
        
        # 2. חישוב Lorentzian Residual (המרחק בין המאמר לסביבתו)
        # זהו המדד הכי חזק לאנומליות שאינן מלאכותיות
        x_orig_loren = manifold.normalize_input(features, CURVATURE)
        x_context_loren = manifold.normalize_input(x_smooth, CURVATURE)
        # מרחק גיאומטרי במרחב ההיפרבולי
        mismatch_dist = manifold.induced_distance(x_orig_loren, x_context_loren, CURVATURE)
        
        # המרה ל-Numpy
        emb_np = embeddings.cpu().numpy()
        mismatch_np = mismatch_dist.cpu().numpy().reshape(-1, 1)

    # שלב ג': זיהוי אנומליות סופי
    print(f"Running Detection on {args.embedding_dim}D space...")
    # נשלב את ה-Embeddings עם המרחק הלורנציאני ועם הדרגה של הצומת
    detection_space = np.hstack([emb_np, mismatch_np, df[['log_degree']].values])
    
    clf = IsolationForest(contamination='auto', random_state=args.seed, n_jobs=-1)
    clf.fit(detection_space)
    
    # חישוב ציון אנומליה (גבוה יותר = אנומלי יותר)
    df['anomaly_score'] = -clf.decision_function(detection_space)
    
    # שלב ד': פלט לקובץ
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    # שמירת כל הציונים לקובץ CSV
    output_csv = "all_papers_anomaly_results.csv"
    columns_to_show = ['id', 'title', 'fos.name', 'degree', 'is_anomaly', 'anomaly_score']
    df_sorted[columns_to_show].to_csv(output_csv, index=False)
    
    # ולידציה על ההזרקות
    num_fakes = df['is_anomaly'].sum()
    top_detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Realistic Detection Summary (256D):")
    print(f"Precision@{num_fakes}: {top_detected / num_fakes:.4f} ({top_detected}/{num_fakes})")
    print(f"Results for all papers saved to: {output_csv}")
    
    print("\nTop 10 Detected Anomalies:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:40]}...")
    print("="*50)

if __name__ == "__main__":
    main()