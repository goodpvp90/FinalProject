import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

# ייבוא הרכיבים מהקבצים הקיימים
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
from manifolds.lorentzian import Lorentzian

def main():
    # 1. טעינת הנתונים (מקור + הזרקת fakes)
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)

    # 2. בניית תכונות (ללא רעש מלאכותי)
    # טקסט (TF-IDF)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    x_text = tfidf.fit_transform(df['title'].fillna('') + " " + df['abstract'].fillna('')).toarray()
    
    # FOS (One-Hot) - המפתח לזיהוי חוסר התאמה קטגורי
    fos_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_fos = fos_enc.fit_transform(df[['fos.name']].fillna('Other'))
    
    # איחוד (ללא ניפוח ערכים)
    x_all = np.hstack([x_text, x_fos])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.FloatTensor(x_all).to(device)

    # 3. בניית הגרף
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
    edge_index = sparse_mx_to_torch_sparse_tensor(adj)._indices().to(device)

    # 4. עיבוד במרחב לורנץ
    manifold = Lorentzian()
    c = torch.tensor([1.0]).to(device)
    
    # אגרגציה של השכנים (PageRank) - אלפא=0.5 לאיזון בין הצומת לסביבה
    aggregator = PageRankAgg(K=10, alpha=0.5).to(device)
    x_smooth, _ = aggregator(features, edge_index)
    
    # אתחול המודל עם תיקון ה-TypeError (הוספת drop_out=0.0)
    model = LLGC(nfeat=features.size(1), nclass=64, drop_out=0.0, use_bias=True).to(device)
    model.eval()

    with torch.no_grad():
        # הפקת Embeddings (במישור המשיק)
        embeddings = model(x_smooth).cpu().numpy()
        
        # חישוב ה-Mismatch הלורנציאני:
        # מרחק בין המאמר המקורי (במרחב לורנץ) לבין הייצוג הממוצע של שכניו
        x_orig_loren = manifold.normalize_input(features, c)
        x_smooth_loren = manifold.normalize_input(x_smooth, c)
        mismatch_dist = manifold.induced_distance(x_orig_loren, x_smooth_loren, c).cpu().numpy().flatten()

    # 5. זיהוי אנומליות
    # אנומליה היא צומת שהוא גם חריג גלובלית (Isolation Forest) 
    # וגם לא מתאים לסביבה שלו (Mismatch Distance)
    clf = IsolationForest(contamination='auto', random_state=42)
    clf.fit(embeddings)
    if_score = -clf.decision_function(embeddings)
    
    # ציון סופי משולב
    df['anomaly_score'] = if_score + (mismatch_dist / (mismatch_dist.max() + 1e-6))

    # 6. הצגת תוצאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    num_fakes = len(df_fakes)
    detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print(f"\nFinal Results:\nDetected {detected} / {num_fakes} fakes.")
    print(f"Precision@{num_fakes}: {detected/num_fakes:.4f}")
    
    # שמירה ל-CSV לניתוח
    df_sorted[['id', 'title', 'fos.name', 'is_anomaly', 'anomaly_score']].to_csv("deep_detection_results.csv", index=False)
    print("\nTop 5 Candidates:")
    print(df_sorted[['title', 'fos.name', 'is_anomaly', 'anomaly_score']].head(5))

if __name__ == "__main__":
    main()