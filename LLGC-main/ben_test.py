import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from time import perf_counter

# ייבוא הרכיבים מהקוד המקורי
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# הגדרת פרמטרים (בדומה ל-run_cora.py)
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of Lorentzian embeddings.')
parser.add_argument('--K', type=int, default=10, help='Number of propagation steps.')
parser.add_argument('--alpha', type=float, default=0.15, help='Alpha for PageRank.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--trials', type=int, default=5, help='Number of trials.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"

def load_custom_data():
    """
    טעינת הדאטאסט המקורי והזרקת האנומליות מקובץ fakes.csv
    """
    # 1. טעינת הקבצים
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    # סימון מי אנומליה (Ground Truth)
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    
    # איחוד דאטאסטים
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    print(f"Loaded {len(df_real)} real papers and {len(df_fakes)} fake papers.")
    
    # 2. בניית הגרף (מבוסס על עמודת ה-references)
    G = nx.Graph()
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    G.add_nodes_from(node_ids)
    
    for _, row in df.iterrows():
        try:
            # המרת מחרוזת הרשימה לרשימה אמיתית
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                ref_str = str(ref)
                if ref_str in node_to_idx:
                    G.add_edge(row['id'], ref_str)
        except:
            continue
            
    # 3. הפקת פיצ'רים (TF-IDF על כותרת ותקציר)
    df['combined_text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_features = vectorizer.fit_transform(df['combined_text']).toarray()
    features = torch.FloatTensor(tfidf_features).to(device)
    
    # 4. הכנת מטריצת שכנות עבור הפרופגציה
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    # שימוש בפונקציות העזר מ-utils לעיבוד המטריצה
    adj, _ = preprocess_citation(adj, tfidf_features, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features, adj_tensor, df, len(df_fakes)

def main():
    # טעינה והזרקה
    features, adj, df, num_injected = load_custom_data()
    
    # שלב 1: PageRank Smoothing (בדומה ל-run_cora)
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    # PageRankAgg מצפה ל-indices בגרסת ה-forward שלו
    x_smooth, _ = aggregator(features, adj._indices())
    
    # שלב 2: הפקת Embeddings לורנציאניים (LLGC)
    # אנחנו משתמשים במודל כ-Feature Extractor למרחב היפרבולי
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        # המודל מחזיר את הייצוג ב-Tangent Space של מרחב לורנץ
        embeddings = model(x_smooth).cpu().numpy()
        
    # שלב 3: זיהוי אנומליות באמצעות Isolation Forest
# שלב 3: זיהוי אנומליות באמצעות Isolation Forest
    print("Running Isolation Forest on embeddings...")
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    
    # --- התיקון: אימון המודל על ה-embeddings שהופקו ---
    clf.fit(embeddings) 
    # --------------------------------------------------

    # ציון אנומליה (נמוך יותר = אנומלי יותר, לכן נהפוך סימן)
    anomaly_scores = -clf.decision_function(embeddings)
    df['anomaly_score'] = anomaly_scores
    
    # שלב 4: הערכת ביצועים
    # נמיין לפי הציון הגבוה ביותר (הכי אנומלי)
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    # נבדוק כמה מהמוזרקים נמצאים ב-Top N (כאשר N הוא מספר המוזרקים)
    top_candidates = df_sorted.head(num_injected)
    detected_injected = top_candidates['is_anomaly'].sum()
    
    precision_at_k = detected_injected / num_injected
    
    print("\n" + "="*30)
    print(f"Results for this run:")
    print(f"Total injected fakes: {num_injected}")
    print(f"Fakes detected in Top {num_injected}: {detected_injected}")
    print(f"Precision@{num_injected}: {precision_at_k:.4f}")
    
    # הצגת דוגמאות
    print("\nTop 5 Most Anomalous Papers Found:")
    for i, row in df_sorted.head(5).iterrows():
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"{label} Score: {row['anomaly_score']:.4f} | ID: {row['id']} | Title: {row['title'][:60]}...")
        
    return precision_at_k

if __name__ == "__main__":
    results = []
    for t in range(args.trials):
        print(f"\n--- Trial {t+1}/{args.trials} ---")
        results.append(main())
    
    print("\n" + "="*30)
    print(f"Final Average Precision across {args.trials} trials: {np.mean(results):.5f}")