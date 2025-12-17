import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from time import perf_counter

# ייבוא הרכיבים מהקבצים הקיימים ב-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# הגדרת פרמטרים
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of Lorentzian embeddings.')
parser.add_argument('--K', type=int, default=10, help='Number of propagation steps.')
# שינוי ל-0.5 כדי למנוע "מחיקה" של האנומליות ע"י השכנים
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for PageRank (0.5 is better for anomalies).')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--trials', type=int, default=1, help='Number of trials.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = "cuda" if args.cuda else "cpu"

def load_custom_data():
    """
    טעינת הדאטאסט המקורי והזרקת האנומליות מקובץ fakes.csv
    """
    print("Loading data files...")
    # טעינת הקבצים (וודא שהם בתיקיית העבודה)
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    # סימון Ground Truth
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    
    # איחוד דאטאסטים
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    print(f"Dataset Stats: {len(df_real)} real, {len(df_fakes)} fake.")
    
    # בניית הגרף
    G = nx.Graph()
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    G.add_nodes_from(node_ids)
    
    print("Building citation graph...")
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                ref_str = str(ref)
                if ref_str in node_to_idx:
                    G.add_edge(row['id'], ref_str)
        except:
            continue
            
    # הפקת פיצ'רים טקסטואליים (TF-IDF)
    print("Vectorizing text features...")
    df['combined_text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_features = vectorizer.fit_transform(df['combined_text']).toarray()
    features = torch.FloatTensor(tfidf_features).to(device)
    
    # הכנת מטריצת שכנות (Adjacency Matrix)
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, tfidf_features, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features, adj_tensor, df, len(df_fakes)

def main():
    # 1. טעינה
    features, adj, df, num_injected = load_custom_data()
    
    # 2. PageRank Aggregation (החלקת תכונות על הגרף)
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    # 3. יצירת Embeddings במרחב לורנץ (LLGC)
    # שים לב: המודל כאן פועל ללא אימון (Forward בלבד) כ-Projection רנדומלי למרחב היפרבולי
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    # 4. זיהוי אנומליות עם Isolation Forest
    print("Fitting Isolation Forest...")
    clf = IsolationForest(contamination='auto', random_state=args.seed, n_jobs=-1)
    
    # אימון המודל על ה-Embeddings (תיקון לשגיאת ה-NotFittedError)
    clf.fit(embeddings)
    
    # הפקת ציוני אנומליה (הפיכת סימן כך שציון גבוה = אנומלי יותר)
    anomaly_scores = -clf.decision_function(embeddings)
    df['anomaly_score'] = anomaly_scores
    
    # 5. ניתוח תוצאות וחישוב Precision
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    top_candidates = df_sorted.head(num_injected)
    detected_injected = top_candidates['is_anomaly'].sum()
    
    precision_at_k = detected_injected / num_injected
    
    print("\n" + "="*40)
    print(f"Results Summary:")
    print(f"Precision@{num_injected}: {precision_at_k:.4f}")
    print(f"Detected {detected_injected} out of {num_injected} injected fakes.")
    
    print("\nTop 10 Anomaly Candidates:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | ID: {row['id']} | Title: {row['title'][:50]}...")
    print("="*40)
    
    return precision_at_k

if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    results = []
    for t in range(args.trials):
        print(f"\n--- Starting Trial {t+1}/{args.trials} ---")
        results.append(main())
    
    if args.trials > 1:
        print(f"\nFinal Average Precision: {np.mean(results):.5f}")