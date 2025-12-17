import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
import random # נוסף לצורך קיבוע הזרע
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים מהקבצים הקיימים ב-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# פונקציה להבטחת דטרמיניזם (תוצאות קבועות)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # הבטחת דטרמיניזם בפעולות GPU (עלול להאט מעט את הביצועים)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Deterministic mode enabled with seed: {seed}")

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.15, help='Lower alpha for more smoothing')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_inject_anomalies():
    print("Loading data and analyzing context...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx:
                    G.add_edge(row['id'], str(ref))
        except: continue

    # א. TF-IDF משופר
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(
        max_features=1500, 
        stop_words='english',
        min_df=1,
        use_idf=True, 
        smooth_idf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df['text']).toarray()
    
    # ב. צפיפות סמנטית
    semantic_density = np.sum(tfidf_matrix, axis=1).reshape(-1, 1)
    
    # ג. נרמול לוגריתמי של הדרגה
    degrees = np.array([G.degree(n) for n in node_ids]).reshape(-1, 1)
    log_degrees = np.log1p(degrees)
    if log_degrees.max() > 0:
        log_degrees = log_degrees / log_degrees.max()
    
    # שילוב תכונות
    combined_features = np.hstack([tfidf_matrix, log_degrees, semantic_density])
    
    # נרמול שורות
    row_norms = np.linalg.norm(combined_features, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    combined_features = combined_features / row_norms

    features_tensor = torch.FloatTensor(combined_features).to(device)
    
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, combined_features, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features_tensor, adj_tensor, df, len(df_fakes)

def main():
    # קריאה לפונקציית קיבוע הזרע בתחילת הריצה
    set_seed(args.seed)
    
    features, adj, df, num_injected = load_and_inject_anomalies()
    
    # 4. PageRank Smoothing (Aggregation)
    # alpha נמוך (0.15) כפי שהוגדר בפרמטרים
    aggregator = PageRankAgg(K=args.K, alpha=0.8).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    # 5. Lorentzian Embedding (LLGC)
    # המשקולות של LorentzLinear יאותחלו כעת בצורה זהה בכל הרצה בזכות ה-set_seed
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                  drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    # 6. Isolation Forest
    clf = IsolationForest(contamination=0.01, random_state=args.seed)
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