import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים מהקבצים הקיימים בתיקייה
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# הגדרת פרמטרים אופטימליים לזיהוי אנומליות מבניות
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of Lorentzian embeddings')
parser.add_argument('--K', type=int, default=10, help='Number of propagation steps')
# Alpha נמוך (0.1) מאפשר למידע מהגרף לזרום ולזהות אנומליות מבניות
parser.add_argument('--alpha', type=float, default=0.1, help='Lower alpha emphasizes graph structure')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_real_data():
    """
    טעינת הדאטאסט המקורי ללא הזרקות
    """
    print("Loading original dataset...")
    # טעינת קובץ הנתונים המקורי שלך
    df = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df['id'] = df['id'].astype(str)
    
    node_ids = df['id'].tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # בניית גרף הציטוטים
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    print("Building citation graph (Edges based on references)...")
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx:
                    G.add_edge(row['id'], str(ref))
        except:
            continue

    # הפקת פיצ'רים טקסטואליים בסיסיים
    print("Vectorizing titles and abstracts...")
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text']).toarray()
    
    features = torch.FloatTensor(tfidf_matrix).to(device)
    
    # הכנת מטריצת שכנות נורמלית
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, tfidf_matrix, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features, adj_tensor, df

def main():
    # 1. טעינת נתונים
    features, adj, df = load_real_data()
    
    # 2. PageRank Aggregation - שלב ה"החלקה" המבני
    print(f"Running PageRank smoothing (Alpha={args.alpha}, K={args.K})...")
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, adj._indices())
    
    # 3. Lorentzian Projection (LLGC) - הטלה למרחב היפרבולי
    # המודל משמש כ-Encoder שממפה את הקשרים למרחב לורנץ
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        # הפקת ה-Embeddings הסופיים
        embeddings = model(x_smooth).cpu().numpy()
        
    # 4. זיהוי חריגים באמצעות Isolation Forest
    print("Detecting real anomalies in the dataset...")
    clf = IsolationForest(contamination=0.01, random_state=args.seed, n_jobs=-1)
    clf.fit(embeddings)
    
    # חישוב ציוני אנומליה (גבוה יותר = אנומלי יותר)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    # 5. הצגת תוצאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 10 NATURAL ANOMALY CANDIDATES FOUND:")
    print("="*50)
    
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        print(f"#{i+1} | Score: {row['anomaly_score']:.4f}")
        print(f"    ID: {row['id']}")
        print(f"    Title: {row['title']}")
        print(f"    Year: {row.get('year', 'N/A')}")
        print("-" * 30)

    # שמירת התוצאות לקובץ CSV לניתוח מעמיק
    output_file = "real_detection_results.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")

if __name__ == "__main__":
    main()