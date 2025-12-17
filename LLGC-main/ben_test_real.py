import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler # נוסף לטיפול ב-Hubs

# ייבוא הרכיבים מהקבצים הקיימים ב-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Deterministic mode enabled with seed: {seed}")

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1, help='Lower alpha for stronger smoothing')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_inject_anomalies():
    print("Loading data and applying structural filters...")
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

    # 1. TF-IDF עם Sublinear scaling
    # זה מפחית את ההשפעה של מילים שחוזרות המון פעמים (כמו ב-Workflows הגרמני)
    df['text'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words='english',
        min_df=1,
        sublinear_tf=True, # קריטי לנטרול חריגים סמנטיים קיצוניים
        use_idf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df['text']).toarray()
    
    # 2. שימוש ב-RobustScaler עבור הדרגה (Degree)
    # בניגוד לנרמול רגיל, הוא מתעלם מ-Outliers (כמו מאמרי סקר עם 1000 ציטוטים)
    degrees = np.array([G.degree(n) for n in node_ids]).reshape(-1, 1)
    scaler = RobustScaler()
    robust_degrees = scaler.fit_transform(np.log1p(degrees))
    
    # 3. הוספת פיצ'ר אורך טקסט (לוגריתמי)
    # זיופים הם לרוב קצרים/ארוכים בצורה חשודה
    text_len = df['text'].str.len().values.reshape(-1, 1)
    text_len = np.log1p(text_len)
    text_len = text_len / text_len.max()

    # שילוב: טקסט (משקל גבוה), דרגה (משקל נמוך כדי לא לסמן Hubs), אורך טקסט
    combined_features = np.hstack([tfidf_matrix * 2.0, robust_degrees * 0.5, text_len])
    
    # נרמול שורות - מבטיח שכל הצמתים נמצאים על אותה "קליפה" לפני ההטמעה
    row_norms = np.linalg.norm(combined_features, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    combined_features = combined_features / row_norms

    features_tensor = torch.FloatTensor(combined_features).to(device)
    
    # הכנת מטריצת שכנות
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, combined_features, normalization="AugNormAdj")
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    return features_tensor, adj_tensor, df, len(df_fakes)

def main():
    set_seed(args.seed)
    features, adj, df, num_injected = load_and_inject_anomalies()
    
    # 4. Aggressive Smoothing (Alpha=0.1)
    # זה הצעד הכי חשוב: מאמרים שמקושרים למאמרים דומים יתאחדו למרכז הענן.
    # מאמרים מבודדים (fakes) יישארו רחוקים מהמרכז.
    aggregator = PageRankAgg(K=args.K, alpha=0.1).to(device) # שינוי ל-0.1
    x_smooth, _ = aggregator(features, adj._indices())
    
    # 5. Lorentzian Embedding (LLGC)
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                  drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = model(x_smooth).cpu().numpy()
        
    # 6. Isolation Forest
    # הגדרת contamination=0.05 בהתאם ליחס ה-fakes הידוע
    clf = IsolationForest(contamination=0.05, random_state=args.seed, n_estimators=200)
    clf.fit(embeddings)
    df['anomaly_score'] = -clf.decision_function(embeddings)
    
    # הערכת ביצועים
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    top_detected = df_sorted.head(num_injected)['is_anomaly'].sum()
    
    print(f"\n--- FINAL Results (Alpha 0.1 + Robust Scaling) ---")
    print(f"Precision@{num_injected}: {top_detected / num_injected:.4f}")
    print(f"Detected {top_detected} out of {num_injected} injected fakes.")
    
    print("\nTop 10 Anomaly Candidates Found:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | Title: {row['title'][:60]}...")

if __name__ == "__main__":
    main()