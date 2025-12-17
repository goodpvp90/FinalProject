import pandas as pd
import numpy as np
import torch
import networkx as nx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, precision_score, f1_score

# ייבוא רכיבים מתוך ה-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

# --- הפונקציה המתוקנת ---
def build_adjacency_matrix(df):
    print("--- Building graph structure (Fixed Isolated Nodes) ---")
    node_to_idx = {node_id: i for i, node_id in enumerate(df['id'])}
    edges = []
    
    for i, row in df.iterrows():
        source_idx = node_to_idx[row['id']]
        for target_id in row['references']:
            if target_id in node_to_idx:
                edges.append((source_idx, node_to_idx[target_id]))
    
    # יצירת גרף ריק והוספת כל הצמתים מראש
    G = nx.Graph()
    G.add_nodes_from(range(len(df))) # מבטיח שכל הצמתים מ-0 עד N-1 קיימים בגרף
    G.add_edges_from(edges)
    
    # כעת הפונקציה תעבוד ללא שגיאה כי כל רשימת ה-nodelist קיימת ב-G
    adj = nx.adjacency_matrix(G, nodelist=range(len(df)))
    
    # נרמול המטריצה כפי שמוגדר ב-utils.py
    adj_normalized, _ = preprocess_citation(adj, np.zeros((len(df), 1)))
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

# --- שאר הלוגיקה לשיפור הדיוק ---

def extract_features(df):
    print("--- Extracting features (Semantic + Structural) ---")
    df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
    X_text = vectorizer.fit_transform(df['text_combined']).toarray()
    
    scaler = StandardScaler()
    # הוספת num_refs כפיצ'ר קריטי לזיהוי אנומליות מבניות
    X_num = scaler.fit_transform(df[['year', 'n_citation', 'num_refs']].fillna(0))
    
    # תמיכה בגרסאות שונות של sklearn
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_fos = encoder.fit_transform(df[['fos.name']].fillna('Unknown'))
    
    return torch.FloatTensor(np.hstack((X_text, X_num, X_fos)))

def run_anomaly_detection():
    # קבצי המקור ב-repo
    real_file = "final_filtered_by_fos_and_reference.csv"
    fakes_file = "fakes.csv"
    
    # 1. הכנת נתונים (עם פונקציית safe_parse_list מהשלב הקודם)
    def safe_parse_list(val):
        if pd.isna(val) or val == "" or val == "[]": return []
        return re.findall(r'[a-zA-Z0-9]+', str(val))

    df_real = pd.read_csv(real_file)
    df_fakes = pd.read_csv(fakes_file)
    df_real['is_anomaly'], df_fakes['is_anomaly'] = 0, 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    df['references'] = df['references'].apply(safe_parse_list)
    df['num_refs'] = df['references'].apply(len)
    
    features = extract_features(df)
    adj = build_adjacency_matrix(df)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, adj = features.to(device), adj.to(device)

    # 2. שימוש במודל ה-LLGC להטמעה היפרבולית
    # הגברת K ל-30 כדי להבליט את הבידוד המבני של ה-Fakes
    aggregator = PageRankAgg(K=30, alpha=0.1).to(device)
    x_agg, _ = aggregator(features, adj._indices())
    
    model = LLGC(nfeat=features.shape[1], nclass=128, drop_out=0.0, use_bias=1).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model(x_agg).cpu().numpy()

    # 3. זיהוי אנומליות - Isolation Forest על שילוב של מבנה ותוכן
    contamination = len(df[df['is_anomaly'] == 1]) / len(df)
    combined_input = np.hstack((embeddings, features.cpu().numpy()))
    
    clf = IsolationForest(n_estimators=500, contamination=contamination, random_state=42)
    df['prediction'] = [1 if p == -1 else 0 for p in clf.fit_predict(combined_input)]

    # 4. תוצאות
    recall = recall_score(df['is_anomaly'], df['prediction'])
    print("\n" + "═"*40)
    print(f"Recall (Fake Detection): {recall:.2%}")
    print(f"Precision: {precision_score(df['is_anomaly'], df['prediction']):.2%}")
    print("═"*40)

    if recall >= 0.85:
        print("✅ Target Reached! The model effectively identifies injected fakes.")

if __name__ == "__main__":
    run_anomaly_detection()