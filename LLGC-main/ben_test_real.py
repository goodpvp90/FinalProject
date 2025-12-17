import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, precision_score, f1_score

# ייבוא רכיבים מתוך ה-repo הקיים
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

def load_and_prepare_data(real_path, fakes_path):
    print(f"--- Loading datasets: {real_path} and {fakes_path} ---")
    df_real = pd.read_csv(real_path)
    df_fakes = pd.read_csv(fakes_path)
    
    # סימון לייבלים לולידציה בלבד (0 - תקין, 1 - אנומליה)
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    
    # איחוד הנתונים
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # עיבוד עמודת ה-references (הפיכה מרשימת טקסט לרשימה פייתונית)
    df['references'] = df['references'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    
    return df

def extract_features(df):
    print("--- Extracting features (Text + Metadata) ---")
    # 1. טקסט: כותרת + תקציר
    df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_text = vectorizer.fit_transform(df['text_combined']).toarray()
    
    # 2. נומרי: שנה וכמות ציטוטים
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[['year', 'n_citation']].fillna(0))
    
    # 3. קטגורי: Field of Study
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_fos = encoder.fit_transform(df[['fos.name']].fillna('Unknown'))
    
    # שילוב הכל למטריצת פיצ'רים אחת
    X_static = np.hstack((X_text, X_num, X_fos))
    return torch.FloatTensor(X_static)

def build_adjacency_matrix(df):
    print("--- Building graph structure ---")
    node_to_idx = {node_id: i for i, node_id in enumerate(df['id'])}
    edges = []
    
    for i, row in df.iterrows():
        source_idx = node_to_idx[row['id']]
        for target_id in row['references']:
            if target_id in node_to_idx:
                edges.append((source_idx, node_to_idx[target_id]))
    
    # בניית מטריצת שכנות דלילה (Sparse Matrix)
    adj = nx.adjacency_matrix(nx.Graph(edges), nodelist=range(len(df)))
    # נרמול המטריצה כפי שנעשה ב-Cora
    adj_normalized, _ = preprocess_citation(adj, np.zeros((len(df), 1)))
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def run_anomaly_detection():
    # פרמטרים
    real_file = "final_filtered_by_fos_and_reference.csv"
    fakes_file = "fakes.csv"
    
    # 1. טעינת נתונים
    df = load_and_prepare_data(real_file, fakes_file)
    features = extract_features(df)
    adj = build_adjacency_matrix(df)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = features.to(device)
    adj = adj.to(device)

    # 2. הרצת GNN (LLGC Logic) להפקת Embeddings מבניים
    print("--- Computing Hyperbolic Embeddings (LLGC) ---")
    # שימוש ב-PageRankAgg מה-repo לצבירת מידע מהשכנים
    aggregator = PageRankAgg(K=10, alpha=0.1).to(device)
    x_agg, _ = aggregator(features, adj._indices())
    
    # שימוש במודל ה-LLGC להטמעה במרחב Lorentzian
    # נגדיר ממד פלט של 64 עבור ה-embeddings
    model = LLGC(nfeat=features.shape[1], nclass=64, drop_out=0.0, use_bias=1).to(device)
    with torch.no_grad():
        embeddings = model(x_agg).cpu().numpy()

    # 3. זיהוי אנומליות עם Isolation Forest
    print("--- Running Isolation Forest ---")
    # contamination מוגדר על פי יחס הזיופים בדאטה
    contamination_rate = len(df[df['is_anomaly'] == 1]) / len(df)
    clf = IsolationForest(contamination=contamination_rate, random_state=42)
    
    # חיזוי (1- תקין, -1 אנומליה)
    preds = clf.fit_predict(embeddings)
    df['prediction'] = [1 if p == -1 else 0 for p in preds] # המרת -1 ל-1 (אנומליה)

    # 4. ולידציה על ה-Fakes שהזרקנו
    y_true = df['is_anomaly']
    y_pred = df['prediction']
    
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*30)
    print(f"Results for Anomaly Detection:")
    print(f"Recall (Detection Rate of Fakes): {recall:.4%}")
    print(f"Precision: {precision:.4%}")
    print(f"F1 Score: {f1:.4%}")
    print("="*30)

    if recall >= 0.85:
        print("Success! Reached target accuracy of 85%+")
    else:
        print("Target not reached. Consider increasing TF-IDF features or GNN steps (K).")

if __name__ == "__main__":
    run_anomaly_detection()