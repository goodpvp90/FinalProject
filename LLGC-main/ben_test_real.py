import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, precision_score, f1_score

# ייבוא רכיבים מתוך ה-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

def safe_parse_list(val):
    """
    מחלץ IDs מתוך מחרוזת של רשימה בצורה בטוחה, גם אם חסרות מרכאות.
    """
    if pd.isna(val) or val == "" or val == "[]":
        return []
    if isinstance(val, list):
        return val
    # מוצא את כל הרצפים האלפא-נומריים בתוך הסוגריים
    return re.findall(r'[a-zA-Z0-9]+', str(val))

def load_and_prepare_data(real_path, fakes_path):
    print(f"--- Loading datasets: {real_path} and {fakes_path} ---")
    df_real = pd.read_csv(real_path)
    df_fakes = pd.read_csv(fakes_path)
    
    # סימון לייבלים לולידציה
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # שימוש בפונקציה הבטוחה החדשה במקום ast.literal_eval
    df['references'] = df['references'].apply(safe_parse_list)
    
    # הוספת פיצ'ר מבני בסיסי: כמות רפרנסים (Out-degree)
    df['num_refs'] = df['references'].apply(len)
    
    return df

def extract_features(df):
    print("--- Extracting features (Deep Semantic Analysis) ---")
    # 1. טקסט: שילוב כותרת ותקציר עם דגש על מילים ייחודיות
    df['text_combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1500, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['text_combined']).toarray()
    
    # 2. נומרי: שנה, כמות ציטוטים וכמות רפרנסים (מנורמל)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[['year', 'n_citation', 'num_refs']].fillna(0))
    
    # 3. קטגורי: Field of Study (FoS)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_fos = encoder.fit_transform(df[['fos.name']].fillna('Unknown'))
    
    X_all = np.hstack((X_text, X_num, X_fos))
    return torch.FloatTensor(X_all)

def build_adjacency_matrix(df):
    print("--- Building graph structure ---")
    node_to_idx = {node_id: i for i, node_id in enumerate(df['id'])}
    edges = []
    
    for i, row in df.iterrows():
        source_idx = node_to_idx[row['id']]
        for target_id in row['references']:
            if target_id in node_to_idx:
                edges.append((source_idx, node_to_idx[target_id]))
    
    adj = nx.adjacency_matrix(nx.Graph(edges), nodelist=range(len(df)))
    adj_normalized, _ = preprocess_citation(adj, np.zeros((len(df), 1)))
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def run_anomaly_detection():
    real_file = "final_filtered_by_fos_and_reference.csv"
    fakes_file = "fakes.csv"
    
    # 1. הכנת נתונים
    df = load_and_prepare_data(real_file, fakes_file)
    features = extract_features(df)
    adj = build_adjacency_matrix(df)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = features.to(device)
    adj = adj.to(device)

    # 2. הפקת Embeddings היפרבוליים בעזרת המודל מה-Repo
    print("--- Running Lorentzian GNN (LLGC) ---")
    # K=15 מאפשר פעפוע עמוק יותר של המבנה לתוך ה-Embeddings
    aggregator = PageRankAgg(K=15, alpha=0.15).to(device)
    x_agg, _ = aggregator(features, adj._indices())
    
    model = LLGC(nfeat=features.shape[1], nclass=128, drop_out=0.1, use_bias=1).to(device)
    model.eval()
    with torch.no_grad():
        # ההטמעה במרחב Lorentzian מבליטה חריגות מבניות
        embeddings = model(x_agg).cpu().numpy()

    # 3. זיהוי אנומליות עם Isolation Forest
    print("--- Running Isolation Forest Analysis ---")
    contamination_rate = len(df[df['is_anomaly'] == 1]) / len(df)
    
    # שילוב ה-embeddings יחד עם הפיצ'רים הגולמיים לזיהוי מקסימלי
    combined_input = np.hstack((embeddings, features.cpu().numpy()))
    
    clf = IsolationForest(n_estimators=200, contamination=contamination_rate, random_state=42, n_jobs=-1)
    preds = clf.fit_predict(combined_input)
    df['prediction'] = [1 if p == -1 else 0 for p in preds]

    # 4. ולידציה
    y_true = df['is_anomaly']
    y_pred = df['prediction']
    
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    
    print("\n" + "═"*40)
    print(f"Final Anomaly Detection Results:")
    print(f"Recall (Detection Rate): {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.2%}")
    print("═"*40)

    if recall >= 0.85:
        print("🎯 Goal Achieved! Successfully detected over 85% of fake papers.")
    else:
        print("💡 Tip: Try increasing Tfidf max_features or GNN K parameter.")

if __name__ == "__main__":
    run_anomaly_detection()