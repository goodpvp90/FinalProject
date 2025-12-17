import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים מהקבצים הקיימים ב-repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
from manifolds.lorentzian import Lorentzian

# הגדרת פרמטרים
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=256, help='Requested 256D embedding')
parser.add_argument('--K', type=int, default=10, help='Propagation steps')
parser.add_argument('--alpha', type=float, default=0.15, help='Smoothing factor')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_data():
    print("Loading data and calculating deep features...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. בניית גרף וחישוב פיצ'ר מבני (Degree)
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
    
    # חישוב לוג-דרגה (צמתים מבודדים יקבלו ערך נמוך מאוד)
    degrees = dict(G.degree())
    df['log_degree'] = df['id'].map(degrees).apply(lambda x: np.log1p(x))

    # 2. הנדסת תכונות (Features)
    # א. טקסט (TF-IDF)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    x_text = tfidf.fit_transform(df['title'].fillna('') + " " + df['abstract'].fillna('')).toarray()
    
    # ב. FOS (One-Hot) - זיהוי קטגוריות זרות
    fos_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_fos = fos_enc.fit_transform(df[['fos.name']].fillna('Unknown'))
    
    # ג. שילוב ופיצ'ר מבני
    x_struct = df[['log_degree']].values
    x_combined = np.hstack([x_text, x_fos, x_struct])
    
    # נרמול סטנדרטי למניעת "ניפוח" מלאכותי
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x_combined)
    features = torch.FloatTensor(x_normalized).to(device)
    
    # הכנת מטריצת שכנות למודל
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, x_normalized, normalization="AugNormAdj")
    edge_index = sparse_mx_to_torch_sparse_tensor(adj)._indices().to(device)
    
    return features, edge_index, df

def main():
    # שלב 1: טעינה ועיבוד
    features, edge_index, df = load_and_prepare_data()
    
    # שלב 2: הפקת ייצוגים במרחב לורנץ
    # א. PageRank Aggregation - יצירת הקונטקסט של השכנים
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, edge_index)
    
    # ב. אתחול מודל LLGC בממד 256
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, drop_out=0.0, use_bias=True).to(device)
    model.eval()
    manifold = Lorentzian()
    
    with torch.no_grad():
        # המרת הפיצ'רים המקוריים והמוחלקים למרחב לורנץ (באותו ממד 256)
        # אנחנו מעבירים את שניהם דרך שכבת ה-W כדי שיהיו ניתנים להשוואה
        z_orig = model.W(manifold.normalize_input(features, model.c))
        z_smooth = model.W(manifold.normalize_input(x_smooth, model.c))
        
        # חישוב מרחק המיסמאץ' (Induced Distance) במרחב לורנץ
        # זה מודד כמה המאמר "לא מתאים" לסביבה שלו בגרף
        mismatch_dist = manifold.induced_distance(z_orig, z_smooth, model.c).cpu().numpy().flatten()
        
        # הפקת ה-embeddings הסופיים (במישור המשיק) לצורך Isolation Forest
        embeddings_tan = manifold.log_map_zero(z_smooth, model.c).cpu().numpy()

    # שלב 3: זיהוי אנומליות
    print("Running Isolation Forest on 256D embeddings...")
    # נשלב את ה-embeddings עם המרחק הלורנציאני כסיגנל נוסף
    detection_space = np.hstack([embeddings_tan, mismatch_dist.reshape(-1, 1)])
    
    clf = IsolationForest(contamination='auto', random_state=args.seed, n_jobs=-1)
    clf.fit(detection_space)
    
    # ציון אנומליה (גבוה יותר = אנומלי יותר)
    df['anomaly_score'] = -clf.decision_function(detection_space)
    
    # שלב 4: שמירה וסיכום תוצאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    # שמירה לקובץ CSV של כולם
    output_file = "all_anomaly_scores_deep_logic.csv"
    columns = ['id', 'title', 'fos.name', 'log_degree', 'is_anomaly', 'anomaly_score']
    df_sorted[columns].to_csv(output_file, index=False)
    
    num_fakes = df['is_anomaly'].sum()
    top_candidates = df_sorted.head(num_fakes)
    detected = top_candidates['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Results Summary (Realistic Mode - 256D):")
    print(f"Detected {detected} / {num_fakes} fakes in Top {num_fakes}.")
    print(f"Precision: {detected / num_fakes:.4f}")
    print(f"Full results saved to: {output_file}")
    
    print("\nTop 10 Detected Anomalies:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:40]}...")
    print("="*50)

if __name__ == "__main__":
    main()