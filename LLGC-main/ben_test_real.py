import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים המקוריים
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
from manifolds.lorentzian import Lorentzian

# הגדרת פרמטרים
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=256, help='Requested Embedding Dimension')
parser.add_argument('--K', type=int, default=10, help='Propagation steps')
parser.add_argument('--alpha', type=float, default=0.15, help='Smoothing factor')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_features():
    print("Loading datasets...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. בניית הגרף לחישוב פיצ'רים מבניים
    print("Building graph and calculating structural features...")
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
    
    # פיצ'ר מבני: דרגת הצומת (Log scale כדי לנרמל)
    degrees = dict(G.degree())
    df['degree'] = df['id'].map(degrees).fillna(0)
    df['log_degree'] = np.log1p(df['degree'])

    # 2. וקטוריזציה של טקסט (TF-IDF)
    print("Vectorizing titles and abstracts...")
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    x_text = tfidf.fit_transform(df['title'].fillna('') + " " + df['abstract'].fillna('')).toarray()
    
    # 3. קידוד FOS (One-Hot)
    print("Encoding FOS names...")
    # נשתמש ב-handle_unknown כי ב-fakes יש FOS חדשים
    fos_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_fos = fos_enc.fit_transform(df[['fos.name']].fillna('Unknown'))
    
    # 4. איחוד כל הפיצ'רים (טקסט + FOS + מבנה)
    # שימוש ב-StandardScaler כדי שכולם יהיו באותו קנה מידה ללא ניפוח מלאכותי
    x_structural = df[['log_degree']].values
    x_combined = np.hstack([x_text, x_fos, x_structural])
    
    scaler = StandardScaler()
    x_final = scaler.fit_transform(x_combined)
    features = torch.FloatTensor(x_final).to(device)
    
    # הכנת מטריצת שכנות למודל
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, x_final, normalization="AugNormAdj")
    edge_index = sparse_mx_to_torch_sparse_tensor(adj)._indices().to(device)
    
    return features, edge_index, df

def main():
    # שלב 1: הכנת נתונים
    features, edge_index, df = load_and_prepare_features()
    
    # שלב 2: הפקת Embeddings לורנציאניים
    # נשתמש ב-PageRankAgg כדי לערבב את התכונות עם השכנים
    aggregator = PageRankAgg(K=args.K, alpha=args.alpha).to(device)
    x_smooth, _ = aggregator(features, edge_index)
    
    # הטלה למרחב לורנץ בממד 256
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, 
                 drop_out=0.0, use_bias=True).to(device)
    model.eval()
    
    manifold = Lorentzian()
    with torch.no_grad():
        # המרת הפלט לייצוג במרחב לורנץ
        embeddings_loren = model(x_smooth)
        
        # חישוב "מרחק המיסמאץ'" (Mismatch Distance)
        # כמה המאמר שונה מהממוצע של שכניו במרחב ההיפרבולי
        x_orig_loren = manifold.normalize_input(features, torch.tensor([1.0]).to(device))
        dist_to_context = manifold.induced_distance(x_orig_loren, embeddings_loren, torch.tensor([1.0]).to(device))
        
        # המרה ל-Numpy לצורך Isolation Forest
        embeddings_np = embeddings_loren.cpu().numpy()
        dists_np = dist_to_context.cpu().numpy().reshape(-1, 1)

    # שלב 3: זיהוי אנומליות (Isolation Forest)
    print(f"Running Isolation Forest on {args.embedding_dim}D embeddings + Mismatch distance...")
    # נשלב את ה-embeddings עם מרחק המיסמאץ' כפיצ'ר נוסף
    detection_features = np.hstack([embeddings_np, dists_np])
    
    clf = IsolationForest(contamination='auto', random_state=args.seed, n_jobs=-1)
    clf.fit(detection_features)
    
    # ציון אנומליה (גבוה יותר = אנומלי יותר)
    df['anomaly_score'] = -clf.decision_function(detection_features)
    
    # שלב 4: סיכום ופלט
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    # שמירה לקובץ CSV
    output_file = "final_anomaly_scores_all.csv"
    columns_to_save = ['id', 'title', 'fos.name', 'degree', 'is_anomaly', 'anomaly_score']
    df_sorted[columns_to_save].to_csv(output_file, index=False)
    
    num_fakes = df['is_anomaly'].sum()
    top_candidates = df_sorted.head(num_fakes)
    detected = top_candidates['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Deep Validation Results (No Artificial Noise):")
    print(f"Embedding Dim: {args.embedding_dim}")
    print(f"Detected {detected} / {num_fakes} fakes in Top {num_fakes}")
    print(f"Precision@{num_fakes}: {detected / num_fakes:.4f}")
    print(f"Full results saved to: {output_file}")
    
    print("\nTop 10 Detected Anomalies:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:40]}...")
    print("="*50)

if __name__ == "__main__":
    main()