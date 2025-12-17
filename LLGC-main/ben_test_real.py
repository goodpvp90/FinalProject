import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

# ייבוא הרכיבים מה-Repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
from manifolds.lorentzian import Lorentzian

# הגדרות
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.15)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data_and_labels():
    print("Loading datasets...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. קידוד לייבלים לאימון (רק עבור המאמרים האמיתיים)
    le = LabelEncoder()
    # אנחנו מאמנים רק על הקטגוריות הקיימות בדאטאסט האמיתי
    df_real['label_idx'] = le.fit_transform(df_real['fos.name'].fillna('Unknown'))
    num_classes = len(le.classes_)
    
    # 2. הנדסת פיצ'רים (Text + Structure)
    print("Vectorizing text and building structural features...")
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    x_text = tfidf.fit_transform(df['title'].fillna('') + " " + df['abstract'].fillna('')).toarray()
    
    # פיצ'ר מבני: Degree
    node_ids = df['id'].tolist()
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for _, row in df.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx: G.add_edge(row['id'], str(ref))
        except: continue
    
    degrees = np.array([G.degree(nid) for nid in node_ids]).reshape(-1, 1)
    x_combined = np.hstack([x_text, np.log1p(degrees)])
    
    scaler = StandardScaler()
    x_final = scaler.fit_transform(x_combined)
    features = torch.FloatTensor(x_final).to(device)
    
    # הכנת מטריצת שכנות
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, x_final, normalization="AugNormAdj")
    edge_index = sparse_mx_to_torch_sparse_tensor(adj)._indices().to(device)
    
    # אינדקסים לאימון (רק המאמרים האמיתיים)
    idx_train = torch.LongTensor(range(len(df_real))).to(device)
    labels = torch.LongTensor(df_real['label_idx'].values).to(device)
    
    return features, edge_index, df, idx_train, labels, num_classes

def main():
    # שלב 1: הכנה
    features, edge_index, df, idx_train, labels, num_classes = prepare_data_and_labels()
    
    # שלב 2: הגדרת מודלים (LLGC + Classification Head)
    # המודל שלך יפיק ייצוג ב-256D
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, drop_out=0.1, use_bias=True).to(device)
    # ראש סיווג לצורך האימון
    cls_head = torch.nn.Linear(args.embedding_dim, num_classes).to(device)
    
    aggregator = PageRankAgg(K=10, alpha=args.alpha).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(cls_head.parameters()), lr=args.lr)

    # שלב 3: TRAINING (כמו בקוד של הידידה)
    print(f"--- Starting Training on {num_classes} categories ---")
    model.train()
    cls_head.train()
    
    for epoch in range(args.max_epoch):
        optimizer.zero_grad()
        
        # Forward pass: Smoothing -> Lorentzian Embedding
        x_smooth, _ = aggregator(features, edge_index)
        embeddings = model(x_smooth) # [N, 256]
        
        # סיווג רק של צמתי האימון
        logits = cls_head(embeddings[idx_train])
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f}")

    # שלב 4: FINAL REPRESENTATIONS (הפקת ייצוגים סופיים)
    print("\n--- Training Done. Extracting Final 256D Representations ---")
    model.eval()
    cls_head.eval()
    
    with torch.no_grad():
        x_smooth_eval, _ = aggregator(features, edge_index)
        # אלו הייצוגים הסופיים ("att_out" במושגים של הידידה)
        final_embeddings = model(x_smooth_eval).cpu().numpy()

    # שלב 5: DETECTION (זיהוי אנומליות על הייצוגים המלומדים)
    print("Running Isolation Forest on learned embeddings...")
    clf = IsolationForest(contamination='auto', random_state=args.seed, n_jobs=-1)
    clf.fit(final_embeddings)
    
    # ציון אנומליה (גבוה יותר = חשוד יותר)
    df['anomaly_score'] = -clf.decision_function(final_embeddings)
    
    # שלב 6: פלט וסיכום
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    
    output_file = "trained_256d_anomaly_results.csv"
    df_sorted[['id', 'title', 'fos.name', 'is_anomaly', 'anomaly_score']].to_csv(output_file, index=False)
    
    num_fakes = df['is_anomaly'].sum()
    top_candidates = df_sorted.head(num_fakes)
    detected = top_candidates['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Final Results (Trained 256D Model):")
    print(f"Precision@{num_fakes}: {detected / num_fakes:.4f} ({detected}/{num_fakes})")
    print(f"Full scores saved to: {output_file}")
    
    print("\nTop 10 Detected Anomalies:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:40]}...")
    print("="*50)

if __name__ == "__main__":
    main()