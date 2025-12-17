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

# ייבוא הרכיבים המקוריים מה-Repo
from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor
from manifolds.lorentzian import Lorentzian

# הגדרות
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--max_epoch', type=int, default=120)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.2) # איזון לטובת המבנה
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifold = Lorentzian()

def load_and_prepare():
    print("Loading datasets and calculating structural features...")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    df_fakes = pd.read_csv("fakes.csv")
    
    df_real['is_anomaly'] = 0
    df_fakes['is_anomaly'] = 1
    df = pd.concat([df_real, df_fakes], ignore_index=True)
    df['id'] = df['id'].astype(str)
    
    # 1. בניית גרף וחישוב פיצ'רים מבניים (Degree)
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
    
    # דרגת צומת (ככל שנמוך יותר - חשוד יותר כצומת מבודד)
    df['degree'] = df['id'].map(dict(G.degree())).fillna(0)
    
    # 2. הנדסת פיצ'רים טקסטואליים
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    x_text = tfidf.fit_transform(df['title'].fillna('') + " " + df['abstract'].fillna('')).toarray()
    
    # 3. הכנת אינדקסים לאימון
    le = LabelEncoder()
    df_real['label_idx'] = le.fit_transform(df_real['fos.name'].fillna('Unknown'))
    num_classes = len(le.classes_)
    
    scaler = StandardScaler()
    x_final = scaler.fit_transform(np.hstack([x_text, df[['degree']].values]))
    features = torch.FloatTensor(x_final).to(device)
    
    # הכנת מטריצת שכנות
    adj = nx.adjacency_matrix(G, nodelist=node_ids)
    adj, _ = preprocess_citation(adj, x_final, normalization="AugNormAdj")
    edge_index = sparse_mx_to_torch_sparse_tensor(adj)._indices().to(device)
    
    idx_train = torch.LongTensor(range(len(df_real))).to(device)
    labels = torch.LongTensor(df_real['label_idx'].values).to(device)
    
    return features, edge_index, df, idx_train, labels, num_classes

def main():
    features, edge_index, df, idx_train, labels, num_classes = load_and_prepare()
    
    # אתחול המודל בממד 256
    model = LLGC(nfeat=features.size(1), nclass=args.embedding_dim, drop_out=0.1, use_bias=True).to(device)
    cls_head = torch.nn.Linear(args.embedding_dim, num_classes).to(device)
    aggregator = PageRankAgg(K=10, alpha=args.alpha).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(cls_head.parameters()), lr=args.lr)

    # שלב ה-TRAIN (למידת המרחב הנורמלי)
    print(f"--- Training 256D Model on {num_classes} categories ---")
    model.train()
    for epoch in range(args.max_epoch):
        optimizer.zero_grad()
        x_smooth, _ = aggregator(features, edge_index)
        embeddings = model(x_smooth)
        logits = cls_head(embeddings[idx_train])
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if epoch % 30 == 0: print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # שלב ה-EVAL (הפקת ייצוגים סופיים)
    model.eval()
    with torch.no_grad():
        x_smooth_eval, _ = aggregator(features, edge_index)
        # att_out - הייצוגים הסופיים במרחב לורנץ
        att_out = model(x_smooth_eval) 
        
        # חישוב "מרכזי הקטגוריות" (Prototypes) במרחב לורנץ
        # לכל קטגוריה, נמצא את הממוצע של המאמרים האמיתיים שלה
        prototypes = []
        for i in range(num_classes):
            p = att_out[idx_train][labels == i].mean(dim=0, keepdim=True)
            prototypes.append(p)
        prototypes = torch.cat(prototypes, dim=0) # [num_classes, 256]

        # לכל מאמר בגרף (כולל המזויפים), נבדוק את המרחק לקטגוריה הכי קרובה שלו
        # מאמר אנומלי יהיה רחוק מהמרכז של כל קטגוריה הגיונית
        logits_all = cls_head(att_out)
        pred_labels = logits_all.argmax(dim=1)
        
        # חישוב מרחק לורנציאני לשיוך המנובא
        assigned_protos = prototypes[pred_labels]
        # שימוש במרחק לורנץ מהמחלקה
        dist_to_proto = manifold.induced_distance(att_out, assigned_protos, model.c)
        dist_to_proto = dist_to_proto.cpu().numpy().flatten()

    # שלב ה-DETECTION: שילוב של Isolation Forest עם המרחק הקהילתי
    print("Running detection on learned 256D embeddings...")
    final_features = np.hstack([att_out.cpu().numpy(), dist_to_proto.reshape(-1, 1)])
    
    clf = IsolationForest(contamination='auto', random_state=args.seed)
    clf.fit(final_features)
    df['anomaly_score'] = -clf.decision_function(final_features)
    
    # פלט תוצאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    output_csv = "final_256d_realistic_results.csv"
    df_sorted[['id', 'title', 'fos.name', 'is_anomaly', 'anomaly_score']].to_csv(output_csv, index=False)
    
    num_fakes = df['is_anomaly'].sum()
    detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Deep Realistic Results (256D + Community Mismatch):")
    print(f"Detected {detected} / {num_fakes} fakes in Top {num_fakes}")
    print(f"Precision: {detected / num_fakes:.4f}")
    
    print("\nTop 10 Anomaly Candidates:")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        src = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {src} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:40]}...")
    print("="*50)

if __name__ == "__main__":
    main()