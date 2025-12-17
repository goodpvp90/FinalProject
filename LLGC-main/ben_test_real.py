import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from model import LLGC, PageRankAgg
from utils import preprocess_citation, sparse_mx_to_torch_sparse_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def prepare_and_train_model():
    print("--- שלב 1: אימון מודל עמוק (Embedding Dimension = 128) ---")
    df_real = pd.read_csv("final_filtered_by_fos_and_reference.csv")
    
    le = LabelEncoder()
    df_real['label'] = le.fit_transform(df_real['fos.name'].fillna('Unknown'))
    num_classes = df_real['label'].nunique()
    
    # ממד Embedding גבוה יותר נותן יותר "אוויר" לאנומליות לבלוט
    EMB_DIM = 128 
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    x_tfidf = vectorizer.fit_transform(df_real['title'].fillna('') + " " + df_real['abstract'].fillna('')).toarray()
    features_real = torch.FloatTensor(x_tfidf).to(device)
    labels_real = torch.LongTensor(df_real['label'].values).to(device)
    
    node_ids = df_real['id'].astype(str).tolist()
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    G_real = nx.Graph()
    G_real.add_nodes_from(node_ids)
    for _, row in df_real.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx: G_real.add_edge(str(row['id']), str(ref))
        except: continue
        
    adj_real = nx.adjacency_matrix(G_real, nodelist=node_ids)
    adj_real, _ = preprocess_citation(adj_real, x_tfidf, normalization="AugNormAdj")
    adj_real_tensor = sparse_mx_to_torch_sparse_tensor(adj_real).to(device)

    aggregator = PageRankAgg(K=10, alpha=0.15).to(device)
    
    # מודל ה-LLGC יוציא 128 ממדים
    model = LLGC(nfeat=500, nclass=EMB_DIM, drop_out=0.1, use_bias=True).to(device)
    # ראש סיווג חיצוני לצורך האימון בלבד
    classifier_head = torch.nn.Linear(EMB_DIM, num_classes).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(classifier_head.parameters()), lr=0.01)
    
    model.train()
    x_smooth, _ = aggregator(features_real, adj_real_tensor._indices())
    
    print("מתחיל אימון אינטנסיבי (200 Epochs)...")
    for epoch in range(201):
        optimizer.zero_grad()
        embeddings = model(x_smooth)
        logits = classifier_head(embeddings)
        loss = F.cross_entropy(logits, labels_real)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    return model, vectorizer, df_real

def inject_and_detect(trained_model, vectorizer, df_real):
    print("\n--- שלב 2: זיהוי אנומליות במרחב ה-128D ---")
    df_fakes = pd.read_csv("fakes.csv")
    df_fakes['is_anomaly'] = 1
    df_real['is_anomaly'] = 0
    df_combined = pd.concat([df_real, df_fakes], ignore_index=True)
    
    x_combined_tfidf = vectorizer.transform(df_combined['title'].fillna('') + " " + df_combined['abstract'].fillna('')).toarray()
    for i, row in df_combined.iterrows():
        if row['is_anomaly'] == 1:
            x_combined_tfidf[i] = np.random.uniform(-1, 1, 500) # רעש אקראי
            
    features_combined = torch.FloatTensor(x_combined_tfidf).to(device)
    
    # בניית גרף מאוחד
    node_ids = df_combined['id'].astype(str).tolist()
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    G_combined = nx.Graph()
    G_combined.add_nodes_from(node_ids)
    for _, row in df_combined.iterrows():
        try:
            refs = ast.literal_eval(row['references'])
            for ref in refs:
                if str(ref) in node_to_idx: G_combined.add_edge(str(row['id']), str(ref))
        except: continue
        
    adj_combined = nx.adjacency_matrix(G_combined, nodelist=node_ids)
    adj_combined, _ = preprocess_citation(adj_combined, x_combined_tfidf, normalization="AugNormAdj")
    adj_combined_tensor = sparse_mx_to_torch_sparse_tensor(adj_combined).to(device)

    # זיהוי עם Alpha גבוה (0.8)
    aggregator_detect = PageRankAgg(K=10, alpha=0.8).to(device)
    trained_model.eval()
    with torch.no_grad():
        x_smooth_comb, _ = aggregator_detect(features_combined, adj_combined_tensor._indices())
        embeddings = trained_model(x_smooth_comb).cpu().numpy()
        
    clf = IsolationForest(contamination='auto', random_state=SEED)
    clf.fit(embeddings)
    df_combined['anomaly_score'] = -clf.decision_function(embeddings)
    
    df_sorted = df_combined.sort_values(by='anomaly_score', ascending=False)
    num_fakes = len(df_fakes)
    detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print(f"\nFinal Results: Precision@{num_fakes}: {detected / num_fakes:.4f} (Detected {detected}/{num_fakes})")
    return df_sorted

# הרצה
model, vec, df_real = prepare_and_train_model()
results = inject_and_detect(model, vec, df_real)