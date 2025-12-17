import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model import LLGC, PageRankAgg
from manifolds.lorentzian import Lorentzian
import ast
import networkx as nx

# ... (prepare_data_and_labels נשאר זהה לקוד הקודם שלך) ...

def main():
    features, edge_index, df, idx_train, labels, num_classes = prepare_data_and_labels()
    
    # אתחול המודל (256D)
    model = LLGC(nfeat=features.size(1), nclass=256, drop_out=0.1, use_bias=True).to(device)
    cls_head = torch.nn.Linear(256, num_classes).to(device)
    aggregator = PageRankAgg(K=10, alpha=0.15).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(cls_head.parameters()), lr=0.01)
    manifold = Lorentzian()

    # שלב 1: אימון מהיר כדי לייצב את המרחב
    print("--- Training 256D Base Space ---")
    model.train()
    for epoch in range(101):
        optimizer.zero_grad()
        # אימון על הזרם המוחלק (כדי ללמוד קהילות)
        x_smooth, _ = aggregator(features, edge_index)
        embeddings = model(x_smooth)
        logits = cls_head(embeddings[idx_train])
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # שלב 2: זיהוי לפי Residual Distance (מרחק לורנציאני בין המקור להקשר)
    print("\n--- Calculating Lorentzian Residuals (The Deep Way) ---")
    model.eval()
    with torch.no_grad():
        # א. ייצוג מקומי (רק התוכן של המאמר)
        z_local = model(features) 
        
        # ב. ייצוג הקשרי (מה שהשכנים אומרים)
        x_smooth, _ = aggregator(features, edge_index)
        z_context = model(x_smooth)
        
        # ג. חישוב המרחק הלורנציאני בין השניים
        # במאמר תקין, התוכן תואם את השכנים (מרחק קטן)
        # באנומליה (כמו Cooking בתוך Computer Science), יש סתירה (מרחק גדול)
        residual_dist = manifold.induced_distance(z_local, z_context, model.c)
        residual_dist = residual_dist.cpu().numpy().flatten()

    # שלב 3: שקלול עם ציון מבני (Degree)
    # צמתים מבודדים הם אנומליות מבניות
    degrees = df['degree'].values
    # נרמול המרחק והדרגה
    norm_dist = (residual_dist - residual_dist.min()) / (residual_dist.max() - residual_dist.min())
    # ציון אנומליה: מרחק שארית גבוה + דרגה נמוכה
    df['anomaly_score'] = norm_dist + (1.0 / (degrees + 1.0))
    
    # תוצאות
    df_sorted = df.sort_values(by='anomaly_score', ascending=False)
    num_fakes = df['is_anomaly'].sum()
    detected = df_sorted.head(num_fakes)['is_anomaly'].sum()
    
    print("\n" + "="*50)
    print(f"Lorentzian Residual Detection Results:")
    print(f"Detected {detected} / {num_fakes} fakes in Top {num_fakes}")
    print(f"Precision: {detected / num_fakes:.4f}")
    
    print("\nTop 10 Lorentzian Anomalies (Structural + Content Mismatch):")
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows()):
        label = "[FAKE]" if row['is_anomaly'] == 1 else "[REAL]"
        print(f"#{i+1} {label} Score: {row['anomaly_score']:.4f} | FOS: {row['fos.name']} | Title: {row['title'][:45]}...")
    print("="*50)
    
    df_sorted.to_csv("residual_anomaly_results.csv", index=False)

if __name__ == "__main__":
    main()