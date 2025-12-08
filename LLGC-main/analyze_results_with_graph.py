import pandas as pd
import matplotlib.pyplot as plt

# load results
df = pd.read_csv("C:\\Users\\nadir\\FinalProject\\temporal_anomaly_results_real_only.csv")

df.head()
# Plot 1: anomaly score distribution
plt.figure(figsize=(8,5))
plt.hist(df['anomaly_score'], bins=50)
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.title("Distribution of Anomaly Scores")
plt.show()
# Plot 2: anomaly scores per segment (mean)
segment_means = df.groupby(["t_start", "t_end"])["anomaly_score"].mean().reset_index()

plt.figure(figsize=(10,5))
plt.plot(segment_means["t_start"], segment_means["anomaly_score"], marker="o")
plt.xlabel("Segment Start Year")
plt.ylabel("Mean Anomaly Score")
plt.title("Mean Anomaly Score Over Time Segments")
plt.show()
# Plot 3: Top 20 most anomalous papers (lowest anomaly scores)
top20 = df.nsmallest(20, "anomaly_score")

plt.figure(figsize=(10,6))
plt.barh(top20["paper_id"].astype(str), top20["anomaly_score"])
plt.xlabel("Anomaly Score")
plt.ylabel("Paper ID")
plt.title("Top 20 Most Anomalous Papers")
plt.gca().invert_yaxis()
plt.show()

# Group by time window
grouped = df.groupby(["t_start", "t_end"])

results = []

for (start, end), group in grouped:
    total = len(group)
    anomalies = (group["prediction"] == -1).sum()
    pct = (anomalies / total) * 100
    results.append({
        "period": f"{start}-{end}",
        "percentage_anomalies": pct
    })

plot_df = pd.DataFrame(results)

# Sort chronologically
plot_df["start"] = plot_df["period"].apply(lambda x: int(x.split("-")[0]))
plot_df = plot_df.sort_values("start")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(plot_df["period"], plot_df["percentage_anomalies"], marker="o", linewidth=2)
plt.xticks(rotation=45)
plt.ylabel("Percentage of Anomalous Papers (%)")
plt.title("Anomalous Paper Rate per Time Period")
plt.grid(True)
plt.tight_layout()
plt.show()

# Keep only real nodes (just in case)
df = df[df["is_synthetic"] == False]

# Convert to numeric (your years look numeric already, but this is safe)
df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")

# Identify the last time segment
latest_period = df["t_end"].max()
print("Latest time segment:", latest_period)

# Extract only rows from that last segment
df_last = df[df["t_end"] == latest_period]

print(f"\nTotal papers in last period: {len(df_last)}")

# Filter TRUE anomalies (prediction == -1)
df_last_anomalies = df_last[df_last["prediction"] == -1]

print(f"\n🔥 TRUE ANOMALIES in final time period: {len(df_last_anomalies)}")

# Sort by anomaly score: lower = more anomalous
df_last_anomalies = df_last_anomalies.sort_values(by="anomaly_score")

print("\nMost anomalous papers in the last time segment:\n")
print(df_last_anomalies[["paper_id", "anomaly_score", "prediction", "t_start", "t_end"]])