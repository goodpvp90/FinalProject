import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Load your results file
df = pd.read_csv("temporal_anomaly_results_real_only.csv")

# Only real papers
df = df[df["is_synthetic"] == False]

# Sort for consistency
df = df.sort_values(["paper_id", "t_end"])

# Dictionary: paper_id → {t_end: anomaly_score}
paper_time_scores = defaultdict(dict)

for _, row in df.iterrows():
    pid = row["paper_id"]
    t = row["t_end"]
    score = row["anomaly_score"]
    paper_time_scores[pid][t] = score

# Time windows available
all_time_stamps = sorted(df["t_end"].unique())

paper_vectors = {}
avg_scores = {}

for pid, time_dict in paper_time_scores.items():
    vector = []
    valid_scores = []

    for t in all_time_stamps:
        if t in time_dict:
            score = time_dict[t]
            vector.append(score)
            valid_scores.append(score)
        else:
            vector.append(None)

    paper_vectors[pid] = vector

    # Average anomaly score for this paper
    avg_scores[pid] = np.mean(valid_scores) if valid_scores else None

# Convert to DataFrame
vectors_df = pd.DataFrame.from_dict(
    paper_vectors,
    orient="index",
    columns=[f"score_{t}" for t in all_time_stamps]
)

# Add average anomaly score column
vectors_df["avg_anomaly_score"] = vectors_df.mean(axis=1, skipna=True)

vectors_df.index.name = "paper_id"

# Save to CSV
vectors_df.to_csv("anomaly_vectors_with_avg.csv")

print("\n🔥 Created vectors + average anomaly scores for each paper!")
print(vectors_df.head())


# === Histogram of average anomaly scores ===

# Drop papers that have no average (rare)
avg_scores_clean = vectors_df["avg_anomaly_score"].dropna()

plt.figure(figsize=(10, 6))
plt.hist(avg_scores_clean, bins=40)
plt.title("Distribution of Average Anomaly Scores Across Papers")
plt.xlabel("Average Anomaly Score")
plt.ylabel("Number of Papers")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




# ---------- PARAMETERS ----------
Z_THRESH = 2.0          # z-score threshold for spike
DELTA_MULT = 2.0        # delta threshold in units of delta_std*DELTA_MULT
MIN_OBS = 3             # minimum number of timestamps for a node to be considered
SAVE_CSV = True
# --------------------------------

# load the CSV you created previously
vectors_df = pd.read_csv("anomaly_vectors_with_avg.csv", index_col="paper_id")

# columns are like score_1964, score_1969, ... or similar
time_cols = [c for c in vectors_df.columns if c.startswith("score_")]
time_stamps = [int(c.replace("score_","")) for c in time_cols]

# results accumulation
spike_records = []

for pid, row in vectors_df.iterrows():
    scores = row[time_cols].values.astype(float)  # NaN where missing
    # require some observations
    valid_mask = ~np.isnan(scores)
    n_obs = np.sum(valid_mask)
    if n_obs < MIN_OBS:
        continue

    # compute per-node mean/std using its observed values
    obs_vals = scores[valid_mask]
    mean = np.mean(obs_vals)
    std = np.std(obs_vals, ddof=0)
    if std == 0:
        std = 1e-6  # avoid div by zero

    # compute z-scores for observed positions
    z_scores = (scores - mean) / std

    # compute deltas between consecutive observed timestamps (use forward diff)
    # we'll compute delta only where both t and t-1 exist
    deltas = np.full_like(scores, np.nan)
    prev_val = np.nan
    prev_idx = None
    for i, v in enumerate(scores):
        if np.isnan(v):
            prev_val = np.nan
            prev_idx = None
            continue
        if prev_idx is not None:
            deltas[i] = v - prev_val
        prev_val = v
        prev_idx = i
    # compute stats of deltas on observed deltas
    delta_vals = deltas[~np.isnan(deltas)]
    if len(delta_vals) == 0:
        delta_std = 1e-6
    else:
        delta_std = np.std(delta_vals, ddof=0)

    # identify spikes by z-score OR by delta jump
    for i, t in enumerate(time_stamps):
        val = scores[i]
        if np.isnan(val):
            continue
        z = z_scores[i]
        d = deltas[i]
        is_z_spike = (z >= Z_THRESH)
        is_delta_spike = (not np.isnan(d)) and (abs(d) >= DELTA_MULT * max(1e-6, delta_std))
        if is_z_spike or is_delta_spike:
            spike_records.append({
                "paper_id": pid,
                "time": t,
                "score": val,
                "z_score": z,
                "delta": d,
                "is_z_spike": bool(is_z_spike),
                "is_delta_spike": bool(is_delta_spike),
                "n_observations": int(n_obs),
                "mean_score": float(mean),
                "std_score": float(std)
            })

# aggregate spikes into DataFrame
spikes_df = pd.DataFrame(spike_records)
spikes_df = spikes_df.sort_values(["paper_id","time"])

if SAVE_CSV:
    spikes_df.to_csv("detected_spikes_per_node.csv", index=False)

print("Total spikes found:", len(spikes_df))
print(spikes_df.head(20))

# ---------- OPTIONAL: show top nodes by number of spikes ----------
top_nodes = spikes_df['paper_id'].value_counts().head(20)
print("\nTop nodes with most spikes:")
print(top_nodes)

# ---------- OPTIONAL: function to plot a node timeline and highlight spikes ----------
def plot_node_timeline(paper_id, save=False):
    if paper_id not in vectors_df.index:
        print("paper_id not found:", paper_id)
        return
    row = vectors_df.loc[paper_id]
    scores = row[time_cols].values.astype(float)
    plt.figure(figsize=(10,4))
    plt.plot(time_stamps, scores, marker='o', label='anomaly_score')
    # highlight spikes for this node
    node_spikes = spikes_df[spikes_df['paper_id']==paper_id]
    if not node_spikes.empty:
        spike_times = node_spikes['time'].tolist()
        spike_scores = node_spikes['score'].tolist()
        plt.scatter(spike_times, spike_scores, color='red', s=80, label='spike')
    plt.title(f"Node {paper_id} anomaly timeline")
    plt.xlabel("time (segment end year)")
    plt.ylabel("anomaly score")
    plt.grid(alpha=0.3)
    plt.legend()
    if save:
        plt.savefig(f"timeline_{paper_id}.png", bbox_inches='tight')
    plt.show()

# Example: plot top node
if not spikes_df.empty:
    top_pid = spikes_df['paper_id'].value_counts().index[0]
    plot_node_timeline(top_pid)