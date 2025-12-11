import pandas as pd
import ast
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\nadir\\FinalProject\\LLGC-main\\final_filtered_by_fos_and_reference.csv")

# Convert references column to list lengths
df["num_references"] = df["references"].apply(
    lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0
)

# Distribution of references
distribution = df["num_references"].value_counts().sort_index()
print(distribution)

# Plot histogram with numbers on top
plt.figure(figsize=(12,6))
bars = plt.bar(distribution.index, distribution.values)
plt.xlabel("Number of References")
plt.ylabel("Number of Papers")
plt.title("Distribution of References per Paper")

# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, str(height), ha='center', va='bottom')

plt.show()
