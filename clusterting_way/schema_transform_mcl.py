import networkx as nx
import markov_clustering as mc
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import pandas as pd

# Your dataset
data = {
    "Lineitem": [1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20],
    "Orders": [3, 4, 5, 7, 8, 9, 10, 12, 13, 17, 20, 21],
    "Customer": [3, 5, 7, 8, 10, 12, 17, 21],
    "Supplier": [2, 5, 7, 8, 9, 11, 15, 19, 20],
    "Part": [2, 8, 9, 14, 15, 16, 18, 19],
    "PartSupp": [2, 9, 11, 16, 19],
    "Region": [2, 5, 8],
    "Nation": [2, 5, 7, 8, 9, 10, 11, 19, 20],
}

# Step 1: Create a co-occurrence matrix between tables based on shared IDs
tables = list(data.keys())
num_tables = len(tables)
co_matrix = np.zeros((num_tables, num_tables))

for i in range(num_tables):
    for j in range(num_tables):
        if i != j:
            co_matrix[i, j] = len(set(data[tables[i]]) & set(data[tables[j]]))

# Normalize the matrix row-wise for MCL
norm_matrix = co_matrix / np.maximum(co_matrix.sum(axis=1, keepdims=True), 1)

# Convert to sparse matrix
sparse_matrix = csr_matrix(norm_matrix)

# Step 2: Run MCL
result = mc.run_mcl(sparse_matrix, inflation=2.0)
clusters = mc.get_clusters(result)

# Step 3: Show cluster result
print("🔷 MCL Clustering Result:")
for i, cluster in enumerate(clusters, 1):
    print(f"Cluster {i}: {[tables[idx] for idx in cluster]}")

# Step 4: PCA Visualization (similar to your agglomerative version)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(norm_matrix)

# Assign each table to its cluster
cluster_map = {}
for cluster_id, cluster in enumerate(clusters):
    for idx in cluster:
        cluster_map[idx] = cluster_id + 1

# Plotting
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
for idx in range(len(tables)):
    cluster_id = cluster_map[idx]
    plt.scatter(pca_result[idx, 0], pca_result[idx, 1], color=colors[cluster_id % len(colors)], label=f"Cluster {cluster_id}" if f"Cluster {cluster_id}" not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(pca_result[idx, 0], pca_result[idx, 1], tables[idx], fontsize=9)

plt.title("MCL Table Clustering (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
