import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Data input: Dictionary of tables and the queries they are used in
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

# Step 1: Create a binary matrix
queries = range(1, 22)  # Query IDs (1 to 21)
binary_matrix = pd.DataFrame(0, index=data.keys(), columns=queries)

for table, query_ids in data.items():
    binary_matrix.loc[table, query_ids] = 1

# Step 2: Determine optimal number of clusters using Elbow Method# Step 2: Determine optimal number of clusters using Elbow Method
wcss = []
range_clusters = range(1, len(binary_matrix) + 1)  # Max clusters = number of rows (tables)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(binary_matrix)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.plot(range_clusters, wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Step 3: Choose optimal k and perform clustering
optimal_k = 4  # Set based on the Elbow Method result
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
binary_matrix['Cluster'] = kmeans.fit_predict(binary_matrix)

# Step 4: Display tables in each cluster
clusters = binary_matrix.groupby('Cluster').apply(lambda x: list(x.index)).to_dict()

print("Tables in each cluster:")
for cluster_id, tables in clusters.items():
    print(f"Cluster {cluster_id}: {tables}")

