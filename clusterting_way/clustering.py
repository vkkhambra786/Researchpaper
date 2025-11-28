import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 1: Create the binary matrix
data = {
    "Lineitem": [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    "Orders": [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    "Customer": [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    "Supplier": [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    "Part": [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    "PartSupp": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "Region": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Nation": [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
}

binary_matrix = pd.DataFrame(data)
X = binary_matrix.values  # Convert DataFrame to numpy array

# Step 2: Determine the optimal number of clusters
inertia = []
silhouette_scores = []
clusters_range = range(2, min(10, X.shape[0]))  # Test clusters from 2 to the number of samples

if len(clusters_range) == 0:
    raise ValueError("Not enough data points to form clusters (minimum required: 2).")

for k in clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot the elbow method and silhouette scores
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(clusters_range, inertia, marker='o', label='Inertia')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(clusters_range, silhouette_scores, marker='o', label='Silhouette Score')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.legend()

plt.tight_layout()
#plt.show()
plt.show(block=False)
plt.pause(3)  # Keeps the plot open for 3 seconds
plt.close('all')  # Closes the plot


# Step 3: Apply K-means clustering
optimal_k = clusters_range[silhouette_scores.index(max(silhouette_scores))]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)

binary_matrix['Cluster'] = kmeans.labels_

# Display the clusters
#groups = binary_matrix.groupby('Cluster').apply(lambda x: list(x.index)).to_dict()
groups = binary_matrix.groupby('Cluster', group_keys=False).apply(
    lambda x: list(x.index), include_groups=False
).to_dict()

print("Tables in each cluster:")
for cluster, tables in groups.items():
    print(f"Cluster {cluster}: {tables}")
