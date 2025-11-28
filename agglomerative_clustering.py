import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Input Data: Binary Mapping
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

# Create a binary matrix
binary_matrix = pd.DataFrame([
    [1 if i + 1 in data[key] else 0 for key in data] for i in range(21)
], columns=data.keys())

# Transpose to match the clustering format
binary_matrix = binary_matrix.T

# Compute Jaccard Distance
jaccard_distance = pdist(binary_matrix, metric='jaccard')

# Perform Agglomerative Clustering
linkage_matrix = linkage(jaccard_distance, method='average')  # 'average', 'single', or 'complete'

# Plot Dendrogram to visualize clusters
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=binary_matrix.index.tolist(), leaf_rotation=90)
plt.title('Agglomerative Clustering with Jaccard Distance')
plt.xlabel('Tables')
plt.ylabel('Distance')
plt.show()

# Determine clusters (e.g., 3 clusters)
num_clusters = 3
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Assign tables to clusters
clustered_tables = pd.DataFrame({
    'Table': binary_matrix.index,
    'Cluster': clusters
}).groupby('Cluster')['Table'].apply(list)

# Output clusters
print("Tables in each cluster:")
for cluster_id, tables in clustered_tables.items():
    print(f"Cluster {cluster_id}: {tables}")

# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# import matplotlib.pyplot as plt

# # Input Data: Binary Mapping
# data = {
#     "Lineitem": [1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20],
#     "Orders": [3, 4, 5, 7, 8, 9, 10, 12, 13, 17, 20, 21],
#     "Customer": [3, 5, 7, 8, 10, 12, 17, 21],
#     "Supplier": [2, 5, 7, 8, 9, 11, 15, 19, 20],
#     "Part": [2, 8, 9, 14, 15, 16, 18, 19],
#     "PartSupp": [2, 9, 11, 16, 19],
#     "Region": [2, 5, 8],
#     "Nation": [2, 5, 7, 8, 9, 10, 11, 19, 20],
# }

# # Create a binary matrix
# binary_matrix = pd.DataFrame([
#     [1 if i + 1 in data[key] else 0 for key in data] for i in range(21)
# ], columns=data.keys())

# # Transpose to match the clustering format
# binary_matrix = binary_matrix.T

# # Compute Cosine Distance
# cosine_distance = pdist(binary_matrix, metric='cosine')

# # Perform Agglomerative Clustering
# linkage_matrix = linkage(cosine_distance, method='average')  # 'average', 'single', or 'complete'

# # Plot Dendrogram to visualize clusters
# plt.figure(figsize=(10, 7))
# dendrogram(linkage_matrix, labels=binary_matrix.index.tolist(), leaf_rotation=90)
# plt.title('Agglomerative Clustering with Cosine Distance')
# plt.xlabel('Tables')
# plt.ylabel('Distance')
# plt.show()

# # Determine clusters (e.g., 3 clusters)
# num_clusters = 3
# clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# # Assign tables to clusters
# clustered_tables = pd.DataFrame({
#     'Table': binary_matrix.index,
#     'Cluster': clusters
# }).groupby('Cluster')['Table'].apply(list)

# # Output clusters
# print("Tables in each cluster:")
# for cluster_id, tables in clustered_tables.items():
#     print(f"Cluster {cluster_id}: {tables}")