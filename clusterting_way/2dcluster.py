# import numpy as np
# import pandas as pd

# # Data points
# data = np.array([[2, 3], [5, 6], [8, 7], [1, 4], [2, 2], [3, 4], [6, 7], [8, 6]])

# # Initial centroids
# centroids = np.array([[2, 3], [5, 6]])

# # Function to calculate Euclidean distance
# def calculate_distance(point, centroid):
#     return np.sqrt(np.sum((point - centroid) ** 2))

# # Assign points to the nearest centroid
# def assign_clusters(data, centroids):
#     clusters = []
#     for point in data:
#         distances = [calculate_distance(point, centroid) for centroid in centroids]
#         clusters.append(np.argmin(distances))  # Assign to the closest centroid
#     return np.array(clusters)

# # Recalculate centroids
# def update_centroids(data, clusters, k):
#     new_centroids = []
#     for i in range(k):
#         cluster_points = data[clusters == i]
#         new_centroids.append(cluster_points.mean(axis=0))
#     return np.array(new_centroids)

# # K-means iteration
# clusters = assign_clusters(data, centroids)  # Step 1: Assign clusters
# updated_centroids = update_centroids(data, clusters, k=2)  # Step 2: Update centroids

# # Print results
# print("Initial Centroids:")
# print(centroids)

# print("\nCluster Assignments:")
# for i, cluster in enumerate(clusters):
#     print(f"Point {data[i]} -> Cluster {cluster}")

# print("\nUpdated Centroids:")
# print(updated_centroids)

# # Verify if the centroids match the expected values
# expected_c1 = np.array([2, 3.25])
# expected_c2 = np.array([6.75, 6.5])

# if np.allclose(updated_centroids[0], expected_c1) and np.allclose(updated_centroids[1], expected_c2):
#     print("\nNAS: The centroids are as expected.")
# else:
#     print("\nThe centroids do not match the expected values.")


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data definition
data = {
    "Lineitem": [1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20],
    "Orders": [3, 4, 5, 7, 8, 9, 10, 12, 13, 17, 20, 21],
    "Customer": [3, 5, 7, 8, 10, 12, 17, 21],
    "Supplier": [2, 5, 7, 8, 9, 11, 15, 19, 20],
    "Part": [2, 8, 9, 14, 15, 16, 18, 19],
    "PartSupp": [2, 9, 11, 16, 19],
    "Region": [2, 5, 8],
    "Nation": [2, 5, 7, 8, 9, 10, 11, 19, 20]
}

# Step 1: Convert data into a binary matrix
num_tables = 21  # Total tables are 1 to 21
binary_matrix = pd.DataFrame(
    [[1 if i + 1 in data[key] else 0 for key in data] for i in range(num_tables)],
    columns=data.keys()
)

# Step 2: Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares
range_clusters = range(1, 10)  # Try k from 1 to 10

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(binary_matrix)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(range_clusters)
plt.grid()
plt.show()

# Step 3: Apply K-means clustering with the optimal number of clusters
optimal_k = int(input("Enter the optimal number of clusters (k) based on the Elbow graph: "))
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
binary_matrix['Cluster'] = kmeans.fit_predict(binary_matrix)

# Step 4: Display the tables in each cluster
clusters = binary_matrix.groupby('Cluster', group_keys=False).apply(lambda x: list(x.index + 1)).to_dict()

print("\nTables in each cluster:")
for cluster_id, tables in clusters.items():
    print(f"Cluster {cluster_id}: {tables}")
