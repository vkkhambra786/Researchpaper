

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Step 1: Define the data (binary matrix)
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


# Convert the data into a binary matrix
binary_matrix = pd.DataFrame([
    [1 if i + 1 in data[key] else 0 for key in data] for i in range(1, 21)
], columns=data.keys())

# Check for duplicate rows
duplicate_count = binary_matrix.duplicated().sum()
if duplicate_count > 0:
    print(f"Warning: Found {duplicate_count} duplicate rows. Consider deduplicating your data.")

# Step 2: Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares
range_clusters = range(1, len(binary_matrix) + 1)  # Possible values for k

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
plt.show(block=False)  # Show the plot non-blocking
plt.pause(3)           # Pause for 3 seconds
plt.close()            # Close the plot after 3 seconds

# Step 3: Choose the optimal number of clusters based on the Elbow Graph
optimal_k = int(input("Enter the optimal number of clusters (k) based on the Elbow graph: "))

# Step 4: Apply K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
binary_matrix['Cluster'] = kmeans.fit_predict(binary_matrix)

# Step 5: Display the tables in each cluster
#clusters = binary_matrix.groupby('Cluster').apply(lambda x: list(x.index + 1)).to_dict()
# Display the tables in each cluster without the deprecation warning
clusters = binary_matrix.groupby('Cluster', group_keys=False).apply(lambda x: list(x.index + 1)).to_dict()
print("clusters",clusters)
print("\nTables in each cluster:")
for cluster_id, tables in clusters.items():
    print(f"Cluster {cluster_id}: {tables}")
