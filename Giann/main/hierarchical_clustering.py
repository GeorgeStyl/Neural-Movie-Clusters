
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


# Specify directories paths
ROOT = Path(__file__).parent.parent
DATASETS_DIR = ROOT / 'datasets'

# Load matrices
feature_matrix = np.load(DATASETS_DIR / 'feature_matrix.npy')
print(f'Feature matrix loaded. Shape: {feature_matrix.shape}')

distance_matrix = np.load(DATASETS_DIR / 'distance_matrix_condensed.npy')
print('\nDistance matrix loaded.')

# Confirm
print(f'Shape: {distance_matrix.shape}')

# Perform hierarchical clustering
Z = linkage(distance_matrix, method='average')

print('\nLinkage matrix computed.')
print(f'Z shape: {Z.shape}')

# Extract clusters based on average distance of pairs of points inside each cluster
labels = fcluster(Z, t=0.985, criterion='distance')

print(f'\nNumber of clusters found: {len(set(labels))}')

# Get and show cluster sizes (ignore clusters with small size)
unique, counts = np.unique(labels, return_counts=True)
print('\nCluster sizes (focusing on meaningful clusters only):')
for u, c in zip(unique, counts):
    if c > 100:
        print(f'Cluster {u}: {c} users')


min_cluster_size = 100
large_clusters = [u for u, c in zip(unique, counts) if c >= min_cluster_size]

# Create mask for points in large clusters
mask = np.isin(labels, large_clusters)
filtered_labels = labels[mask]

# Convert condensed distance matrix to square form
square_dist = squareform(distance_matrix)

# Get number of original training points
n_original = len(feature_matrix)

# Extract submatrix for points in large clusters
filtered_indices = np.where(mask)[0]
filtered_dist_matrix = square_dist[np.ix_(filtered_indices, filtered_indices)]

# Filter original feature matrix
filtered_features = feature_matrix[filtered_indices]


# # === Plot ===
#
# # Apply PCA
# pca = PCA(n_components=2)
# coords_2d = pca.fit_transform(filtered_features)
#
# # Plot clusters in same grid
# plt.figure(figsize=(8, 6))
#
# for cluster_id in large_clusters:
#     cluster_points = coords_2d[filtered_labels == cluster_id]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', alpha=0.6)
#
# plt.title('Large Clusters Visualization (PCA)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()
# plt.show()
#
# # Plot each cluster in separate grid
# for cluster_id in large_clusters:
#     cluster_points = coords_2d[filtered_labels == cluster_id]
#
#     plt.figure(figsize=(6, 5))
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.6)
#
#     plt.title(f'Cluster {cluster_id} ({len(cluster_points)} pts)')
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#
#     plt.show()


# === Compute validity measures for large clusters ===

print(f'\n=== Metrics for clusters with more than {min_cluster_size} users ===')

# Silhouette Score
if len(set(filtered_labels)) >= 2:
    silhouette_avg = silhouette_score(filtered_dist_matrix, filtered_labels, metric='precomputed')
    print(f'Silhouette Score: {silhouette_avg:.4f}')

# Davies-Bouldin Index
if len(set(filtered_labels)) >= 2:
    db_index = davies_bouldin_score(filtered_dist_matrix, filtered_labels)
    print(f'Davies-Bouldin Index: {db_index:.4f}')

# Intra-cluster statistics
print('\nIntra-cluster distances:')
for cluster_id in large_clusters:
    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]

    if len(cluster_indices) > 1:
        cluster_dist = square_dist[np.ix_(cluster_indices, cluster_indices)]
        upper_tri = cluster_dist[np.triu_indices_from(cluster_dist, k=1)]

        avg_dist = np.mean(upper_tri)
        max_dist = np.max(upper_tri)

        print(f'Cluster {cluster_id} ({len(cluster_indices)} users): avg={avg_dist:.4f}, max={max_dist:.4f}')


# Save labels for large clusters only
np.save(DATASETS_DIR / 'cluster_labels.npy', filtered_labels)


