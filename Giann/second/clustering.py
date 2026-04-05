
import numpy as np
from scipy.sparse import csr_matrix
from assignment.second.KMeans_clustering import CustomKMeans
from assignment.second.clustering_validity_measures import CustomDaviesBouldin
from pathlib import Path
from datetime import datetime
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


# Specify directories paths
ROOT = Path(__file__).parent.parent
DATASETS_DIR = ROOT / 'datasets'
PLOTS_DIR = ROOT / 'plots/clustering_plots'
LOGS_DIR = ROOT / 'logs'

# ---------------------------
# Load dataset
# ---------------------------
X = np.load(DATASETS_DIR / 'feature_matrix.npy', allow_pickle=True)
print('Dataset loaded:', X.shape)

# Ensure CSR format for efficiency
if not isinstance(X, csr_matrix):
    X = csr_matrix(X)

# Decide number of clusters and distance metric to perform clustering
n_clusters = 30
metric:str = 'euclidean' # ('euclidean' or 'cosine')

timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
log_file = LOGS_DIR / f'clustering_{metric}_L{n_clusters}_{timestamp}.log'

# ---------------------------
# Fit KMeans
# ---------------------------

kmeans = CustomKMeans(
    n_clusters=n_clusters,
    metric=metric,
    log_file=log_file
)
labels = kmeans.fit(X, max_iters=200, tol=1e-6)
print("KMeans clustering completed.")

# ---------------------------
# Custom Davies-Bouldin Index
# ---------------------------
db_metric = CustomDaviesBouldin(n_clusters=n_clusters, metric=metric)
db_score = db_metric.score(X, labels, kmeans.centroids)
print(f"Custom Davies-Bouldin Index: {db_score:.4f}")

# ---------------------------
# PCA / TruncatedSVD for visualization
# ---------------------------
n_components = 2
svd = TruncatedSVD(n_components=n_components, random_state=42)

# Fit and transform the user matrix
X_reduced = svd.fit_transform(X)

# Project centroids as well
centroids_reduced = svd.transform(kmeans.centroids)

# ---------------------------
# Plot clusters
# ---------------------------
plt.figure(figsize=(10, 8))

colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

color = colors[:n_clusters]

for k in range(n_clusters):
    cluster_points = X_reduced[labels == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                s=10, color=colors[k], alpha=0.5, label=f'Cluster {k}')

# Plot centroids
plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1],
            s=200, color='black', marker='X', label='Centroids')

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA Visualization of K-Means Clusters')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(PLOTS_DIR / f'Clustering visualization for {n_clusters} clusters (metric = {metric}).png')
plt.show()


# ---------------------------
# Hyper-plot: one scatter per cluster
# ---------------------------

n_cols = 4  # number of plots per row
n_rows = math.ceil(n_clusters / n_cols)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4 * n_cols, 4 * n_rows),
                         sharex=True, sharey=True)

axes = axes.flatten()  # make indexing easy

# Optional: sample points per cluster to reduce overplotting
subset_size = 3000  # adjust for speed and clarity

for k in range(n_clusters):
    ax = axes[k]
    cluster_points = X_reduced[labels == k]

    if cluster_points.shape[0] > subset_size:
        idx = np.random.choice(cluster_points.shape[0], subset_size, replace=False)
        cluster_points = cluster_points[idx]

    ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
               s=10, color=colors[k], alpha=0.5)
    ax.scatter(centroids_reduced[k, 0], centroids_reduced[k, 1],
               s=100, color='black', marker='X', label='Centroid')
    ax.set_title(f'Cluster {k}')
    ax.grid(True, alpha=0.3)

fig.suptitle('PCA Scatter Plots per Cluster')
plt.savefig(PLOTS_DIR / f'PCA Scatter Plots per Cluster ({n_clusters} clusters, metric={metric}).png')
plt.show()



