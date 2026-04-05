
import numpy as np
from scipy.sparse import csr_matrix
from assignment.second.KMeans_clustering import CustomKMeans
from assignment.second.clustering_validity_measures import CustomDaviesBouldin
from pathlib import Path
import matplotlib.pyplot as plt


# Specify directories paths
ROOT = Path(__file__).parent.parent
DATASETS_DIR = ROOT / 'datasets'
PLOTS_DIR = ROOT / 'plots/clustering'


# ---------------------------
# Load dataset
# ---------------------------
X = np.load(DATASETS_DIR / 'feature_matrix.npy', allow_pickle=True)
print('Dataset loaded:', X.shape)

# Ensure CSR format for efficiency
if not isinstance(X, csr_matrix):
    X = csr_matrix(X)

# ---------------------------
# Parameters
# ---------------------------
metric = 'euclidean'  # or 'cosine'
min_clusters = 3
max_clusters = 50
n_iterations = 3  # Number of runs per cluster count (increase for more robust results)

# ---------------------------
# Compute Davies-Bouldin for different cluster counts
# ---------------------------
db_scores = []
cluster_range = range(min_clusters, max_clusters + 1)

for n_clusters in cluster_range:
    print(f"\n{'=' * 50}")
    print(f"Testing n_clusters = {n_clusters}")
    print(f"{'=' * 50}")

    cluster_db_scores = []

    for run in range(n_iterations):
        print(f"  Run {run + 1}/{n_iterations}")

        # Fit KMeans
        kmeans = CustomKMeans(n_clusters=n_clusters, metric=metric)
        labels = kmeans.fit(X, max_iters=200, tol=1e-6)

        # Calculate Davies-Bouldin Index
        db_metric = CustomDaviesBouldin(n_clusters=n_clusters, metric=metric)
        db_score = db_metric.score(X, labels, kmeans.centroids)
        cluster_db_scores.append(db_score)

    # Average score across runs
    avg_db_score = np.mean(cluster_db_scores)
    db_scores.append(avg_db_score)
    print(f"  Average DB Index: {avg_db_score:.4f}")

# ---------------------------
# Plot results
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(cluster_range, db_scores, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Davies-Bouldin Index', fontsize=12)
plt.title(f'Davies-Bouldin Index vs Number of Clusters (metric = {metric})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(cluster_range[::5])  # Show every 5th cluster number
plt.tight_layout()
plt.show()

# Print optimal k
optimal_k = cluster_range[np.argmin(db_scores)]
print(f"\n{'=' * 50}")
print(f"Optimal number of clusters: {optimal_k}")
print(f"Lowest Davies-Bouldin Index: {min(db_scores):.4f}")
print(f"{'=' * 50}")