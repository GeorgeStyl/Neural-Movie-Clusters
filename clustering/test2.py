import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from test3 import KMeansEuclidean as KMeans

# -------------------------------------------------
# Efficient validity measures
# -------------------------------------------------

def davies_bouldin_score(X, labels, kmeans_model):
    """
    Davies-Bouldin Index - lower is better
    Complexity: O(n*k) where n=samples, k=clusters
    """
    n_clusters = len(np.unique(labels))

    # Calculate intra-cluster distances (average distance to centroid)
    intra_dists = []
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            # Note: Ensure this method in test3 handles the data type of X
            dists = kmeans_model.dist_user_centroid_vectorized(
                cluster_points, kmeans_model.centroids[k:k + 1])
            intra_dists.append(np.mean(dists))
        else:
            intra_dists.append(0)

    # Calculate Davies-Bouldin
    db_score = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                # Distance between centroids
                centroid_dist = kmeans_model.dist_user_centroid_vectorized(
                    kmeans_model.centroids[i:i + 1],
                    kmeans_model.centroids[j:j + 1]
                )[0, 0]

                if centroid_dist > 0 and centroid_dist != np.inf:
                    ratio = (intra_dists[i] + intra_dists[j]) / centroid_dist
                    max_ratio = max(max_ratio, ratio)

        db_score += max_ratio

    return db_score / n_clusters


def better_coverage_metrics(X, labels):
    """
    More nuanced coverage analysis
    Returns both heavy and light coverage metrics
    """
    n_clusters = len(np.unique(labels))
    heavy_coverages = []
    light_coverages = []

    print("\n--- Detailed Coverage Analysis ---")

    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            # % of movies with > 10% user coverage within the cluster
            heavy_coverage = np.mean(np.mean(cluster_points > 0, axis=0) > 0.1)
            heavy_coverages.append(heavy_coverage)

            # % of movies with > 1% user coverage within the cluster
            light_coverage = np.mean(np.mean(cluster_points > 0, axis=0) > 0.01)
            light_coverages.append(light_coverage)

            print(f"\nCluster {k} ({len(cluster_points)} users):")
            print(f"  Movies with >10% user coverage: {heavy_coverage:.1%}")
            print(f"  Movies with >1% user coverage: {light_coverage:.1%}")

    return {
        'heavy_avg': np.mean(heavy_coverages),
        'light_avg': np.mean(light_coverages),
        'heavy_per_cluster': heavy_coverages,
        'light_per_cluster': light_coverages
    }

# -------------------------------------------------
# Load Data (Handling SciPy Sparse Format)
# -------------------------------------------------

path = '/home/george/WorkSpace/Master/Eks1/ML/Final_Project/data/user_movie_matrix_rmin_7.npz'
print(f"Loading sparse matrix from {path}...")

# Load the sparse matrix
X_sparse = sp.load_npz(path)
print(f"Loaded sparse matrix of shape: {X_sparse.shape}")

# Convert to dense array for the KMeans fit method
# WARNING: If this causes a MemoryError, your KMeans class must be refactored for sparse input.
X = X_sparse.toarray()
print("Converted to dense array. Starting clustering...")

# -------------------------------------------------
# Initialize KMeans
# -------------------------------------------------

k = 5
model = KMeans(n_clusters=k)

# -------------------------------------------------
# Run clustering
# -------------------------------------------------

print("Running KMeans...")
labels = model.fit(X)
print("Clustering finished.")

# -------------------------------------------------
# Print results
# -------------------------------------------------

print("\nCluster labels shape:", labels.shape)
print("Centroids shape:", model.centroids.shape)

# Number of users per cluster
for i in range(k):
    count = np.sum(labels == i)
    print(f"Cluster {i} size: {count}")

# -------------------------------------------------
# Calculate efficient validity measures
# -------------------------------------------------

print("\n--- Cluster Validity Measures ---")

db_score = davies_bouldin_score(X, labels, model)
print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")

coverage_results = better_coverage_metrics(X, labels)

print("\n--- Summary ---")
print(f"Average Heavy Coverage (>10% users): {coverage_results['heavy_avg']:.1%}")
print(f"Average Light Coverage (>1% users): {coverage_results['light_avg']:.1%}")

# -------------------------------------------------
# Simple interpretation
# -------------------------------------------------

print("\n--- Interpretation ---")
if db_score < 1:
    print("✓ Davies-Bouldin: Good separation")
elif db_score < 2:
    print("✓ Davies-Bouldin: Moderate separation")
else:
    print("✗ Davies-Bouldin: Poor separation")

if coverage_results['heavy_avg'] > 0.3:
    print("✓ Heavy coverage: Many movies are popular within clusters")
elif coverage_results['heavy_avg'] > 0.1:
    print("✓ Heavy coverage: Some movies are popular within clusters")
else:
    print("✗ Heavy coverage: Few movies have concentrated interest")

if coverage_results['light_avg'] > 0.8:
    print("✓ Light coverage: Most movies have at least some viewers in each cluster")
else:
    print("✗ Light coverage: Many movies are completely ignored by some clusters")