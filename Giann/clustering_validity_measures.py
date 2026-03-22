
import numpy as np
from scipy.sparse import csr_matrix


# This is a custom implementation of the Davies-Bouldin Score metric. Reason for the need of a custom implementation
# is included in the doc.
class CustomDaviesBouldin:

    def __init__(self, n_clusters):
        # Store the number of clusters
        self.n_clusters = n_clusters

    def _distance_user_centroid(self, user_idx, centroid, X):

        # Extract user's data from CSR matrix
        start = X.indptr[user_idx]
        end = X.indptr[user_idx + 1]

        # Get the movies this user rated and the corresponding ratings
        movies = X.indices[start:end]
        ratings = X.data[start:end]

        # Get the coordinates of the centroid only for its common with the user dimensions (movies)
        centroid_vals = centroid[movies]
        mask = centroid_vals > 0

        # If there are no common movies, distance is undefined
        if not np.any(mask):
            return None

        # Compute and return the distance between user and centroid (on common movies only)
        diff = ratings[mask] - centroid_vals[mask]
        return np.sqrt(np.sum(diff ** 2))

    def _distance_centroid_centroid(self, c1, c2):

        # Get the "common dimensions" of the two centroids
        mask = (c1 > 0) & (c2 > 0)

        # If centroids have no overlap, distance is undefined
        if not np.any(mask):
            return None

        # Compute and return their distance (on common dimensions only)
        diff = c1[mask] - c2[mask]
        return np.sqrt(np.sum(diff ** 2))

    def _compute_scatter(self, X, labels, centroids):

        # Initialize scatter values for each cluster
        S = np.zeros(self.n_clusters)

        # Compute scatter for each cluster independently
        for k in range(self.n_clusters):

            # Get indices of users assigned to this cluster
            cluster_users = np.where(labels == k)[0]

            # If cluster is empty, scatter is zero
            if len(cluster_users) == 0:
                S[k] = 0
                continue

            distances = []

            # Compute distance of each user to the cluster centroid
            for u in cluster_users:
                d = self._distance_user_centroid(u, centroids[k], X)

                # Ignore users with no overlap with centroid
                if d is not None:
                    distances.append(d)

            # If no valid distances exist, set scatter to zero
            if len(distances) == 0:
                S[k] = 0
            else:
                # Scatter is the mean distance to centroid
                S[k] = np.mean(distances)

        return S

    def score(self, X, labels, centroids):

        # Ensure matrix is in CSR format for efficient row access
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)

        # Step 1: Compute cluster scatter values
        S = self._compute_scatter(X, labels, centroids)

        DB_values = []

        # Step 2: For each cluster, compute its worst-case similarity to another cluster
        for i in range(self.n_clusters):

            # Initialize maximum ratio for cluster i
            max_ratio = -np.inf

            for j in range(self.n_clusters):

                # Skip comparison with itself
                if i == j:
                    continue

                # Compute distance between centroids i and j
                M_ij = self._distance_centroid_centroid(
                    centroids[i], centroids[j]
                )

                # Skip if centroids do not overlap or distance is zero
                if M_ij is None or M_ij == 0:
                    continue

                # Compute similarity ratio between clusters i and j
                ratio = (S[i] + S[j]) / M_ij

                # Keep the worst (maximum) ratio
                if ratio > max_ratio:
                    max_ratio = ratio

            # If no valid comparison exists, set value to zero
            if max_ratio == -np.inf:
                max_ratio = 0

            DB_values.append(max_ratio)

        # Step 3: Return the average over all clusters
        return np.mean(DB_values)


class CustomSilhouetteScore:
    """
    Sparse-aware silhouette score for clustering user-item matrices.
    Computes s(i) for each user using only overlapping features.
    Supports optional sampling to reduce computation time.
    Shows progress during computation.
    """
    def __init__(self, n_clusters, sample_size=None, progress_every=500):
        self.n_clusters = n_clusters
        self.sample_size = sample_size  # number of users to sample for silhouette calculation
        self.progress_every = progress_every  # print progress every N users

    def _distance_users(self, u1_idx, u2_idx, X):
        start1, end1 = X.indptr[u1_idx], X.indptr[u1_idx + 1]
        movies1, ratings1 = X.indices[start1:end1], X.data[start1:end1]

        start2, end2 = X.indptr[u2_idx], X.indptr[u2_idx + 1]
        movies2, ratings2 = X.indices[start2:end2], X.data[start2:end2]

        common = np.intersect1d(movies1, movies2, assume_unique=True)
        if common.size == 0:
            return 0.0

        idx1 = np.searchsorted(movies1, common)
        idx2 = np.searchsorted(movies2, common)
        diff = ratings1[idx1] - ratings2[idx2]

        return np.sqrt(np.sum(diff ** 2))

    def _average_distance_to_cluster(self, user_idx, cluster_indices, X):
        if len(cluster_indices) == 0:
            return 0.0
        distances = [self._distance_users(user_idx, other_idx, X)
                     for other_idx in cluster_indices if other_idx != user_idx]
        if len(distances) == 0:
            return 0.0
        return np.mean(distances)

    def score(self, X, labels):
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)

        n_users = X.shape[0]

        # Optional sampling
        if self.sample_size is not None and n_users > self.sample_size:
            idx = np.random.choice(n_users, self.sample_size, replace=False)
        else:
            idx = np.arange(n_users)

        # Precompute users in each cluster
        clusters = [np.where(labels == k)[0] for k in range(self.n_clusters)]

        silhouette_values = []

        print(f"Computing silhouette score for {len(idx)} users...")
        for count, u in enumerate(idx, start=1):
            own_cluster = labels[u]
            a_i = self._average_distance_to_cluster(u, clusters[own_cluster], X)

            b_i = np.inf
            for k, cluster_users in enumerate(clusters):
                if k == own_cluster or len(cluster_users) == 0:
                    continue
                dist = self._average_distance_to_cluster(u, cluster_users, X)
                if dist < b_i:
                    b_i = dist

            s_i = 0.0 if max(a_i, b_i) == 0 else (b_i - a_i) / max(a_i, b_i)
            silhouette_values.append(s_i)

            # Print progress every N users
            if count % self.progress_every == 0 or count == len(idx):
                print(f"Processed {count}/{len(idx)} users ({count/len(idx)*100:.1f}%)")

        print("Silhouette computation completed.")
        return np.mean(silhouette_values)