# This is a custom implementation of the Davies-Bouldin Score metric. Reason for the need of a custom implementation
# is included in the doc.

import numpy as np
from scipy.sparse import csr_matrix


class CustomDaviesBouldin:

    def __init__(self, n_clusters, metric='euclidean'):
        # Store the number of clusters
        self.n_clusters = n_clusters
        # Store the distance metric ('euclidean' or 'cosine')
        self.metric = metric


    def _distance_user_centroid(self, user_idx, centroid, X):
        """
        Compute the distance between a user and a cluster centroid using the specified metric
        :param user_idx: Index of the user in the sparse matrix.
        :param centroid: The centroid vector of the cluster.
        :param X: User-item rating matrix in CSR format
        :return: Distance between the user and centroid on common dimensions according to the chosen metric (None if
        no common movies)
        """
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

        # Compute distance based on the specified metric
        user_common = ratings[mask]
        centroid_common = centroid_vals[mask]

        if self.metric == 'euclidean':
            # Euclidean distance
            diff = user_common - centroid_common
            return np.sqrt(np.sum(diff ** 2))
        else:  # cosine distance
            # Cosine similarity on common movies
            numerator = np.abs(np.sum(user_common * centroid_common))
            user_norm = np.sqrt(np.sum(user_common ** 2))
            centroid_norm = np.sqrt(np.sum(centroid_common ** 2))
            if user_norm > 0 and centroid_norm > 0:
                cosine_sim = numerator / (user_norm * centroid_norm)
            else:
                cosine_sim = 0
            return 1 - cosine_sim


    def _distance_centroid_centroid(self, c1, c2):
        """
        Compute the distance between two cluster centroids using the specified metric
        :param c1: First centroid vector.
        :param c2: Second centroid vector
        :return: Distance between centroids on common dimensions according to the chosen metric (None if there are no
        overlapping dimensions)
        """
        # Get the "common dimensions" of the two centroids
        mask = (c1 > 0) & (c2 > 0)

        # If centroids have no overlap, distance is undefined
        if not np.any(mask):
            return None

        # Compute distance based on the specified metric
        c1_common = c1[mask]
        c2_common = c2[mask]

        if self.metric == 'euclidean':
            # Euclidean distance
            diff = c1_common - c2_common
            return np.sqrt(np.sum(diff ** 2))
        else:  # cosine distance
            # Cosine similarity
            numerator = np.abs(np.sum(c1_common * c2_common))
            norm1 = np.sqrt(np.sum(c1_common ** 2))
            norm2 = np.sqrt(np.sum(c2_common ** 2))
            if norm1 > 0 and norm2 > 0:
                cosine_sim = numerator / (norm1 * norm2)
            else:
                cosine_sim = 0
            return 1 - cosine_sim


    def _compute_scatter(self, X, labels, centroids):
        """
        Compute the scatter (within-cluster dispersion) for each cluster.
        :param X: User-item rating matrix in CSR format
        :param labels: Array of cluster assignments for each user
        :param centroids: Array of centroid vectors for each cluster
        :return: Array of scatter values, one per cluster
        """
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
        """
        Compute the Davies-Bouldin score for a clustering solution
        :param X: User-item rating matrix (will be converted to CSR format if needed)
        :param labels: Array of cluster assignments for each user
        :param centroids: Array of centroid vectors for each cluster
        :return: Davies-Bouldin score (average over all clusters).
        """
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