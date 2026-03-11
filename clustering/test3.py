import numpy as np


class KMeansEuclidean:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = None

    @staticmethod
    def dist_user_centroid_vectorized(users, centroids):
        """
        Vectorized distance calculation between all users and all centroids
        """
        # Ensure inputs are float type to avoid integer operations
        users = users.astype(float)
        centroids = centroids.astype(float)

        n_users = users.shape[0]
        n_centroids = centroids.shape[0]

        # Find coordinates where both user and centroid have positive values
        both = (users[:, np.newaxis, :] > 0) & (centroids[np.newaxis, :, :] > 0)

        # Create result array filled with infinity
        distances = np.full((n_users, n_centroids), np.inf, dtype=float)

        # For each centroid
        for i in range(n_centroids):
            # Get users that have at least one positive coordinate with this centroid
            valid_users = np.any(both[:, i, :], axis=1)
            if np.any(valid_users):
                # Compute squared differences
                diff = users[valid_users, :] - centroids[i, :]
                diff_squared = diff ** 2

                # Sum only where both were positive
                sum_squares = np.sum(diff_squared * both[valid_users, i, :], axis=1)

                # Calculate sqrt and assign (avoid sqrt of negative numbers)
                distances[valid_users, i] = np.sqrt(np.maximum(sum_squares, 0))

        return distances

    def update_centroids(self, cluster_points_list):
        """
        Vectorized centroid update for all clusters at once
        """
        n_clusters = len(cluster_points_list)
        if n_clusters == 0:
            return np.array([])

        # Determine feature dimension from first non-empty cluster
        for points in cluster_points_list:
            if len(points) > 0:
                n_features = points.shape[1]
                break
        else:
            return np.array([])

        new_centroids = np.zeros((n_clusters, n_features), dtype=float)

        for i, points in enumerate(cluster_points_list):
            if len(points) > 0:
                # Ensure points are float
                points = points.astype(float)

                # Create mask for rated items
                rated_mask = points > 0

                # For each feature, compute mean only where rated
                for j in range(n_features):
                    rated_values = points[rated_mask[:, j], j]
                    if len(rated_values) > 0:
                        new_centroids[i, j] = np.mean(rated_values)
                    # else: leave as 0 (already initialized to 0)
            else:
                # Keep old centroid for empty clusters
                new_centroids[i] = self.centroids[i].copy()

        return new_centroids

    def fit(self, X, max_iters=200, tol=1e-6):
        # Convert X to float to avoid integer operations
        X = X.astype(float)

        # Initialize centroids with random data points
        n = self.n_clusters
        total_rows = X.shape[0]
        random_indices = np.random.choice(total_rows, size=n, replace=False)
        self.centroids = X[random_indices, :].copy()

        # Pre-allocate arrays for better memory efficiency
        labels = np.zeros(X.shape[0], dtype=int)

        for iteration in range(max_iters):

            print(f"Iteration {iteration+1}")

            # Step 1: Vectorized assignment
            distances = self.dist_user_centroid_vectorized(X, self.centroids)
            new_labels = np.argmin(distances, axis=1)

            # Step 2: Vectorized centroid update
            cluster_points = [X[new_labels == i] for i in range(self.n_clusters)]
            new_centroids = self.update_centroids(cluster_points)

            # Step 3: Check convergence
            if np.max(np.abs(new_centroids - self.centroids)) < tol:
                self.centroids = new_centroids
                labels = new_labels
                break

            self.centroids = new_centroids
            labels = new_labels

            if (iteration + 1) % 10 == 0:
                print(f'Iteration {iteration + 1}')

        return labels
