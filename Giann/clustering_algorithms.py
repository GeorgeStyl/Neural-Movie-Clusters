
import numpy as np
from scipy.sparse import csr_matrix


# K-Means clustering algorithm optimized for sparse user-movie rating data.
class KMeansEuclidean:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = None


    def _distance_user_centroid(self, user_idx, X):
        """
        Compute distances from a single user to all centroids
        :param user_idx: Index of user in the sparse matrix
        :param X: CSR sparse matrix of shape (n_users, n_features)
        :return: Index of the closest centroid, or -1 if no centroid shares movies with user
        """

        # Extract user's data from CSR matrix
        start = X.indptr[user_idx]         # Where this user's data starts
        end = X.indptr[user_idx + 1]       # Where this user's data ends

        # Get the movies (by grabbing their indices in the feature matrix) this user rated and the corresponding ratings
        movies = X.indices[start:end]      # Array of movie indices
        ratings = X.data[start:end]        # Array of rating values

        best_cluster = -1                  # Index of current cluster with nearest centroid to user
        best_dist = np.inf                 # Current distance to nearest centroid

        # Check distance to centroids
        # For each centroid
        for k in range(self.n_clusters):

            # Get the centroid vector
            centroid = self.centroids[k]

            # Extract centroid's values for movies this user has rated
            centroid_vals = centroid[movies]

            # Find which of these movies the centroid also "rated" (its vector has positive values in the respective
            # coordinates)
            mask = centroid_vals > 0

            # If centroid and user have no "common movies", skip
            if not np.any(mask):
                continue

            # Compute Euclidean distance only on common movies
            diff = ratings[mask] - centroid_vals[mask]
            dist = np.sqrt(np.sum(diff ** 2))

            # Keep track of the closest centroid
            if dist < best_dist:
                best_dist = dist
                best_cluster = k

        return best_cluster


    def _assign_clusters(self, X):
        """
        Assign every user to their nearest centroids.
        :param X: CSR sparse matrix of shape (n_users, n_features)
        :return: Array of cluster labels for each user, shape (n_users,)
        """

        # Get the number of users
        n_users = X.shape[0]

        # Initialize the "labels" array
        labels = np.zeros(n_users, dtype=int)

        # Update every user's label using the distance function
        for u in range(n_users):
            labels[u] = self._distance_user_centroid(u, X)

        return labels


    def _update_centroids(self, X, labels):
        """
        Update all centroids based on current cluster assignments.
        :param X: CSR sparse matrix of shape (n_users, n_features)
        :param labels: Current cluster assignments for all users
        :return: New centroids array of shape (n_clusters, n_features)
        """

        # Get the dimension of the data
        n_features = X.shape[1]

        # Initialize the new centroids array
        new_centroids = np.zeros((self.n_clusters, n_features))

        # Update each centroid independently. For every cluster label
        for k in range(self.n_clusters):

            # Find all users assigned to this cluster
            cluster_users = np.where(labels == k)[0]

            # If a cluster is empty, keep its old centroid
            if len(cluster_users) == 0:
                new_centroids[k] = self.centroids[k]
                continue

            # Extract sparse matrix (users, movies) for this cluster only
            submatrix = X[cluster_users]

            # Get the total rating sum for each movie across all users in cluster
            sums = submatrix.sum(axis=0).A1

            # Get the number of users who rated each movie
            counts = (submatrix > 0).sum(axis=0).A1

            # Compute means only for movies that were rated by at least one user
            mask = counts > 0
            centroid = np.zeros(n_features)
            centroid[mask] = sums[mask] / counts[mask]

            # Update centroids
            new_centroids[k] = centroid

        return new_centroids


    def fit(self, X, max_iters=100, tol=1e-6):
        """
        Fit K-Means clustering to the data.
        :param X: Sparse matrix of shape (n_users, n_features)
        :param max_iters: Maximum number of iterations
        :param tol: Convergence threshold. Stop when every feature of every centroid changes by less than this value.
        :return: Array of cluster labels for each user, shape (n_users,)
        """

        # Ensure we're working with CSR format for efficient row slicing
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)

        # Get the number of training points (users)
        n_users = X.shape[0]

        # Initialize centroids to be random data points from our feature matrix
        random_indices = np.random.choice(n_users, self.n_clusters, replace=False)
        self.centroids = X[random_indices].toarray()

        # Pre-allocate labels array
        labels = np.zeros(n_users, dtype=int)

        for iteration in range(max_iters):

            # Progress indicator
            print(f"Iteration {iteration + 1}")

            # Step 1: Assignment; assign all users to the cluster with the nearest centroid
            new_labels = self._assign_clusters(X)

            # Step 2: Centroid update
            new_centroids = self._update_centroids(X, new_labels)

            # Step 3: Convergence check
            movement = np.max(np.abs(new_centroids - self.centroids))
            if movement < tol:
                print(f"Converged after {iteration + 1} iterations")
                labels = new_labels
                self.centroids = new_centroids
                break

            # Update for next iteration
            self.centroids = new_centroids
            labels = new_labels

        return labels