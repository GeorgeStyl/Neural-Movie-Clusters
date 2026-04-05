# K-Means clustering algorithm optimized for sparse user-movie rating data. Includes configurable logging to
# # monitor behavior across repeated runs, helping track convergence, cluster stability, and edge cases

import numpy as np
from scipy.sparse import csr_matrix
import logging


class CustomKMeans:

    def __init__(self, n_clusters, metric='euclidean', log_file=None, console_level=logging.INFO,
                 file_level=logging.DEBUG):
        self.n_clusters = n_clusters
        self.metric = metric  # 'euclidean' or 'cosine' only
        self.centroids = None

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Avoid adding multiple handlers if logger already has handlers
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)  # Capture everything

            # Create formatters
            console_formatter = logging.Formatter('%(message)s')  # Simple format for console
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # File handler
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(file_level)
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)


    def _distance_user_centroid(self, user_idx, X):
        """
        Compute distances from a single user to all centroids
        :param user_idx: Index of user in the sparse matrix
        :param X: CSR sparse matrix of shape (n_users, n_features)
        :return: Index of the closest centroid, or -1 if no centroid shares movies with user
        """

        # Extract user's data from CSR matrix
        start = X.indptr[user_idx]  # Where this user's data starts
        end = X.indptr[user_idx + 1]  # Where this user's data ends

        # Get the movies (by grabbing their indices in the feature matrix) this user rated and the corresponding ratings
        movies = X.indices[start:end]  # Array of movie indices
        ratings = X.data[start:end]  # Array of rating values

        best_cluster = -1  # Index of current cluster with nearest centroid to user
        best_dist = np.inf  # Current distance to nearest centroid

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

            # Compute distance only on common movies
            if self.metric == 'euclidean':
                # Euclidean distance
                diff = ratings[mask] - centroid_vals[mask]
                dist = np.sqrt(np.sum(diff ** 2))
            else:  # cosine distance
                # Cosine similarity on common movies
                user_common = ratings[mask]
                centroid_common = centroid_vals[mask]
                numerator = np.abs(np.sum(user_common * centroid_common))
                user_norm = np.sqrt(np.sum(user_common ** 2))
                centroid_norm = np.sqrt(np.sum(centroid_common ** 2))
                if user_norm > 0 and centroid_norm > 0:
                    cosine_sim = numerator / (user_norm * centroid_norm)
                else:
                    cosine_sim = 0
                dist = 1 - cosine_sim

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
                self.logger.debug(f"Cluster {k} is empty, keeping previous centroid")
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


    def fit(self, X, max_iters=100, tol=1e-6, max_restarts=5, empty_cluster_threshold=1):
        """
        Fit K-Means clustering to the data.
        :param X: Sparse matrix of shape (n_users, n_features)
        :param max_iters: Maximum number of iterations
        :param tol: Convergence threshold. Stop when every feature of every centroid changes by less than this value.
        :param max_restarts: Maximum number of re-initializations before giving up
        :param empty_cluster_threshold: How many empty clusters per iteration trigger a restart
        :return: Array of cluster labels for each user, shape (n_users,)
        """

        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)

        n_users = X.shape[0]

        for restart in range(max_restarts):

            self.logger.debug('=' * 60)
            self.logger.debug(f'NEW FIT RUN  (attempt {restart + 1}/{max_restarts})')
            self.logger.debug('=' * 60)
            self.logger.info(
                f"[CONFIG] metric={self.metric} | clusters={self.n_clusters} | max_iters={max_iters} | tol={tol}")
            self.logger.info(f"[DATA]   {n_users} users | {X.shape[1]} movies")

            random_indices = np.random.choice(n_users, self.n_clusters, replace=False)
            self.centroids = X[random_indices].toarray()
            self.logger.info(f"[INIT]   Centroids from user indices: {random_indices.tolist()}")

            labels = np.zeros(n_users, dtype=int)
            converged = False
            restart_needed = False

            for iteration in range(max_iters):

                self.logger.debug(f"  {'- ' * 27}")
                self.logger.debug(f"  ITERATION {iteration + 1:>3}")
                self.logger.debug(f"  {'- ' * 27}")

                new_labels = self._assign_clusters(X)

                assigned = np.sum(new_labels != -1)
                self.logger.debug(f"  [ASSIGN] {assigned}/{n_users} assigned | {n_users - assigned} unassigned")

                # Count empty clusters and decide whether to restart
                empty_clusters = sum(1 for k in range(self.n_clusters) if np.sum(new_labels == k) == 0)
                for k in range(self.n_clusters):
                    self.logger.debug(f"           cluster {k}: {np.sum(new_labels == k):>5} users")

                if empty_clusters >= empty_cluster_threshold:
                    self.logger.warning(
                        f"[RESTART] {empty_clusters} empty cluster(s) at iteration {iteration + 1} "
                        f"— triggering re-initialization (attempt {restart + 1}/{max_restarts})"
                    )
                    restart_needed = True
                    break

                new_centroids = self._update_centroids(X, new_labels)
                movement = np.max(np.abs(new_centroids - self.centroids))
                self.logger.debug(f"  [MOVE]   max centroid movement: {movement:.6f}  (tol={tol})")

                self.centroids = new_centroids
                labels = new_labels

                if movement < tol:
                    converged = True
                    self.logger.info(f"[RESULT] Converged after {iteration + 1} iterations (movement={movement:.2e})")
                    break

            if restart_needed:
                continue  # Jump to next restart attempt

            if not converged:
                self.logger.info(f"[RESULT] Stopped at max_iters={max_iters}, no convergence (movement={movement:.2e})")

            self.logger.info("[FINAL]  Cluster distribution:")
            for k in range(self.n_clusters):
                self.logger.info(f"           cluster {k}: {np.sum(labels == k):>5} users")
            self.logger.debug('=' * 60)
            self.logger.debug('\n')

            return labels

        # If all restarts are exhausted
        self.logger.error(f"[FAILED] Could not find stable initialization after {max_restarts} attempts")
        raise RuntimeError(f"K-Means failed to initialize stably after {max_restarts} restarts")