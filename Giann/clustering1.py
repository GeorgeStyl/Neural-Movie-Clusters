
import numpy as np


class KMeansEuclidean:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = None


    @staticmethod
    def dist_user_centroid(user_vec, centroid):

        # Find the coordinates in which both user and centroid have positive values
        both = (user_vec > 0) & (centroid > 0)

        # If there are none such, set distance between them to inf
        if not np.any(both):
            return np.inf

        # Otherwise, calculate the distance as proposed in project description and return
        diff = user_vec[both] - centroid[both]

        return np.sqrt(np.sum(diff ** 2))


    @staticmethod
    def update_centroid(cluster_points):

        # Initialize the new centroid to be a zero-vector of length equal to the dimension of the points that belong in it
        c = np.zeros(cluster_points.shape[1])

        # For each of these dimensions
        for i in range(cluster_points.shape[1]):

            # Get the rates by users that actually rated this "dimension" (movie), if there are any
            rated = cluster_points[:, i] > 0

            # Compute the mean of these rates and place it in this coordinate
            if np.any(rated):
                c[i] = np.mean(cluster_points[rated, i])

        return c


    def fit(self, X, max_iters=200):

        # Initially, pick as centroids data points (users) from our dataset
        n = self.n_clusters
        total_rows = X.shape[0]
        random_indices = np.random.choice(total_rows, size=n, replace=False)
        self.centroids = X[random_indices, :]

        # Start looping
        for iteration in range(max_iters):

            print(f'Iteration {iteration+1}')

            # Step 1: Assignment of users to clusters

            # Initialize an empty map-container. Here, index "i" will have as value the label of the cluster in which
            # each user belongs to, at the start of current iteration
            y = []
            for user in X:
                distances = [KMeansEuclidean.dist_user_centroid(user, centroid) for centroid in self.centroids]
                y.append(np.argmin(distances))
            y = np.array(y)

            # Step 2: Updating centroids

            # Create a container to store the users that belong in each cluster
            cluster_indices = []
            for i in range(self.n_clusters):
                # Assign to this container the users that are labeled to belong in cluster "i"
                cluster_indices.append(np.where(y == i)[0])

            # Update the centroids
            cluster_centers = []
            # Get the label of each cluster, along with the users that are labeled to belong in it
            for i, indices in enumerate(cluster_indices):
                # If a cluster contains no users
                if len(indices) == 0:
                    # Simply add it again to the list of centroids
                    cluster_centers.append(self.centroids[i])
                else:
                    # Otherwise get the users that belong in it
                    cluster_points = X[indices, :]
                    # Compute the new centroid that these users "create"
                    center = KMeansEuclidean.update_centroid(cluster_points)
                    # And add this centroid to our list of updated centroids
                    cluster_centers.append(center)
            cluster_centers = np.array(cluster_centers)

            # Step 3: Exit condition

            if np.max(np.abs(cluster_centers - self.centroids)) < 1e-6:
                break
            else:
                self.centroids = cluster_centers

        return y






