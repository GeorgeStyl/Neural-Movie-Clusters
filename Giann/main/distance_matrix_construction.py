
# DistanceMatrixConstructor: Builds an NxN distance matrix from an NxM feature matrix, based on the specified dist
# metric (default=Jaccard)

# Input: NxM feature matrix (rows = training points, cols = features)
# Output: Condensed upper-triangular distance matrix (size N*(N-1)//2)

# Custom distance functions must follow this structure:
#   def custom_distance(user_sets: List[set], u1: int, u2: int) -> float:
#       - user_sets: list of sets where user_sets[i] contains the feature indices for user i
#       - u1, u2: indices of the two users to compare
#       - returns: distance value (0=identical, 1=no overlap)

import numpy as np
from scipy.sparse import csr_matrix


def jaccard_distance(user_sets, u1: int, u2: int) -> float:
    """
    Jaccard distance between two users
    :param user_sets: list of sets containing movie indices per user
    :param u1: index of first user
    :param u2: index of second user
    :return: distance(u1, u2)
    """
    movies1 = user_sets[u1]
    movies2 = user_sets[u2]

    # Based on our filtering this check won't ever be successful in the project environment, but for completeness
    if not (movies1 and movies2):
        return 1.0

    denom = len(movies1.union(movies2))
    numer = len(movies1.intersection(movies2))

    return 1.0 - (numer / denom)


class DistanceMatrixConstructor:

    def __init__(self, feature_matrix, distance_metric=None):
        self.X = None  # Feature matrix (sparse)
        self.user_sets = None  # Precomputed sets of movies per user
        self.distance_matrix = None  # Condensed distance matrix
        self.distance_metric = distance_metric if distance_metric is not None else jaccard_distance
        self.load_features(feature_matrix)


    def load_features(self, X):

        # Convert to binary
        X = (X > 0).astype(np.int8)

        # Convert to sparse (row-wise)
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)

        self.X = X
        print('\nFeature matrix loaded and brought to suitable form.')
        self._precompute_user_sets()


    def _precompute_user_sets(self):

        self.user_sets = [set(self.X.indices[self.X.indptr[i]:self.X.indptr[i+1]]) for i in range(self.X.shape[0])]
        print('\nUser sets precomputed.')


    def compute_distance_matrix(self):
        """
        Compute NxN distance matrix
        :return: Upper triangular, condensed distance matrix
        """
        # Get the number of users to determine matrix size
        n_users = self.X.shape[0]
        size = n_users * (n_users - 1) // 2

        # Pre-allocate condensed distance array to improve performance
        self.distance_matrix = np.zeros((size,), dtype=np.float32)

        print(f'\nTotal entries to fill: {size} (only upper triangle part of the matrix will be calculated).\n')
        k = 0
        for i in range(n_users):
            for j in range(i + 1, n_users):
                # Compute distance between user i and user j
                self.distance_matrix[k] = self.distance_metric(self.user_sets, i, j)
                k += 1
                # Progress indicator
                if k % 1_000_000 == 0:
                    print(f'{k} entries filled.')

        print('\nDistance matrix computed.')


    def save_distance_matrix(self, path):

        np.save(path, self.distance_matrix)
        print(f'\n(Upper triangular) Distance matrix created and saved to: {path}')


    def distance_statistics(self):

        print(f'\nDistance matrix shape: {self.distance_matrix.shape}')
        print(f'Min distance: {self.distance_matrix.min():.4f}')
        print(f'Max distance: {self.distance_matrix.max():.4f}')
        print(f'Average distance: {self.distance_matrix.mean():.4f}')
        print(f'Median distance: {np.median(self.distance_matrix):.4f}')

        percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
        print('\nPercentiles:')
        for p in percentiles:
            print(f'{p}th percentile: {np.percentile(self.distance_matrix, p):.4f}')


if __name__ == "__main__":

    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    DATASETS_DIR = ROOT / 'datasets'

    # Load feature matrix
    X = np.load(DATASETS_DIR / 'feature_matrix.npy')

    # Create distance matrix constructor with the feature matrix and default Jaccard metric
    calc = DistanceMatrixConstructor(X)
    calc.compute_distance_matrix()
    calc.save_distance_matrix(DATASETS_DIR / 'distance_matrix_condensed.npy')
    calc.distance_statistics()
