"""
1) b)
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns



def my_cosine_metric(r_u, r_v, m, f_u, f_v):
    """Calculates custom cosine distance between two vectors."""
    # Slicing the relevant features
    r_u_m, r_v_m = r_u[m], r_v[m]
    f_u_m, f_v_m = f_u[m], f_v[m]

    def numerator():
        return np.sum(r_u_m * r_v_m * f_u_m * f_v_m)

    def denominator():
        term_u = np.sqrt(np.sum((r_u_m ** 2) * f_u_m * f_v_m))
        term_v = np.sqrt(np.sum((r_v_m ** 2) * f_u_m * f_v_m))
        return term_u * term_v

    num_sum = numerator()
    den_sum = denominator()

    if den_sum == 0:
        return 1.0

    return 1 - (num_sum / den_sum)


def calculate_custom_distance_matrix(data_tensor, m_indices, f_u, f_v):
    """Computes the full NxN distance matrix for the dataset."""
    n_samples = data_tensor.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    print("Calculating custom distance matrix for {n_samples} samples...")

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = my_cosine_metric(data_tensor[i], data_tensor[j], m_indices, f_u, f_v)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


# --- Execution ---
# Assuming 'X' is movie feature matrix
# 1. Load your data (Assuming it's a 2D array of movie features)
# For this example, let's assume X is loaded from RAW_DATA_NPY
X = np.load(RAW_DATA_NPY)

# 2. Setup your parameters (m should be indices of features to consider)
m = np.arange(X.shape[1])
f_u = np.ones(X.shape[1]) # Example weight vectors
f_v = np.ones(X.shape[1])

# 3. Calculate Distance Matrix
dist_mat = calculate_custom_distance_matrix(X, m, f_u, f_v)

# 4. Perform Clustering
# We use KMeans on the distances to group movies with similar "profiles"
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(dist_mat)

print("Clustering complete. {len(np.unique(clusters))} clusters identified.")

# 5. Visualization & Saving
plt.figure(figsize=(10, 7))
sns.scatterplot(x=dist_mat[:, 0], y=dist_mat[:, 1], hue=clusters, palette='deep')
plt.title(f'Movie Clusters (k={n_clusters}) - Custom Metric')

# Save to your plots directory
SAVE_PATH = os.path.join(PLOTS_DIR, 'movie_clustering_results.png')
plt.savefig(SAVE_PATH)
plt.show()

