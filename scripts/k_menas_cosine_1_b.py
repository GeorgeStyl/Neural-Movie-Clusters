import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import time

total_start = time.perf_counter()  # ← add this line

# --- Live Timer Utility ---
def live_timer(label="Running"):
    start = time.perf_counter()
    while not live_timer.stop:
        elapsed = time.perf_counter() - start
        print(f"\r⏱  {label}: {elapsed:.1f}s", end="", flush=True)
        time.sleep(0.1)
    elapsed = time.perf_counter() - start
    print(f"\r✅ {label}: {elapsed:.2f}s")

def run_timed(label, fn, *args, **kwargs):
    live_timer.stop = False
    t = threading.Thread(target=live_timer, args=(label,))
    t.start()
    result = fn(*args, **kwargs)
    live_timer.stop = True
    t.join()
    return result


# --- Custom Cosine Distance ---
def calculate_sparse_custom_cosine(X, f_u, f_v):
    weights = np.sqrt(f_u * f_v)
    X_weighted = X.multiply(weights)
    numerator = X_weighted @ X_weighted.T
    row_norms = np.sqrt(np.array(X_weighted.power(2).sum(axis=1)).flatten())
    row_norms[row_norms == 0] = 1e-10
    denominator = np.outer(row_norms, row_norms)
    similarity = numerator.toarray() / denominator
    dist_matrix = 1 - similarity
    np.fill_diagonal(dist_matrix, 0)
    return np.clip(dist_matrix, 0, 1)


# --- 1. Load Data ---
FILE_PATH = '../data/dataset_clean.npy'
print(f"Loading data from {FILE_PATH}...")
raw_data = run_timed("Loading .npy file", np.load, FILE_PATH, allow_pickle=True)

# --- 2. Build DataFrame ---
df = pd.DataFrame(raw_data, columns=['userId', 'movieId', 'rating', 'timestamp'])
df = df[['userId', 'movieId', 'rating']].astype({'userId': int, 'movieId': int, 'rating': float})
print(f"Ratings log: {df.shape[0]:,} entries | "
      f"{df['userId'].nunique():,} users | "
      f"{df['movieId'].nunique():,} movies")

# --- 3. Filter ---
MIN_RATINGS_PER_USER  = 50
MIN_RATINGS_PER_MOVIE = 20
user_counts  = df['userId'].value_counts()
movie_counts = df['movieId'].value_counts()
df = df[df['userId'].isin(user_counts[user_counts   >= MIN_RATINGS_PER_USER].index)]
df = df[df['movieId'].isin(movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index)]
print(f"After filtering: {df['userId'].nunique():,} users | {df['movieId'].nunique():,} movies")

# --- 4. Pivot ---
def build_pivot(df):
    return df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

pivot = run_timed("Building User x Movie matrix", build_pivot, df)
X_sparse = csr_matrix(pivot.values.astype(np.float64))
print(f"Matrix shape: {X_sparse.shape} (Users x Movies)")

# --- 5. Weights ---
n_features = X_sparse.shape[1]
f_u = np.ones(n_features)
f_v = np.ones(n_features)

# --- 6. Distance Matrix (computed once, reused for all cluster counts) ---
dist_mat = run_timed(
    "Calculating distance matrix",
    calculate_sparse_custom_cosine, X_sparse, f_u, f_v
)

# --- 7. KMeans + Plot for k = 4, 5, 6, 7 ---
cluster_counts = [4, 5, 6, 7]
palettes = ['viridis', 'plasma', 'tab10', 'Set2']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Movie Clusters — Optimized Sparse Cosine', fontsize=16, fontweight='bold', y=1.01)
axes = axes.flatten()

for ax, k, palette in zip(axes, cluster_counts, palettes):
    clusters = run_timed(f"KMeans k={k}", KMeans(
        n_clusters=k, random_state=42, n_init='auto'
    ).fit_predict, dist_mat)

    sns.scatterplot(
        x=dist_mat[:, 0],
        y=dist_mat[:, 1],
        hue=clusters,
        palette=palette,
        ax=ax,
        legend='full',
        s=15,           # smaller dots — looks cleaner with many points
        alpha=0.7
    )
    ax.set_title(f'k = {k}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Distance Dimension 1')
    ax.set_ylabel('Distance Dimension 2')
    ax.legend(title='Cluster', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig('../../plots/movie_clustering_results.png', bbox_inches='tight', dpi=150)
plt.show()

total_elapsed = time.perf_counter() - total_start
print(f"\nDone in {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
