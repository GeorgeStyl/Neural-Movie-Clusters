import numpy as np
from scipy.sparse import csr_matrix
from assignment.second.KMeans_clustering import CustomKMeans
from assignment.second.clustering_validity_measures import CustomDaviesBouldin
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
import logging
import logging.handlers


# Specify directories paths
ROOT = Path(__file__).parent.parent
DATASETS_DIR = ROOT / 'datasets'
PLOTS_DIR = ROOT / 'plots/clustering'
LOGS_DIR = ROOT / 'logs'


def logger_listener(log_file, queue):
    """Single process that owns the log file and writes all messages from the queue."""
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    while True:
        record = queue.get()
        if record is None:  # Shutdown signal
            break
        file_handler.emit(record)
    file_handler.close()


def run_cluster(n_clusters, metric, X, n_iterations, log_queue):
    """Worker function: runs n_iterations of KMeans for a given n_clusters."""

    # Each worker sets up a logger that forwards to the shared queue
    logger = logging.getLogger('CustomKMeans')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(queue_handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)

    cluster_db_scores = []

    for run in range(n_iterations):
        kmeans = CustomKMeans(n_clusters=n_clusters, metric=metric)
        labels = kmeans.fit(X, max_iters=200, tol=1e-6)

        db_metric = CustomDaviesBouldin(n_clusters=n_clusters, metric=metric)
        db_score = db_metric.score(X, labels, kmeans.centroids)
        cluster_db_scores.append(db_score)

    avg_db_score = float(np.mean(cluster_db_scores))
    return n_clusters, avg_db_score


if __name__ == '__main__':

    # ---------------------------
    # Load dataset
    # ---------------------------
    X = np.load(DATASETS_DIR / 'feature_matrix.npy', allow_pickle=True)
    print('Dataset loaded:', X.shape)

    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    # ---------------------------
    # Parameters
    # ---------------------------
    metric = 'euclidean'  # or 'cosine'
    min_clusters = 3
    max_clusters = 50
    n_iterations = 3
    n_workers = mp.cpu_count() - 1  # Leave one core free for the OS

    # ---------------------------
    # Set up log file and listener
    # ---------------------------
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = LOGS_DIR / f'clustering_{metric}_run_{timestamp}.log'

    # ---------------------------
    # Parallel computation
    # ---------------------------
    cluster_range = range(min_clusters, max_clusters + 1)

    with mp.Manager() as manager:
        log_queue = manager.Queue()

        listener = mp.Process(target=logger_listener, args=(log_file, log_queue))
        listener.start()

        with mp.Pool(processes=n_workers) as pool:
            results = pool.starmap(
                run_cluster,
                [(n_clusters, metric, X, n_iterations, log_queue) for n_clusters in cluster_range]
            )

        # Shut down the listener
        log_queue.put(None)
        listener.join()

    # Sort results by n_clusters (pool doesn't guarantee order)
    results.sort(key=lambda x: x[0])
    db_scores = [score for _, score in results]

    # ---------------------------
    # Plot results
    # ---------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(cluster_range, db_scores, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Davies-Bouldin Index', fontsize=12)
    plt.title(f'Davies-Bouldin Index vs Number of Clusters (metric = {metric})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(cluster_range[::5])
    plt.tight_layout()
    plt.show()

    # Print optimal k
    optimal_k = cluster_range[np.argmin(db_scores)]
    print(f"\n{'=' * 50}")
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Lowest Davies-Bouldin Index: {min(db_scores):.4f}")
    print(f"{'=' * 50}")