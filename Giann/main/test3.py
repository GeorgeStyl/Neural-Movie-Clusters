
import numpy as np
from pathlib import Path


# Paths
ROOT = Path(__file__).parent.parent
DATASETS_DIR = ROOT / 'datasets'

# Load data
feature_matrix = np.load(DATASETS_DIR / 'feature_matrix.npy')
labels = np.load(DATASETS_DIR / 'cluster_labels.npy')
labels_unique = np.unique(labels)

print(f'Found {len(labels_unique)} clusters')
print('=' * 60)

# Store results
normal_mae_test = []
uniform_mae_test = []

# Process each cluster
for cluster_id in labels_unique:
    cluster_indices = np.where(labels == cluster_id)[0]
    num_cluster_users = len(cluster_indices)

    # Same split as your original script
    # np.random.seed(42)
    shuffled = np.random.permutation(num_cluster_users)
    train_size = int(0.8 * num_cluster_users)

    train_users = cluster_indices[shuffled[:train_size]]
    test_users = cluster_indices[shuffled[train_size:]]

    # Get all TEST ratings from this cluster
    all_test_ratings = []
    for user_idx in test_users:
        user_ratings = feature_matrix[user_idx, :]
        ratings = user_ratings[user_ratings > 0]
        all_test_ratings.extend(ratings)

    all_test_ratings = np.array(all_test_ratings)

    if len(all_test_ratings) == 0:
        print(f'Cluster {cluster_id}: No test ratings found')
        continue

    # Generate random predictions for test set
    np.random.seed(42)  # Same seed for reproducibility

    # Normal distribution (μ=5, σ=5, clip to 1-10, round)
    normal_preds = np.random.normal(5, 5, len(all_test_ratings))
    normal_preds = np.clip(normal_preds, 1, 10)
    normal_preds = np.round(normal_preds)

    # Uniform discrete (1-10)
    uniform_preds = np.random.randint(1, 11, len(all_test_ratings))

    # Calculate MAE on test set only
    normal_mae_val = np.mean(np.abs(normal_preds - all_test_ratings))
    uniform_mae_val = np.mean(np.abs(uniform_preds - all_test_ratings))

    normal_mae_test.append(normal_mae_val)
    uniform_mae_test.append(uniform_mae_val)

    print(f'Cluster {cluster_id}: {len(all_test_ratings)} test ratings')
    print(f'  Normal baseline MAE:  {normal_mae_val:.4f}')
    print(f'  Uniform baseline MAE: {uniform_mae_val:.4f}')
    print()

# Overall averages
print('=' * 60)
print('TEST SET AVERAGES (for direct comparison with your model)')
print('=' * 60)
print(f'Normal baseline MAE:  {np.mean(normal_mae_test):.4f} ± {np.std(normal_mae_test):.4f}')
print(f'Uniform baseline MAE: {np.mean(uniform_mae_test):.4f} ± {np.std(uniform_mae_test):.4f}')