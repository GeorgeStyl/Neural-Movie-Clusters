
from model import NeuralNet
import numpy as np
from pathlib import Path
from scipy.spatial.distance import squareform


# Paths
ROOT = Path(__file__).parent.parent
DATASETS_DIR = ROOT / 'datasets'

# Load matrices
feature_matrix = np.load(DATASETS_DIR / 'feature_matrix.npy')
distance_matrix = np.load(DATASETS_DIR / 'distance_matrix_condensed.npy')

# Convert distance matrix to squared form
distance_matrix = squareform(distance_matrix)

# Load cluster labels and get the distinct labels
labels = np.load(DATASETS_DIR / 'cluster_labels.npy')
labels_unique = np.unique(labels)

print(f'\nFound {len(labels_unique)} clusters to test')
print('-'*80)

# Store results
cluster_results = {}

# Process each cluster
for cluster_id in labels_unique:
    print(f'\nProcessing Cluster {cluster_id}')

    # Get indices of users in the cluster
    cluster_indices = np.where(labels == cluster_id)[0]
    num_cluster_users = len(cluster_indices)

    print(f'\nNumber of users in cluster: {num_cluster_users}')

    # Create train and test sets

    # Shuffle the users
    shuffled = np.random.permutation(num_cluster_users)
    # Set size proportions for sets
    train_size = int(0.8 * num_cluster_users)

    # Get train and test sets of training points
    train_users = cluster_indices[shuffled[:train_size]]
    test_users = cluster_indices[shuffled[train_size:]]

    print(f'\nTrain users: {len(train_users)}, Test users: {len(test_users)}')

    # Initialize model
    nn = NeuralNet(feature_matrix, distance_matrix)

    # Precompute input vectors for training users only
    nn.precompute_input_vectors(valid_indices=train_users)

    # Train the model
    nn.train(valid_indices=train_users)

    # Evaluate accuracy on test users
    # Evaluate accuracy on test users
    test_losses = []
    total_rated = 0
    total_exact = 0
    total_missed = 0
    total_within_one = 0
    all_preds = []  # collect all predictions across test users

    for user_ind in test_users:
        y_pred = nn.predict(
            user_ind,
            feature_matrix,
            distance_matrix,
            valid_indices=train_users
        )
        R_true = feature_matrix[user_ind, :]

        rated_mask = R_true > 0

        # MSE on rated movies only
        error = y_pred[rated_mask] - R_true[rated_mask]
        loss = np.mean(error ** 2)
        test_losses.append(loss)

        # Exact predictions (on rated movies only)
        total_exact += np.sum(y_pred[rated_mask] == R_true[rated_mask])

        # Missed: user rated it but model predicted 0
        total_missed += np.sum((R_true > 0) & (y_pred == 0))

        within_one = np.sum(np.abs(y_pred[rated_mask] - R_true[rated_mask]) <= 1)
        total_within_one += within_one

        total_rated += rated_mask.sum()

        all_preds.extend(y_pred[rated_mask].tolist())

    avg_test_loss = np.mean(test_losses)
    cluster_results[cluster_id] = np.sqrt(avg_test_loss)

    # Print metrics
    print(f'Average test MSE on cluster {cluster_id}: {avg_test_loss:.4f}')
    print(f'RMSE: {np.sqrt(avg_test_loss):.4f}')
    print(f'Exact predictions: {total_exact} / {total_rated} ({100 * total_exact / total_rated:.2f}%)')
    print(f'Missed ratings: {total_missed} / {total_rated} ({100 * total_missed / total_rated:.2f}%)')
    print(f'Within-one: {total_within_one} / {total_rated} ({100 * total_within_one / total_rated:.2f}%)')

    all_preds = np.array(all_preds)
    print(f'\nPrediction distribution (on rated movies):')
    for rating in range(0, 11):
        count = np.sum(all_preds == rating)
        print(f'  Rating {rating:2d}: {count:5d} ({100 * count / len(all_preds):.2f}%)')
    print('-' * 50)

# Summary
print('\nSummary of Results\n')
for cluster_id, rmse in cluster_results.items():
    print(f'Cluster {cluster_id}: RMSE = {rmse:.4f}')

print(f'\nOverall average RMSE across all clusters: {np.mean(list(cluster_results.values())):.4f}')