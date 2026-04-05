
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
print('=' * 80)

# Store results for summary tables
cluster_results = {
    'cluster_id': [],
    'train_rmse': [],
    'test_rmse': [],
    'train_mae': [],
    'test_mae': [],
    'train_exact_pct': [],
    'test_exact_pct': [],
    'train_within_one_pct': [],
    'test_within_one_pct': [],
    'train_missed_pct': [],
    'test_missed_pct': [],
    'num_train': [],
    'num_test': []
}

# Process each cluster
for cluster_id in labels_unique:
    print(f'\n{"=" * 80}')
    print(f'Processing Cluster {cluster_id}')
    print(f'{"=" * 80}')

    # Get indices of users in the cluster
    cluster_indices = np.where(labels == cluster_id)[0]
    num_cluster_users = len(cluster_indices)

    print(f'\nTotal users in cluster: {num_cluster_users}')

    # Create train and test sets
    shuffled = np.random.permutation(num_cluster_users)
    train_size = int(0.8 * num_cluster_users)

    train_users = cluster_indices[shuffled[:train_size]]
    test_users = cluster_indices[shuffled[train_size:]]

    print(f'Train users: {len(train_users)}, Test users: {len(test_users)}')

    # Initialize model
    nn = NeuralNet(feature_matrix, distance_matrix)

    # Precompute input vectors for training users only
    nn.precompute_input_vectors(valid_indices=train_users)

    # Train the model
    nn.train(valid_indices=train_users)

    # Evaluate on TRAINING set
    train_losses = []
    train_mae_list = []
    train_exact = 0
    train_total_rated = 0
    train_within_one = 0
    train_missed = 0

    for user_ind in train_users:
        x, a, y = nn.forward_prop(user_ind)
        R_true = feature_matrix[user_ind, :]
        rated_mask = R_true > 0

        if rated_mask.sum() == 0:
            continue

        error = y[rated_mask] - R_true[rated_mask]
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))

        train_losses.append(mse)
        train_mae_list.append(mae)

        # Exact predictions
        train_exact += np.sum(y[rated_mask] == R_true[rated_mask])
        # Within one
        train_within_one += np.sum(np.abs(error) <= 1)
        # Missed: user rated it but model predicted 0
        train_missed += np.sum((R_true[rated_mask] > 0) & (y[rated_mask] == 0))
        train_total_rated += rated_mask.sum()

    avg_train_rmse = np.sqrt(np.mean(train_losses))
    avg_train_mae = np.mean(train_mae_list)
    train_exact_pct = 100 * train_exact / train_total_rated if train_total_rated > 0 else 0
    train_within_one_pct = 100 * train_within_one / train_total_rated if train_total_rated > 0 else 0
    train_missed_pct = 100 * train_missed / train_total_rated if train_total_rated > 0 else 0

    # Evaluate on TEST set
    test_losses = []
    test_mae_list = []
    test_exact = 0
    test_total_rated = 0
    test_within_one = 0
    test_missed = 0

    for user_ind in test_users:
        y_pred = nn.predict(
            user_ind,
            feature_matrix,
            distance_matrix,
            valid_indices=train_users
        )
        R_true = feature_matrix[user_ind, :]
        rated_mask = R_true > 0

        if rated_mask.sum() == 0:
            continue

        error = y_pred[rated_mask] - R_true[rated_mask]
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))

        test_losses.append(mse)
        test_mae_list.append(mae)

        test_exact += np.sum(y_pred[rated_mask] == R_true[rated_mask])
        test_within_one += np.sum(np.abs(error) <= 1)
        test_missed += np.sum((R_true[rated_mask] > 0) & (y_pred[rated_mask] == 0))
        test_total_rated += rated_mask.sum()

    avg_test_rmse = np.sqrt(np.mean(test_losses))
    avg_test_mae = np.mean(test_mae_list)
    test_exact_pct = 100 * test_exact / test_total_rated if test_total_rated > 0 else 0
    test_within_one_pct = 100 * test_within_one / test_total_rated if test_total_rated > 0 else 0
    test_missed_pct = 100 * test_missed / test_total_rated if test_total_rated > 0 else 0

    # Store results
    cluster_results['cluster_id'].append(cluster_id)
    cluster_results['train_rmse'].append(avg_train_rmse)
    cluster_results['test_rmse'].append(avg_test_rmse)
    cluster_results['train_mae'].append(avg_train_mae)
    cluster_results['test_mae'].append(avg_test_mae)
    cluster_results['train_exact_pct'].append(train_exact_pct)
    cluster_results['test_exact_pct'].append(test_exact_pct)
    cluster_results['train_within_one_pct'].append(train_within_one_pct)
    cluster_results['test_within_one_pct'].append(test_within_one_pct)
    cluster_results['train_missed_pct'].append(train_missed_pct)
    cluster_results['test_missed_pct'].append(test_missed_pct)
    cluster_results['num_train'].append(len(train_users))
    cluster_results['num_test'].append(len(test_users))

    # Print detailed results for this cluster
    print(f'\n{"-" * 80}')
    print(f'RESULTS FOR CLUSTER {cluster_id}')
    print(f'{"-" * 80}')

    print(f'\nTRAINING SET PERFORMANCE:')
    print(f'  RMSE: {avg_train_rmse:.4f}')
    print(f'  MAE:  {avg_train_mae:.4f}')
    print(f'  Exact predictions: {train_exact}/{train_total_rated} ({train_exact_pct:.2f}%)')
    print(f'  Within-one accuracy: {train_within_one}/{train_total_rated} ({train_within_one_pct:.2f}%)')
    print(f'  Missed ratings: {train_missed}/{train_total_rated} ({train_missed_pct:.2f}%)')

    print(f'\nTEST SET PERFORMANCE:')
    print(f'  RMSE: {avg_test_rmse:.4f}')
    print(f'  MAE:  {avg_test_mae:.4f}')
    print(f'  Exact predictions: {test_exact}/{test_total_rated} ({test_exact_pct:.2f}%)')
    print(f'  Within-one accuracy: {test_within_one}/{test_total_rated} ({test_within_one_pct:.2f}%)')
    print(f'  Missed ratings: {test_missed}/{test_total_rated} ({test_missed_pct:.2f}%)')
    print(f'{"-" * 80}')

# Print summary tables
print(f'\n{"=" * 80}')
print('SUMMARY TABLE: TRAINING PERFORMANCE BY CLUSTER')
print(f'{"=" * 80}')
print(f'{"Cluster":<10} {"Users":<8} {"RMSE":<10} {"MAE":<10} {"Exact %":<10} {"Within 1 %":<10} {"Missed %":<10}')
print(f'{"-" * 80}')
for i in range(len(cluster_results['cluster_id'])):
    print(f"{cluster_results['cluster_id'][i]:<10} "
          f"{cluster_results['num_train'][i]:<8} "
          f"{cluster_results['train_rmse'][i]:<10.4f} "
          f"{cluster_results['train_mae'][i]:<10.4f} "
          f"{cluster_results['train_exact_pct'][i]:<10.2f} "
          f"{cluster_results['train_within_one_pct'][i]:<10.2f} "
          f"{cluster_results['train_missed_pct'][i]:<10.2f}")

print(f'\n{"=" * 80}')
print('SUMMARY TABLE: TEST PERFORMANCE BY CLUSTER')
print(f'{"=" * 80}')
print(f'{"Cluster":<10} {"Users":<8} {"RMSE":<10} {"MAE":<10} {"Exact %":<10} {"Within 1 %":<10} {"Missed %":<10}')
print(f'{"-" * 80}')
for i in range(len(cluster_results['cluster_id'])):
    print(f"{cluster_results['cluster_id'][i]:<10} "
          f"{cluster_results['num_test'][i]:<8} "
          f"{cluster_results['test_rmse'][i]:<10.4f} "
          f"{cluster_results['test_mae'][i]:<10.4f} "
          f"{cluster_results['test_exact_pct'][i]:<10.2f} "
          f"{cluster_results['test_within_one_pct'][i]:<10.2f} "
          f"{cluster_results['test_missed_pct'][i]:<10.2f}")

# Overall averages
print(f'\n{"=" * 80}')
print('OVERALL AVERAGES ACROSS ALL CLUSTERS')
print(f'{"=" * 80}')
print(f'Training RMSE: {np.mean(cluster_results["train_rmse"]):.4f}')
print(f'Test RMSE:     {np.mean(cluster_results["test_rmse"]):.4f}')
print(f'Training MAE:  {np.mean(cluster_results["train_mae"]):.4f}')
print(f'Test MAE:      {np.mean(cluster_results["test_mae"]):.4f}')
print(f'Training Exact %:  {np.mean(cluster_results["train_exact_pct"]):.2f}%')
print(f'Test Exact %:      {np.mean(cluster_results["test_exact_pct"]):.2f}%')
print(f'Training Within 1 %:  {np.mean(cluster_results["train_within_one_pct"]):.2f}%')
print(f'Test Within 1 %:      {np.mean(cluster_results["test_within_one_pct"]):.2f}%')
print(f'Training Missed %:    {np.mean(cluster_results["train_missed_pct"]):.2f}%')
print(f'Test Missed %:        {np.mean(cluster_results["test_missed_pct"]):.2f}%')