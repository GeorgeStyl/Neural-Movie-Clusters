
import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).parent.parent
DATASETS_DIR = ROOT / 'datasets'

# Load the dataset
dataset = np.load(DATASETS_DIR / 'dataset_singles.npy', allow_pickle=True)

# Convert to dataframe
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'], )

# Find unique users in the dataset
users = df.groupby('user')
# Get their count (for comparison after the filtering process)
n_users = len(users)

# Find unique movies in the dataset
movies = df.groupby('movie')


# Determine thresholds
r = 20           # minimum number of ratings per user
m = 100          # minimum number of ratings per movie


# ==============================================
# Filtering stage 1: filter via top N movies
# ==============================================

# Count the ratings for each movie
ratings_per_movie = df.groupby('movie').size()
# Keep top movies with at least "m" ratings
valid_movies = ratings_per_movie[ratings_per_movie >= m].index
df_movies_filtered = df[df['movie'].isin(valid_movies)]

# ==============================================
# Filtering stage 2: filter via ratings per user
# ==============================================

# Get the count of ratings per user, for users that remain in the dataset after stage 1
ratings_per_user = df_movies_filtered.groupby('user').size()
# Keep users with more than "r" ratings
valid_users = ratings_per_user[ratings_per_user >= r].index
df_filtered = df_movies_filtered[df_movies_filtered['user'].isin(valid_users)]

# ==============================================
# Filtering stage 3: Repeat
# ==============================================

ratings_per_movie = df_filtered.groupby('movie').size()
valid_movies = ratings_per_movie[ratings_per_movie >= m].index
df_final = df_filtered[df_filtered['movie'].isin(valid_movies)]

ratings_per_user = df_final.groupby('user').size()
valid_users = ratings_per_user[ratings_per_user >= r].index
df_final = df_final[df_final['user'].isin(valid_users)]


# Save the final dataframe, to load (if needed) without filtering again or dealing with the "0" entries of the
# feature matrix constructed below
df_final.to_csv(DATASETS_DIR / 'df_final.csv', index=False)
print(f'\nFiltered dataframe saved to: datasets/df_final.csv')

# Drop the "date" column, which is irrelevant in the analysis from this point onwards
df_final.drop(['date'], axis=1, inplace=True)


# ==============================================
# Creation of the feature matrix
# ==============================================

# Create user-movie rating matrix, encoding the absence of a rating (user for movie) as 0
user_movie_matrix = df_final.pivot_table(
    index='user',
    columns='movie',
    values='rating',
    fill_value=0
)

# Force numeric dtype
user_movie_matrix = user_movie_matrix.astype(np.float32)

# Convert to np array and save
X = user_movie_matrix.to_numpy()
np.save(DATASETS_DIR / 'feature_matrix.npy', X)


# Print basic statistic for final feature matrix

print(f'\nFinal matrix shape: {X.shape}')
print(f'Number of users (training points): {X.shape[0]}')
print(f'Number of movies (dimensionality): {X.shape[1]}')
print(f'Ratio "training points / dimensionality": {X.shape[0]/X.shape[1]:.4f}')
print(f'Non-zero entries: {np.count_nonzero(X)}')
print(f'Sparsity: {(1 - np.count_nonzero(X)/(X.shape[0]*X.shape[1]))*100:.4f}%')
print(f'Minimum number of ratings per movie: {np.count_nonzero(X, axis=0).min()}')


# 1) Average number of ratings per user
avg_ratings_per_user = X.astype(bool).sum(axis=1).mean()
print(f'Average number of ratings per user: {avg_ratings_per_user:.2f}')

# 2) Average overlap per pair of users
# Convert to binary for overlap
X_bin = (X > 0).astype(np.int8)
n_users = X_bin.shape[0]

# Sum of intersections over all pairs (optimized)
intersections = np.dot(X_bin, X_bin.T)
# Only take upper triangle to avoid double-counting
upper_triangle_sum = intersections[np.triu_indices(n_users, k=1)].sum()
n_pairs = n_users * (n_users - 1) / 2
avg_overlap = upper_triangle_sum / n_pairs
print(f'Average number of common movies per pair of users: {avg_overlap:.2f}')

print(f'\nMatrix saved to: datasets/feature_matrix.npy')