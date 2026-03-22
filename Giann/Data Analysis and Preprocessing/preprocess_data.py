
import numpy as np
import pandas as pd


# Load the dataset
dataset = np.load(r'../datasets/dataset_singles.npy', allow_pickle=True)

# Convert to dataframe
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'], )

# Find unique users in the dataset
users = df.groupby('user')
# Get their count (for comparison after the filtering process)
n_users = len(users)

# Find unique movies in the dataset
movies = df.groupby('movie')


# Determine thresholds
r = 8           # minimum number of ratings per user
N = 10000       # top N movies by ratings count


# ==============================================
# Filtering stage 1: filter via top N movies
# ==============================================

# Count the ratings for each movie
ratings_per_movie = movies.size()
# Keep top N movies
valid_movies = ratings_per_movie.nlargest(N).index
df_movies_filtered = df[df['movie'].isin(valid_movies)]

# ==============================================
# Filtering stage 2: filter via ratings per user
# ==============================================

# Get the count of ratings per user, for users that remain in the dataset after stage 1
ratings_per_user = df_movies_filtered.groupby('user').size()
# Keep users with more than "r" ratings
valid_users = ratings_per_user[ratings_per_user >= r].index
df_final = df_movies_filtered[df_movies_filtered['user'].isin(valid_users)]

# Save the final dataframe, to load (if needed) without filtering again or dealing with the "0" entries of the
# feature matrix constructed below
df_final.to_csv('../datasets/df_final.csv', index=False)
print(f'Filtered dataframe saved to: ../datasets/df_final.csv')

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
np.save('../datasets/matrix_final.npy', X)


# Print basic statistic for final feature matrix

print(f'Final matrix shape: {X.shape}')
print(f'Number of users (training patterns): {X.shape[0]}')
print(f'Number of movies (dimensionality): {X.shape[1]}')
print(f'Ratio "training points / dimensionality": {X.shape[0]/X.shape[1]:.4f}')
print(f'Non-zero entries: {np.count_nonzero(X)}')
print(f'Sparsity: {(1 - np.count_nonzero(X)/(X.shape[0]*X.shape[1]))*100:.4f}%')
print(f'Matrix saved to: ../datasets/matrix_final.npy')