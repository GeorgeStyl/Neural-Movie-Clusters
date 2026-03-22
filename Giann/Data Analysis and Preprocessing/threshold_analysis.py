# In this file, we test multiple thresholds for R_min (minimum number of ratings required for a user to pass the
# filtering state), as well as for movies (number of movies to keep), to achieve a meaningful dimensionality reduction.

import numpy as np
import pandas as pd

# Load dataset
dataset = np.load(r'../datasets/dataset_singles.npy', allow_pickle=True)
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'])

# Drop the "date" column, which is irrelevant in this analysis
df.drop(['date'], axis=1, inplace=True)

# Thresholds to test
n_values = range(8, 15)  # 8 through 14
thresholds = [5000, 7000, 10000, 15000, 20000]

print("n\tThreshold\tUsers\tMovies\tSparsity\tMatrix Size\tNon-zero Entries")
print("-" * 80)

# Analyze the entries that pass the filtering stage for each threshold pair.
# Note: the movie filtering needs to be applied before the "ratings per user" filtering, since otherwise we may
# end up with users with less than "r" ratings in the final dataset.
for threshold in thresholds:
    # Step 1: Get top N movies globally (based on ALL ratings)
    global_ratings_per_movie = df.groupby('movie').size()

    # Keep top "threshold" movies globally
    if len(global_ratings_per_movie) > threshold:
        valid_movies_global = global_ratings_per_movie.nlargest(threshold).index
    else:
        valid_movies_global = global_ratings_per_movie.index

    # Filter dataset to only these movies
    df_movies_filtered = df[df['movie'].isin(valid_movies_global)]

    for n in n_values:  # INNER LOOP: user threshold
        # Step 2: Now filter users based on ratings within these movies
        ratings_per_user = df_movies_filtered.groupby('user').size()
        valid_users = ratings_per_user[ratings_per_user >= n].index
        df_final = df_movies_filtered[df_movies_filtered['user'].isin(valid_users)]

        # Get counts
        n_users = df_final['user'].nunique()
        n_movies = df_final['movie'].nunique()

        # Calculate matrix size and sparsity
        matrix_size = n_users * n_movies
        non_zero = len(df_final)

        if matrix_size > 0:
            sparsity = (1 - (non_zero / matrix_size)) * 100
        else:
            sparsity = 0

        print(f"{n}\t{threshold}\t\t{n_users}\t{n_movies}\t{sparsity:.4f}%\t{matrix_size}\t{non_zero}")

print("-" * 80)