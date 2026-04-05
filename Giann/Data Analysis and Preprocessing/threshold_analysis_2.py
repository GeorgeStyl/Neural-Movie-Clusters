# Iterative filtering: movies with at least m ratings and users with at least n ratings
# Apply: movie -> user -> movie -> user  --- in order to avoid “weak” rows or columns

import numpy as np
import pandas as pd

# Load dataset
dataset = np.load(r'../datasets/dataset_singles.npy', allow_pickle=True)
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'])

# Drop the "date" column, irrelevant for this analysis
df.drop(['date'], axis=1, inplace=True)

# Thresholds to test
n_values = list(range(8, 15)) + [20, 30, 50, 75, 100]  # min ratings per user
m_values = [5, 10, 20, 30, 50, 75, 100]                # min ratings per movie

# Print header
print("n\tm\tUsers\tMovies\tSparsity\tMatrix Size\tNon-zero Entries\tMinRatingsPerMovie\tAvgRatingsPerMovie")
print("-" * 120)

for m in m_values:
    for n in n_values:
        df_iter = df.copy()

        # --- First movie filter ---
        ratings_per_movie = df_iter.groupby('movie').size()
        valid_movies = ratings_per_movie[ratings_per_movie >= m].index
        df_iter = df_iter[df_iter['movie'].isin(valid_movies)]

        # --- First user filter ---
        ratings_per_user = df_iter.groupby('user').size()
        valid_users = ratings_per_user[ratings_per_user >= n].index
        df_iter = df_iter[df_iter['user'].isin(valid_users)]

        # --- Second movie filter ---
        ratings_per_movie = df_iter.groupby('movie').size()
        valid_movies = ratings_per_movie[ratings_per_movie >= m].index
        df_iter = df_iter[df_iter['movie'].isin(valid_movies)]

        # --- Second user filter ---
        ratings_per_user = df_iter.groupby('user').size()
        valid_users = ratings_per_user[ratings_per_user >= n].index
        df_final = df_iter[df_iter['user'].isin(valid_users)]

        # Compute stats
        n_users = df_final['user'].nunique()
        n_movies = df_final['movie'].nunique()
        matrix_size = n_users * n_movies
        non_zero = len(df_final)
        sparsity = (1 - (non_zero / matrix_size)) * 100 if matrix_size > 0 else 0

        min_ratings_per_movie = df_final.groupby('movie').size().min() if n_movies > 0 else 0
        avg_ratings_per_movie = df_final.groupby('movie').size().mean() if n_movies > 0 else 0

        # Print results
        print(f"{n}\t{m}\t{n_users}\t{n_movies}\t{sparsity:.4f}%\t{matrix_size}\t{non_zero}\t{min_ratings_per_movie}\t{avg_ratings_per_movie:.2f}")
    print('-' * 80)
    print()

print("-" * 100)