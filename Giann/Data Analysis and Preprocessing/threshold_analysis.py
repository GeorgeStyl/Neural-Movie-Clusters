# In this file, we test multiple thresholds for R_min (minimum number of ratings required for a user to pass the
# filtering stage), as well as for movies (number of movies to keep), to achieve meaningful dimensionality reduction,
# without suffering a great information loss

import numpy as np
import pandas as pd
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Load dataset
dataset = np.load(r'../datasets/dataset_singles.npy', allow_pickle=True)
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'])

# Drop the "date" column, irrelevant for this analysis
df.drop(['date'], axis=1, inplace=True)

# Thresholds to test
n_values = list(range(8, 15)) + [20, 30, 50, 75, 100]  # min ratings per user
thresholds = [1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000]  # top N movies

# Print header
print("n\tThreshold\tUsers\tMovies\tSparsity\tMatrix Size\tNon-zero Entries\tMinRatingsPerMovie\tAvgRatingsPerMovie")
print("-" * 100)

# --- Step 1: Filter by top N movies globally ---
ratings_per_movie_global = df.groupby('movie').size()

# Analyze the entries that pass the filtering stage for each threshold pair
for threshold in thresholds:

    if len(ratings_per_movie_global) > threshold:
        valid_movies_global = ratings_per_movie_global.nlargest(threshold).index
    else:
        valid_movies_global = ratings_per_movie_global.index

    df_movies_filtered = df[df['movie'].isin(valid_movies_global)]

    # Precompute ratings per user (once per threshold)
    ratings_per_user = df_movies_filtered.groupby('user').size()

    # Precompute ratings per movie (before user filtering)
    ratings_per_movie_filtered = df_movies_filtered.groupby('movie').size()

    for n in n_values:
        # --- Step 2: Filter users based on ratings within these movies ---
        valid_users = ratings_per_user[ratings_per_user >= n].index
        df_final = df_movies_filtered[df_movies_filtered['user'].isin(valid_users)]

        # Users and movies counts
        n_users = len(valid_users)
        n_movies = df_final['movie'].nunique()

        # Minimum number of ratings per movie in the final dataset
        min_ratings_per_movie = df_final.groupby('movie').size().min() if n_movies > 0 else 0

        # Average number of ratings per movie
        avg_ratings_per_movie = len(df_final) if n_movies > 0 else 0

        # Average overlap per pair of users

        # Sparsity and matrix size
        matrix_size = n_users * n_movies
        non_zero = len(df_final)
        sparsity = (1 - (non_zero / matrix_size)) * 100 if matrix_size > 0 else 0

        # Determine output color
        if (n_users < n_movies) or (n_users < 5000):
            color = Fore.RED
        elif n_users >= 3 * n_movies:
            color = Fore.GREEN
        else:
            color = Fore.WHITE

        # Print results
        print(f"{color}{n}\t{threshold}\t\t{n_users}\t{n_movies}\t{sparsity:.4f}%\t{matrix_size}\t{non_zero}\t{min_ratings_per_movie}\t{avg_ratings_per_movie}{Style.RESET_ALL}")
    print('-' * 80)
    print()

print("-" * 100)