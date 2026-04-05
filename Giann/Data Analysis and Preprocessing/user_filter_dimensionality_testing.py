# In this file, we aim to test the dimensionality our problem has if we choose to explicitly remove only users
# (as the project description says), and hwo much dimensionality reduction is succeeded this way

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# Load dataset
dataset = np.load(r'../datasets/dataset_singles.npy', allow_pickle=True)
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'])

# Drop the "date" column, which is irrelevant in this analysis
df.drop(['date'], axis=1, inplace=True)

# Store results for plotting
n_values = list(range(7, 31)) + [50, 60, 70, 80, 90, 100]
users_counts = []
movies_counts = []

# Count ratings per user
ratings_per_user = df.groupby('user').size()

print("Threshold Analysis: Minimum ratings per user")
print("-" * 50)
print(f"{'n':<5} {'Users':<10} {'Movies':<10} {'Avg/User':<10}")
print("-" * 50)

# Iterate through different n values
for n in n_values:

    # Keep users with at least n ratings
    valid_users = ratings_per_user[ratings_per_user >= n].index
    df_filtered = df[df['user'].isin(valid_users)]

    # Get counts
    num_users = len(df_filtered['user'].unique())
    avg_ratings = df_filtered.shape[0] / num_users
    num_movies = df_filtered['movie'].nunique()

    # Store for histogram
    users_counts.append(num_users)
    movies_counts.append(num_movies)

    # Print results
    print(f"{n:<5} {num_users:<10} {num_movies:<10} {avg_ratings:<10.2f}")

print("-" * 50)

# Create histogram
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Users histogram
ax1.bar(n_values, users_counts, color='skyblue', edgecolor='navy', alpha=0.7)
ax1.set_xlabel('Minimum ratings per user (n)')
ax1.set_ylabel('Number of users')
ax1.set_title('Users Remaining vs Minimum Rating Threshold')
ax1.grid(True, alpha=0.3)
for i, v in enumerate(users_counts):
    ax1.text(n_values[i], v + 5, str(v), ha='center', va='bottom', fontsize=8)

# Movies histogram
ax2.bar(n_values, movies_counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
ax2.set_xlabel('Minimum ratings per user (n)')
ax2.set_ylabel('Number of movies')
ax2.set_title('Movies Remaining vs Minimum Rating Threshold')
ax2.grid(True, alpha=0.3)
for i, v in enumerate(movies_counts):
    ax2.text(n_values[i], v + 5, str(v), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


# === Sample-based Average Overlap Analysis (main-project-part-oriented) ===

# Number of user pairs to sample for overlap calculation
num_samples = 5000

print("\nSampled Average Overlap Between Users")
print("-" * 50)
print(f"{'n':<5} {'Users':<10} {'Movies':<10} {'Avg/User':<10} {'AvgOverlap':<12}")
print("-" * 50)

for n in n_values:
    # Keep users with at least n ratings
    valid_users = ratings_per_user[ratings_per_user >= n].index
    df_filtered = df[df['user'].isin(valid_users)]

    num_users = len(df_filtered['user'].unique())
    num_movies = df_filtered['movie'].nunique()
    avg_ratings = df_filtered.shape[0] / num_users

    # Create a dictionary: user -> set of rated movies
    user_movies = df_filtered.groupby('user')['movie'].apply(set).to_dict()
    user_list = list(user_movies.keys())

    # Sample pairs (without replacement) for computing overlap
    sampled_overlaps = []
    for _ in range(min(num_samples, num_users*(num_users-1)//2)):
        u1, u2 = random.sample(user_list, 2)
        intersection_size = len(user_movies[u1] & user_movies[u2])
        sampled_overlaps.append(intersection_size)

    avg_overlap = np.mean(sampled_overlaps) if sampled_overlaps else 0.0

    print(f"{n:<5} {num_users:<10} {num_movies:<10} {avg_ratings:<10.2f} {avg_overlap:<12.4f}")

print("-" * 50)


# The results of this analysis confirm that, in order to apply a meaningful dimensionality reduction, there are
# two options:
# 1) Increase the minimum number of ratings for a user, which a) doesn't guarantee a fix to the problem and b) removes
#    much information
# 2) Reduce the dimensionality in a second step, by explicitly filtering out movies.