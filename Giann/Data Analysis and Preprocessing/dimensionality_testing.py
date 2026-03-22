# In this file, we aim to test the dimensionality our problem has if we choose to explicitly remove only users
# (as the project description says), and hwo much dimensionality reduction is succeeded this way

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = np.load(r'../datasets/dataset_singles.npy', allow_pickle=True)
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'])

# Drop the "date" column, which is irrelevant in this analysis
df.drop(['date'], axis=1, inplace=True)

# Store results for plotting
n_values = list(range(7, 30))
users_counts = []
movies_counts = []

print("Threshold Analysis: Minimum ratings per user")
print("-" * 50)
print(f"{'n':<5} {'Users':<10} {'Movies':<10}")
print("-" * 50)

# Iterate through different n values
for n in n_values:
    # Count ratings per user
    ratings_per_user = df.groupby('user').size()

    # Keep users with at least n ratings
    valid_users = ratings_per_user[ratings_per_user >= n].index
    df_filtered = df[df['user'].isin(valid_users)]

    # Get counts
    num_users = len(df_filtered['user'].unique())
    num_movies = df_filtered['movie'].nunique()

    # Store for histogram
    users_counts.append(num_users)
    movies_counts.append(num_movies)

    # Print results
    print(f"{n:<5} {num_users:<10} {num_movies:<10}")

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


# The results of this analysis confirm that, in order to apply a meaningful dimensionality reduction, there are
# two options:
# 1) Increase the minimum number of ratings for a user, which a) doesn't guarantee a fix to the problem and b) removes
#   much information
# 2) Reduce the dimensionality in a second step, by explicitly filtering out movies.


# ------------------------------------------------------------
# Extended Analysis: High thresholds with upper bound (<= 300)
# ------------------------------------------------------------

high_thresholds = [50, 70, 80, 100, 120]
max_threshold = 300

high_users_counts = []
high_movies_counts = []

print("\nExtended Threshold Analysis (High values with max = 300)")
print("-" * 60)
print(f"{'n_min':<10} {'Users':<10} {'Movies':<10}")
print("-" * 60)

for n in high_thresholds:
    # Count ratings per user
    ratings_per_user = df.groupby('user').size()

    # Apply BOTH min and max filter
    valid_users = ratings_per_user[
        (ratings_per_user >= n) & (ratings_per_user <= max_threshold)
    ].index

    df_filtered = df[df['user'].isin(valid_users)]

    # Get counts
    num_users = len(df_filtered['user'].unique())
    num_movies = df_filtered['movie'].nunique()

    # Store results
    high_users_counts.append(num_users)
    high_movies_counts.append(num_movies)

    # Print results
    print(f"{n:<10} {num_users:<10} {num_movies:<10}")

print("-" * 60)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Users plot
ax1.bar(high_thresholds, high_users_counts, color='mediumseagreen', edgecolor='darkgreen', alpha=0.7)
ax1.set_xlabel('Minimum ratings per user (n)')
ax1.set_ylabel('Number of users')
ax1.set_title('Users Remaining (n ≤ ratings ≤ 300)')
ax1.grid(True, alpha=0.3)

# Movies plot
ax2.bar(high_thresholds, high_movies_counts, color='orange', edgecolor='darkorange', alpha=0.7)
ax2.set_xlabel('Minimum ratings per user (n)')
ax2.set_ylabel('Number of movies')
ax2.set_title('Movies Remaining (n ≤ ratings ≤ 300)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()