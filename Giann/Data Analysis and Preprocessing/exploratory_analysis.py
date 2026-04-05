import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / 'plots'


# Load the dataset
dataset = np.load(r'../datasets/dataset_singles.npy', allow_pickle=True)

# Convert to DataFrame and print general info
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'])
print(f'\n' + '='*60)
print(f'DATASET OVERVIEW')
print(f'='*60)
print(f'Total ratings in dataset: {len(df):,}')
print(f'Unique users: {df['user'].nunique():,}')
print(f'Unique movies: {df['movie'].nunique():,}')
print('='*60)

# Drop the "date" column, which is irrelevant in this analysis
df.drop(['date'], axis=1, inplace=True)

# Get the number of ratings per user
ratings_per_user = df.groupby('user').size()

# Slice the possible rating counts in logical, representative bins
bins = [0, 1, 2, 3, 10, 20, 100, float('inf')]
labels = ['1', '2', '3', '4-10', '11-20', '21-100', '100+']

# Put each user in the respective bin basen on their ratings count
user_bins = pd.cut(ratings_per_user, bins=bins, labels=labels, right=True)

# Count number of users in each bin
bin_counts = user_bins.value_counts().sort_index()

# Print the results, raw and %
print(f'\nUSER RATINGS DISTRIBUTION (Original Dataset)')
print(f'-' * 50)
print(f'Number of users in each rating count bin:')
print(bin_counts)
print(f'\nPercentage of users in each rating count bin:')
print((bin_counts / len(ratings_per_user) * 100).round(2))

# Create a bar plot with scaled values
plt.figure(figsize=(10, 6))
(bin_counts / 100000).plot(kind='bar')
plt.title('Number of Users by Rating Count (Original Dataset)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Ratings per User', fontsize=12)
plt.ylabel('Number of Users (×100,000)', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_original_user_ratings_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print('='*60)


# ======================================================
# ANALYSIS FOR POSSIBLE (NAIVE) DIMENSIONALITY REDUCTION
# ======================================================

# Count how many times each movie has been rated
movie_rating_counts = df['movie'].value_counts()

# Find movies rated exactly once and show their percentage with respect to total number of movies.
# Store in list for later usage (filtering)
movies_rated_once = movie_rating_counts[movie_rating_counts == 1].index.to_list()
print(f'\nMOVIES WITH ONLY ONE RATING')
print(f'-' * 50)
print(f'Number of movies rated exactly once: {len(movies_rated_once):,}')
print(f'Percentage of total movies: {len(movies_rated_once) / len(movie_rating_counts) * 100:.2f}%')

# Find users with exactly one rating.
# Store in list for later usage (filtering)
one_rating_users = ratings_per_user[ratings_per_user == 1].index.to_list()
print(f'\nUSERS WITH ONLY ONE RATING')
print(f'-' * 50)
print(f'Number of users with only one rating: {len(one_rating_users):,}')

# In the initial dataframe, find the intersection of the entries that belong in the two lists above.
# Call them "lonely ratings".
lonely_ratings_mask = (df['user'].isin(one_rating_users)) & (df['movie'].isin(movies_rated_once))
lonely_ratings = df[lonely_ratings_mask]

# Count those lonely ratings
print(f'\nLONELY RATINGS ANALYSIS')
print(f'-' * 50)
print(f'Ratings that are from users with only one rating AND movies with only one rating:')
print(f' {len(lonely_ratings):,} ratings')
print(f'This represents {len(lonely_ratings)/len(df)*100:.2f}% of all ratings')

# Remove lonely ratings (users with one rating AND movies with one rating)
df_clean = df[~lonely_ratings_mask]

print(f'\nAFTER REMOVING LONELY RATINGS')
print(f'-' * 50)
print(f'Original dataset:')
print(f'  - Unique users: {df['user'].nunique():,}')
print(f'  - Unique movies: {df['movie'].nunique():,}')
print(f'  - Total ratings: {len(df):,}')
print(f'\nAfter removing {len(lonely_ratings):,} lonely ratings:')
print(f'  - Unique users: {df_clean['user'].nunique():,}')
print(f'  - Unique movies: {df_clean['movie'].nunique():,}')
print(f'  - Total ratings: {len(df_clean):,}')


# Also check the number of remaining movies directly, in the case of dropping 1-rating users
df_users_multiple_ratings = df[~df['user'].isin(one_rating_users)]
print(f'\nEFFECT OF REMOVING USERS WITH SINGLE RATINGS')
print(f'-' * 50)
print(f'After removing users with only one rating:')
print(f'  • Remaining movies: {df_users_multiple_ratings['movie'].nunique():,}')
print(f'  • Remaining users: {df_users_multiple_ratings['user'].nunique():,}')
print(f'  • Remaining ratings: {len(df_users_multiple_ratings):,}')

print('='*60)

# User-ratings count comparison before & after removing 1-time-rated movies
df_movies_multiple_ratings = df[~df['movie'].isin(movies_rated_once)]

# Comparison statistics
print(f'\nEFFECT OF REMOVING MOVIES WITH SINGLE RATINGS')
print(f'-' * 50)
print(f'Original unique users: {df['user'].nunique():,}')
print(f'After removing movies with only one rating: {df_movies_multiple_ratings['user'].nunique():,}')
print(f'Original unique movies: {df['movie'].nunique():,}')
print(f'After removing movies with only one rating: {df_movies_multiple_ratings['movie'].nunique():,}')

# Repeat the calculation of ratings per user distribution for the filtered dataset, using the same bins as before
ratings_per_user_filtered = df_movies_multiple_ratings.groupby('user').size()
user_bins_filtered = pd.cut(ratings_per_user_filtered, bins=bins, labels=labels, right=True)
bin_counts_filtered = user_bins_filtered.value_counts().sort_index()

# Print the results for the filtered dataset
print(f'\nUSER RATINGS DISTRIBUTION (After Removing Movies Rated Once)')
print(f'-' * 50)
print('Number of users by rating count:')
print(bin_counts_filtered)
print('\nPercentage of users by rating count:')
print((bin_counts_filtered / len(ratings_per_user_filtered) * 100).round(2))

# Create a df and compare side by side
comparison_df = pd.DataFrame({
    'Original': bin_counts,
    'Filtered': bin_counts_filtered
}).fillna(0)
print('\nCOMPARISON: User Counts by Rating Category')
print(f'-' * 50)
print(comparison_df)

# Create a bar plot for filtered data
plt.figure(figsize=(10, 6))
(bin_counts_filtered / 100000).plot(kind='bar', color='lightcoral')
plt.title('Number of Users by Rating Count (After Removing Movies Rated Once)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Ratings per User', fontsize=12)
plt.ylabel('Number of Users (×100,000)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_after_removing_single_rated_movies.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Create a bar plot showing both original and filtered
plt.figure(figsize=(12, 7))

# Set up bar positions
x = np.arange(len(bin_counts.index))
width = 0.35

# Create bars
original_bars = plt.bar(x - width/2, bin_counts.values/100000, width, label='Original', color='skyblue', edgecolor='navy', alpha=0.7)
filtered_bars = plt.bar(x + width/2, bin_counts_filtered.reindex(bin_counts.index, fill_value=0)/100000, width, label='Filtered (Removed Movies with 1 Rating)', color='lightcoral', edgecolor='darkred', alpha=0.7)

# Customize the plot
plt.xlabel('Number of Ratings per User', fontsize=12)
plt.ylabel('Number of Users (×100,000)', fontsize=12)
plt.title('Comparison: Users by Rating Count (Original vs Filtered)', fontsize=14, fontweight='bold')
plt.xticks(x, bin_counts.index, rotation=45)
plt.legend(fontsize=11)

# Add value labels on top of bars
for bar in original_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

for bar in filtered_bars:
    height = bar.get_height()
    if height > 0:  # Only add label if bar has height
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_user_ratings_comparison_original_vs_filtered.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print('='*60)

# Remove users with only one rating AND movies rated only once
multiple_ratings_filter = (~df['user'].isin(one_rating_users)) & (~df['movie'].isin(movies_rated_once))
df_multiple_ratings = df[multiple_ratings_filter]

# Print info regarding the entries that are left after filtering
print(f'\nFILTERING RESULTS (Removing Both Single-Rating Users AND Single-Rating Movies)')
print(f'-' * 50)
print(f'Original dataset entries: {len(df):,}')
print(f'After removing both conditions: {len(df_multiple_ratings):,}')
print(f'Reduction: {(1 - len(df_multiple_ratings)/len(df)) * 100:.1f}%')

# Show what's left in the filtered df
print(f'\nFINAL FILTERED DATASET COMPOSITION')
print(f'-' * 50)
print(f'Unique users: {df_multiple_ratings['user'].nunique():,}')
print(f'Unique movies: {df_multiple_ratings['movie'].nunique():,}')
print(f'Total ratings: {len(df_multiple_ratings):,}')

print('='*60)

# Movie distribution analysis (in terms of ratings received)

# Define bins for movie ratings
movie_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
movie_labels = ['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100', '100+']

# Bin the movies based on number of ratings
movie_rating_bins = pd.cut(movie_rating_counts, bins=movie_bins, labels=movie_labels, right=True)

# Count movies in each bin
movie_bin_counts = movie_rating_bins.value_counts().sort_index()

# Print results (raw and %)
print(f'\nMOVIE RATINGS DISTRIBUTION')
print(f'-' * 50)
print('Number of movies by rating count:')
print(movie_bin_counts)
print('\nPercentage of movies by rating count:')
print((movie_bin_counts / len(movie_rating_counts) * 100).round(2))

# Plot distribution
plt.figure(figsize=(10,6))
(movie_bin_counts / 1000).plot(kind='bar', color='mediumseagreen')
plt.title('Movie Distribution by Number of Ratings', fontsize=14, fontweight='bold')
plt.xlabel('Number of Ratings per Movie', fontsize=12)
plt.ylabel('Number of Movies (×1,000)', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_movie_ratings_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print('='*60)


# Pick a threshold for the number of movies we want to keep from the dataset (top n)
threshold = 10000

# Get the top n movies (based on number of ratings) on the df, after removing users with only 1 rating
# (setting R_min = 1 in terms of our project) (remember we have this df ready from previous analysis)
movie_ratings_count_after_r1 = df_users_multiple_ratings['movie'].value_counts()
top_n_movies = movie_ratings_count_after_r1.nlargest(threshold).index

# Create a df with the entries regarding only those top n movies
df_top_movies = df_users_multiple_ratings[df_users_multiple_ratings['movie'].isin(top_n_movies)]

# Final counts
print(f'\nDIMENSIONALITY REDUCTION SCENARIO 1')
print(f'Keep users with multiple ratings + top {threshold:,} most-rated movies')
print(f'-' * 50)
print(f'Movies kept: {df_top_movies['movie'].nunique():,}')
print(f'Users remaining: {df_top_movies['user'].nunique():,}')
print(f'Total ratings: {len(df_top_movies):,}')

ratings_per_movie = df_top_movies.groupby('movie').size()
print(f'Minimum ratings per movie in filtered dataset: {ratings_per_movie.min():,}')


# Repeat the process above, for the same threshold, but this time keep users with at least n ratings
n = 8
threshold = 5000

# Find users with at least n ratings
at_least_n_rating_users = ratings_per_user[ratings_per_user >= n].index.to_list()

# Keep only those users
df_users_filtered = df[df['user'].isin(at_least_n_rating_users)]

# Count movie ratings after filtering users
movie_ratings_count_after_r1_n = df_users_filtered['movie'].value_counts()

# Keep top N movies
top_n_movies_second = movie_ratings_count_after_r1_n.nlargest(threshold).index

# Create filtered dataframe
df_top_movies_second = df_users_filtered[df_users_filtered['movie'].isin(top_n_movies_second)]

# Final statistics
print(f'\nDIMENSIONALITY REDUCTION SCENARIO 2')
print(f'Keep users with ≥ {n} ratings + top {threshold:,} most-rated movies')
print(f'-' * 50)
print(f'Movies kept (dimensions): {df_top_movies_second["movie"].nunique():,}')
print(f'Users remaining (training points): {df_top_movies_second["user"].nunique():,}')
print(f'Total ratings: {len(df_top_movies_second):,}')
print(f'Matrix density: {len(df_top_movies_second) / (df_top_movies_second["movie"].nunique() * df_top_movies_second["user"].nunique()) * 100:.2f}%')

ratings_per_movie_second = df_top_movies_second.groupby('movie').size()
print(f'Minimum ratings per movie in filtered dataset: {ratings_per_movie_second.min():,}')
print('='*60)

# ======================================================
# ANALYSIS: FILTERING BASED ON MINIMUM RATINGS PER USER
# ======================================================

print('\n' + '=' * 60)
print('FILTERING BASED ON MINIMUM RATINGS PER USER (r)')
print('=' * 60)

# Define the r values to test
r_values = [5, 10, 15, 20, 25, 30, 40, 50, 70, 80, 100]

# Store results
users_remaining = []
movies_remaining = []
ratings_remaining = []

# Calculate for each r value
for r in r_values:
    # Find users with at least r ratings
    users_with_min_ratings = ratings_per_user[ratings_per_user >= r].index.to_list()

    # Filter dataframe to keep only those users
    df_filtered_by_user = df[df['user'].isin(users_with_min_ratings)]

    # Store results
    users_remaining.append(df_filtered_by_user['user'].nunique())
    movies_remaining.append(df_filtered_by_user['movie'].nunique())
    ratings_remaining.append(len(df_filtered_by_user))

    # Print results
    print(f'\nr = {r}:')
    print(f'  • Users remaining: {df_filtered_by_user["user"].nunique():,}')
    print(f'  • Movies remaining: {df_filtered_by_user["movie"].nunique():,}')
    print(f'  • Ratings remaining: {len(df_filtered_by_user):,}')

# Create figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Users remaining
ax1.bar(r_values, users_remaining, color='skyblue', edgecolor='navy', alpha=0.7)
ax1.set_xlabel('Minimum Ratings per User (r)', fontsize=12)
ax1.set_ylabel('Number of Users Remaining', fontsize=12)
ax1.set_title('Users Remaining vs. Minimum Ratings per User', fontsize=14, fontweight='bold')
ax1.set_xticks(r_values)
ax1.set_xticklabels(r_values, rotation=45)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for bar in ax1.patches:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height):,}', ha='center', va='bottom', fontsize=9)

# Plot 2: Movies remaining
ax2.bar(r_values, movies_remaining, color='lightcoral', edgecolor='darkred', alpha=0.7)
ax2.set_xlabel('Minimum Ratings per User (r)', fontsize=12)
ax2.set_ylabel('Number of Movies Remaining', fontsize=12)
ax2.set_title('Movies Remaining vs. Minimum Ratings per User', fontsize=14, fontweight='bold')
ax2.set_xticks(r_values)
ax2.set_xticklabels(r_values, rotation=45)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for bar in ax2.patches:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(height):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_filtering_by_min_ratings_per_user.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Create a summary table
summary_df = pd.DataFrame({
    'r': r_values,
    'Users Remaining': users_remaining,
    'Movies Remaining': movies_remaining,
    'Ratings Remaining': ratings_remaining
})

print('\n' + '=' * 60)
print('SUMMARY TABLE: Filtering by Minimum Ratings per User')
print('=' * 60)
print(summary_df.to_string(index=False))
print('=' * 60)