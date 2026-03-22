
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
dataset = np.load(r'../datasets/dataset_singles.npy', allow_pickle=True)

# Convert to DataFrame and print general info
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'])
print(f'DataFrame info:')
print(f'-Total ratings: {len(df)}')
print(f'-Unique users in the initial dataframe: {df['user'].nunique()}')
print(f'-Unique movies in the initial dataframe: {df['movie'].nunique()}')
print('-'*50)

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
print(f'\nNumber of users in each rating count bin:')
print(bin_counts)
print(f'\nPercentage of users in each rating count bin:')
print((bin_counts / len(ratings_per_user) * 100).round(2))

# # Create a bar plot with scaled values
#
# plt.figure(figsize=(10, 6))
# (bin_counts / 100000).plot(kind='bar')
# plt.title('Number of Users by Rating Count')
# plt.xlabel('Number of Ratings per User')
# plt.ylabel('Number of Users (×100,000)')
# plt.xticks(rotation=45)
#
# plt.tight_layout()
# plt.show()
print('-'*50)


# ======================================================
# ANALYSIS FOR POSSIBLE (NAIVE) DIMENSIONALITY REDUCTION
# ======================================================

# Count how many times each movie has been rated
movie_rating_counts = df['movie'].value_counts()

# Find movies rated exactly once and show their percentage with respect to total number of movies.
# Store in list for later usage (filtering)
movies_rated_once = movie_rating_counts[movie_rating_counts == 1].index.to_list()
print(f'\nNumber of movies rated exactly once: {len(movies_rated_once)}')
print(f'Percentage of movies rated exactly once: {len(movies_rated_once) / len(movie_rating_counts) * 100:.2f}%')

# Find users with exactly one rating.
# Store in list for later usage (filtering)
one_rating_users = ratings_per_user[ratings_per_user == 1].index.to_list()
print(f'\nNumber of users with only one rating: {len(one_rating_users)}')

# In the initial dataframe, find the intersection of the entries that belong in the two lists above.
# Call them "lonely ratings".
lonely_ratings_mask = (df['user'].isin(one_rating_users)) & (df['movie'].isin(movies_rated_once))
lonely_ratings = df[lonely_ratings_mask]

# Count those lonely ratings, to see how many of these 1-time-rated movies we will save ourselves from if we simply
# drop users with only one rating (set R_min = 1, in terms of the project)
print(f'\nNumber of ratings (entries in our initial dataframe) that '
      f'\n1) are the only rating the respective user has left, and'
      f'\n2) at the same time are the only ratings the respective movie has received: \n {len(lonely_ratings)}')

# Also check the number of remaining movies directly, in the case of dropping 1-rating users
df_users_multiple_ratings = df[~df['user'].isin(one_rating_users)]
print(f'\nThe number of movies remaining in the dataset in the case of removing users with only one rating is:'
      f'\n{df_users_multiple_ratings['movie'].nunique()}')

# The result is too small for the above filtering of the dataset (setting R_min = 1) to be considered an effective
# dimensionality reduction. Further inspection is needed.
print('-'*50)

# User-ratings count comparison before & after removing 1-time-rated movies
df_movies_multiple_ratings = df[~df['movie'].isin(movies_rated_once)]

# Comparison statistics
print(f'\nOriginal unique users: {df['user'].nunique()}')
print(f'Unique users after removing movies with only one rating: {df_movies_multiple_ratings['user'].nunique()}')
print(f'Original unique movies: {df['movie'].nunique()}')
print(f'Unique movies after removing movies with only one rating from df: {df_movies_multiple_ratings['movie'].nunique()}')

# Repeat the calculation of ratings per user distribution for the filtered dataset, using the same bins as before
ratings_per_user_filtered = df_movies_multiple_ratings.groupby('user').size()
user_bins_filtered = pd.cut(ratings_per_user_filtered, bins=bins, labels=labels, right=True)
bin_counts_filtered = user_bins_filtered.value_counts().sort_index()

# Print the results for the filtered dataset
print('\nNumber of users by rating count (after removing movies rated once):')
print(bin_counts_filtered)
print('\nPercentage of users by rating count (after removing movies rated once):')
print((bin_counts_filtered / len(ratings_per_user_filtered) * 100).round(2))

# Create a df and compare side by side
comparison_df = pd.DataFrame({
    'Original': bin_counts,
    'Filtered': bin_counts_filtered
}).fillna(0)
print('\nComparison of user counts:')
print(comparison_df)

# # Create a bar plot for filtered data
#
# plt.figure(figsize=(10, 6))
# (bin_counts_filtered / 100000).plot(kind='bar', color='lightcoral')
# plt.title('Number of Users by Rating Count (Filtered Data - Movies Rated Once Removed)')
# plt.xlabel('Number of Ratings per User')
# plt.ylabel('Number of Users (×100,000)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # Create a bar plot showing both original and filtered
# plt.figure(figsize=(12, 7))
#
# # Set up bar positions
# x = np.arange(len(bin_counts.index))
# width = 0.35
#
# # Create bars
# original_bars = plt.bar(x - width/2, bin_counts.values/100000, width, label='Original', color='skyblue', edgecolor='navy', alpha=0.7)
# filtered_bars = plt.bar(x + width/2, bin_counts_filtered.reindex(bin_counts.index, fill_value=0)/100000, width, label='Filtered', color='lightcoral', edgecolor='darkred', alpha=0.7)
#
# # Customize the plot
# plt.xlabel('Number of Ratings per User', fontsize=12)
# plt.ylabel('Number of Users (×100,000)', fontsize=12)
# plt.title('Comparison: Users by Rating Count (Original vs Filtered)', fontsize=14, fontweight='bold')
# plt.xticks(x, bin_counts.index, rotation=45)
# plt.legend(fontsize=11)
#
# # Add value labels on top of bars
# for bar in original_bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height,
#              f'{height:.1f}', ha='center', va='bottom', fontsize=9)
#
# for bar in filtered_bars:
#     height = bar.get_height()
#     if height > 0:  # Only add label if bar has height
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                  f'{height:.1f}', ha='center', va='bottom', fontsize=9)
#
# plt.grid(axis='y', alpha=0.3, linestyle='--')
# plt.tight_layout()
# plt.show()

print('-'*50)

# Remove users with only one rating AND movies rated only once
multiple_ratings_filter = (~df['user'].isin(one_rating_users)) & (~df['movie'].isin(movies_rated_once))
df_multiple_ratings = df[multiple_ratings_filter]

# Print info regarding the entries that are left after filtering
print(f'\nOriginal dataframe length (total entries): {len(df)}')
print(f'Final dataframe length after removing both conditions: {len(df_multiple_ratings)}')

# Show what's left in the filtered df
print(f'\nWhat remains in the final filtered dataframe:')
print(f'Unique users: {df_multiple_ratings['user'].nunique()}')
print(f'Unique movies: {df_multiple_ratings['movie'].nunique()}')
print(f'Total ratings: {len(df_multiple_ratings)}')

print('-'*50)

# Movie distribution analysis (in terms of ratings received)

# Define bins for movie ratings
movie_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
movie_labels = ['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100', '100+']

# Bin the movies based on number of ratings
movie_rating_bins = pd.cut(movie_rating_counts, bins=movie_bins, labels=movie_labels, right=True)

# Count movies in each bin
movie_bin_counts = movie_rating_bins.value_counts().sort_index()

# Print results (raw and %)
print('\nNumber of movies by rating count:')
print(movie_bin_counts)
print('\nPercentage of movies by rating count:')
print((movie_bin_counts / len(movie_rating_counts) * 100).round(2))

# # Plot distribution
# plt.figure(figsize=(10,6))
# (movie_bin_counts / 1000).plot(kind='bar', color='mediumseagreen')
# plt.title('Movie Distribution by Number of Ratings')
# plt.xlabel('Number of Ratings per Movie')
# plt.ylabel('Number of Movies (×1,000)')
# plt.xticks(rotation=45)
#
# plt.tight_layout()
# plt.show()

print('-'*50)


# Pick a threshold for the number of movies we want to keep from the dataset (top n)
threshold = 10000

# Get the top n movies (based on number of ratings) on the df, after removing users with only 1 rating
# (setting R_min = 1 in terms of our project) (remember we have this df ready from previous analysis)
movie_ratings_count_after_r1 = df_users_multiple_ratings['movie'].value_counts()
top_n_movies = movie_ratings_count_after_r1.nlargest(threshold).index

# Create a df with the entries regarding only those top n movies
df_top_movies = df_users_multiple_ratings[df_users_multiple_ratings['movie'].isin(top_n_movies)]

# Final counts
print(f'\nMovies kept after keeping top {threshold} movies: {df_top_movies['movie'].nunique()}')
print(f'Unique users remaining after keeping top {threshold} movies: {df_top_movies['user'].nunique()}')
print(f'Total ratings remaining after keeping top {threshold} movies: {len(df_top_movies)}')

ratings_per_movie = df_top_movies.groupby('movie').size()
print(f'The movies still remaining in the dataset contain at least {ratings_per_movie.min()} ratings.')



# ========================
# Repeat the process above, for the same threshold, but this time keep users with at least n ratings
n = 8
# HERE HERE HERE HERE HERE
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
print(f'\nMovies kept after keeping top {threshold} movies (users with ≥ {n} ratings):'
      f' {df_top_movies_second["movie"].nunique()}. This is the number of dimensions we have.')
print(f'Unique users remaining (users with ≥ {n} ratings): {df_top_movies_second["user"].nunique()}. '
      f'This is the number of training points we have.')
print(f'Total ratings remaining (users with ≥ {n} ratings): {len(df_top_movies_second)}')
print(f'Matrix density: {len(df_top_movies_second) / (df_top_movies_second["movie"].nunique() * df_top_movies_second["user"].nunique()) * 100:.2f}%')

ratings_per_movie_second = df_top_movies_second.groupby('movie').size()

print(f'The movies still remaining in the dataset contain at least {ratings_per_movie_second.min()} ratings.')

