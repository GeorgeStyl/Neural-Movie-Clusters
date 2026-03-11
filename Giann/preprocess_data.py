
import numpy as np
import pandas as pd


# Load the final dataset
arr = np.load(r'datasets/dataset_final.npy', allow_pickle=True)

# Get the respective dataframe
df = pd.DataFrame(arr, columns=['username', 'movie', 'rating', 'date'])

# Find the sets of unique users and unique movies
unique_users = np.unique(arr[:, 0])
unique_movies = np.unique(arr[:, 1])

# Calculate and show their cardinality
print(f'Number of unique users: {len(unique_users)}')
print(f'Number of unique movies: {len(unique_movies)}')

# Group by user and get the movies each has rated. Store the result
user_movies_cache = df.groupby('username')['movie'].apply(list).to_dict()


def get_rated_movies(username):
    """
    Return list of movies rated by the given user, empty list if none found - for completeness; the analysis in the
    previous scripts showed that no such users will be found.
    """
    return user_movies_cache.get(username, [])


# Now we have to create U_cap and I_cap.
# steps to follow:
# 1) determine R_min (r) and R_max (R)
# 2) find users "user" for which len(get_rated_movies(user)) will be inside r and R. Now we have U_cap
# 3) find which movies where rated by users that belong only in U_cap. This is I_cap

# dummy thresholds for now
r = 0 # bottom threshold
R = 10000 # top threshold

# Find U_cap
# First, find the count of ratings per user
user_rating_counts = df['username'].value_counts()
# Find the users with number of ratings inside the allowed range
valid_users = user_rating_counts[(user_rating_counts >= r) & (user_rating_counts <= R)].index
# PLace them in a set
u_cap_set = set(valid_users) # We now have U_cap
# Filter the dataset (for finding I_cap)
filt = df['username'].isin(u_cap_set)
df = df[filt]

# Find I_cap
movies = df['movie'].unique()
i_cap_set = set(movies)



