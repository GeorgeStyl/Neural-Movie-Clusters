
# In this file, we further clean the data by removing duplicate entries;
# Some users have rated the same movie multiple times - we keep only their most recent rating.

# Import necessary libraries
import numpy as np
import pandas as pd


# Load the cleaned dataset
dataset = np.load(r'../datasets/dataset_clean.npy', allow_pickle=True)


# Convert to dataframe
df = pd.DataFrame(dataset, columns=['username', 'movie', 'rating', 'date'], )

len_before = len(df)

# Parse dates
df['date'] = pd.to_datetime(df['date'], format='%d %B %Y')

# Sort based on date - so in the case of double reviews, the last one is the most recent one
df.sort_values(by='date', inplace=True)

# Drop double reviews (same user having rated the same movie more than once), keep only the most recent reviews
df.drop_duplicates(subset=['username', 'movie'], keep='last', inplace=True)

# Restore original order
df.sort_index(inplace=True)

# Calculate the number of rows that were found to be duplicate
len_after = len(df)

print(f'\nEntries dropped: {len_before - len_after}')

# Convert back to np.array and save
array = df.to_numpy()
np.save(r'../datasets/dataset_singles.npy', array)