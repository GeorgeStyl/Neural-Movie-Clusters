
# In this file, we load the raw dataset and clean it, in terms of bringing it to a form suitable for analysis
# and preprocessing. We reshape the data, convert string values to numerical types, and save the cleaned dataset.

# Import the necessary libraries
import pandas as pd
import numpy as np


# Load original dataset
dataset = np.load('../data/Dropbox_Dataset.npy')

# Convert dataset to pandas Series
s = pd.Series(dataset)

# Split each string entry into separate columns and create a DataFrame
# Columns: Username (0), Movie ID (1), Rating (2), Date (3)
split_data = s.str.split(',', expand=True)

# Make sure that no entry had extra ","'s, thus creating extra columns
if len(split_data.columns) > 4:
    print('Original dataset contains rows with unexpected extra comma-separated fields. Please handle.')
else:
    print('All rows in the original dataset have the expected number of fields. Proceed further.')

# Clean and convert each column appropriately
split_data[0] = split_data[0].str[2:].astype(int)  # Remove 'ur' prefix for each user, convert to int
split_data[1] = split_data[1].str[2:].astype(int)  # Remove 'tt' prefix for each movie, convert to int
split_data[2] = split_data[2].astype(int)          # Convert ratings to int's
split_data[3] = pd.to_datetime(split_data[3])      # Convert dates to datetime objects

# Make sure that the "ratings" column contains only valid entries and check for null entries in the dataset
wrong_ratings_flag = len(split_data[(split_data[2] < 0) | (split_data[2] > 10)]) > 0
if wrong_ratings_flag:
    print('Rating column contains invalid entries. Please handle.')
else:
    print('Dataset does not contain invalid ratings or null entries. Proceed further.')

# Regarding Null entries, no explicit check is necessary here. Regarding columns "username", "rating" and "movie id",
# the absence of such entries is ensured by the completion of type conversions of the respective columns.
# Null entries in the "date" column will not have major consequences on our analysis.

# Convert back to numpy array
array = split_data.to_numpy()

# Save the cleaned dataset
np.save(r'datasets/dataset_clean.npy', array, )
