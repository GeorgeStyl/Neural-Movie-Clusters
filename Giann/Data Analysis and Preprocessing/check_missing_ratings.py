
# Check for "0" ratings in the dataset (that denote the absence of a rating)

import numpy as np
import pandas as pd


# Load cleaned dataset
arr = np.load('../datasets/dataset_singles.npy', allow_pickle=True)

# Check for "0" ratings in the cleaned dataset
df = pd.DataFrame(arr, columns=['user', 'movie', 'rating', 'date'])
df = df[df['rating'] == 0]
print(df.head(20))  # --> will return an empty df

# Cross-check in the raw dataset
# Load raw dataset
raw_dataset = np.load('../datasets/Dataset.npy', allow_pickle=True)
zero_count = 0
for i, row in enumerate(raw_dataset):
    if ',0,' in row:
        zero_count += 1

print(f"Total zeros in raw dataset: {zero_count}")

