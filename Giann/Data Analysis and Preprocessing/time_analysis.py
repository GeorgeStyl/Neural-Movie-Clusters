
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt


# Load the dataset
dataset = np.load('../datasets/dataset_singles.npy', allow_pickle=True)

# Convert to pd dataframe
df = pd.DataFrame(dataset, columns=['user', 'movie', 'rating', 'date'])
# Convert "date" column to dates
df['date'] = pd.to_datetime(df['date'])
# Drop "movie" and "rating" column, which are unnecessary in this analysis
df = df[['user', 'date']].copy()


# ==============================================
# HISTOGRAM 1: Number of ratings per user
# ==============================================

# Count ratings per user
ratings_per_user = df.groupby('user').size()

# Create histogram

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(ratings_per_user, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.title('Distribution of Ratings per User')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Add this after creating ratings_per_user to see the distribution
print(f"Ratings per user - Min: {ratings_per_user.min()}, Max: {ratings_per_user.max()}")
print(f"Percentiles: 25%: {ratings_per_user.quantile(0.25)}, 50%: {ratings_per_user.quantile(0.50)}, 75%: {ratings_per_user.quantile(0.75)}")

