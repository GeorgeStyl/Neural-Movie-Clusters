
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / 'plots'


# Load the filtered dataframe
df_final = pd.read_csv('../datasets/df_final.csv')

# Convert "date" column to datetime
df_final['date'] = pd.to_datetime(df_final['date'])

# ======================================================
# Histogram 1: Number of ratings per user (with custom bins)
# ======================================================

ratings_per_user = df_final.groupby('user').size()

# Define custom bins
bins = [20, 30, 40, 50, 75, 100, 150, 200, float('inf')]
labels = ['20-29', '30-39', '40-49', '50-74', '75-99', '100-149', '150-199', '200+']

# Bin the users
user_bins = pd.cut(ratings_per_user, bins=bins, labels=labels, right=True)
bin_counts = user_bins.value_counts().sort_index()

# Create bar plot
plt.figure(figsize=(12, 6))
bin_counts.plot(kind='bar', color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Number of Ratings per User', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.title('Distribution of Ratings per User (Filtered Dataset)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars
for i, (bar, count) in enumerate(zip(plt.gca().patches, bin_counts.values)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'ratings_per_user.png')
plt.show()

print(f"Ratings per user - Min: {ratings_per_user.min()}, Max: {ratings_per_user.max()}")
print(f"Percentiles: 25%: {ratings_per_user.quantile(0.25)}, "
      f"50%: {ratings_per_user.quantile(0.50)}, "
      f"75%: {ratings_per_user.quantile(0.75)}")

# ======================================================
# Histogram 2: Time span of ratings per user
# ======================================================

# Compute the time span (in days) for each user
time_span_per_user = (df_final.groupby('user')['date'].max() - df_final.groupby('user')['date'].min()).dt.days

plt.figure(figsize=(10, 6))
plt.hist(time_span_per_user, bins=50, edgecolor='black', alpha=0.7, color='salmon')
plt.xlabel('Time Span of Ratings (days)')
plt.ylabel('Number of Users')
plt.title('Distribution of Time Span of Ratings per User (Filtered Dataset)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'hist_time_span_per_user.png')
plt.show()

print(f"Time span per user - Min: {time_span_per_user.min()} days, Max: {time_span_per_user.max()} days")
print(f"Percentiles: 25%: {time_span_per_user.quantile(0.25)}, "
      f"50%: {time_span_per_user.quantile(0.50)}, "
      f"75%: {time_span_per_user.quantile(0.75)}")