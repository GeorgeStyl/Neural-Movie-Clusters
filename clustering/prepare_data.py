import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix

# --- Constants ---
R_MIN = 7
INPUT_FILE = r'/home/george/WorkSpace/Master/Eks1/ML/Final_Project/data/Dropbox_Dataset.npy'
OUTPUT_FILE = '../data/user_movie_matrix_rmin_7.npz'  # Note: .npz is better for sparse

# --- Color Constants ---
GREEN, YELLOW, RED, ORANGE, RESET = '\033[92m', '\033[93m', '\033[91m', '\033[38;5;208m', '\033[0m'

print(f"{YELLOW}Loading and splitting data...{RESET}")
arr = np.load(INPUT_FILE, allow_pickle=True)
data_split = [line.split(',') for line in arr]
df = pd.DataFrame(data_split, columns=['username', 'movie', 'rating', 'date'])
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Filter Users
user_counts = df['username'].value_counts()
u_cap_indices = user_counts[user_counts >= R_MIN].index
df_filtered = df[df['username'].isin(u_cap_indices)].copy()

print(f"{GREEN}Filtering complete. Unique Users: {len(u_cap_indices)}{RESET}")

# --- Efficient Sparse Matrix Construction ---
# 1. Map usernames and movies to integer IDs (0 to N-1)
df_filtered['user_id'] = pd.Categorical(df_filtered['username']).codes
df_filtered['movie_id'] = pd.Categorical(df_filtered['movie']).codes

# 2. Extract coordinates and values
row = df_filtered['user_id'].values
col = df_filtered['movie_id'].values
data = df_filtered['rating'].values

# 3. Build the CSR (Compressed Sparse Row) Matrix
# Shape is (Total Users, Total Movies)
print(f"{YELLOW}Building Sparse Matrix...{RESET}")
sparse_matrix = csr_matrix((data, (row, col)),
                            shape=(len(u_cap_indices), df_filtered['movie'].nunique()))

# 4. Save the Sparse Matrix
import scipy.sparse
scipy.sparse.save_npz(OUTPUT_FILE, sparse_matrix)

print(f"{ORANGE}{'-' * 50}{RESET}")
print(f"{GREEN}Success! Sparse matrix saved as: {OUTPUT_FILE}{RESET}")
print(f"Matrix Shape: {sparse_matrix.shape}")
print(f"Stored Ratings: {sparse_matrix.nnz} (Non-zero elements){RESET}")