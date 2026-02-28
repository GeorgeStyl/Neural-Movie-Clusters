# This script file provides fundamental preprocessing operations for the IMDB
# recommendation dataset.

# Import required Python frameworks.
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx

# FUNCTIONS DEFINITION
def plot_histogram(series, title_str):
    n, bins, patches = plt.hist(series, bins='auto', color='red',
                                alpha=0.7, rwidth=0.75)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(series.name)
    plt.ylabel('Frequency')
    plt.title(title_str)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    filename = series.name + ".png"
    figure_url = os.path.join(FIGURES_PATH, filename)
    plt.savefig(figure_url, dpi=100, format='png', bbox_inches='tight')
    plt.show()

# Set the figures path.
FIGURES_PATH = "figures"
os.makedirs(FIGURES_PATH,exist_ok=True)
# Set datafiles folder.
datafolder = "datafiles"
# Create directory in case it does not exist.
os.makedirs(datafolder,exist_ok=True)

# Set the name of the datafile.
datafile = "Dataset.npy"
# Load the dataset.
dataset = np.load(datafile)
# Define the spliter lambda function in order to tokenize the initial string 
# data.
spliter = lambda s: s.split(",")
# Apply the spliter function at each element of the dataset string array.
dataset = np.array([spliter(x) for x in dataset])

# Set the pickle file for storing the initial dataframe.
pickle_file = os.path.join(datafolder,"dataframe.pkl")
# Check the existence of the specifiied file.
if os.path.exists(pickle_file):
    # Load the pickle file.
    dataframe = pd.read_pickle(pickle_file)
else:
    # Create the dataframe object.
    dataframe = pd.DataFrame(dataset, columns=["user","item","rating","date"])
    # Convert the string elements of the "users" series into integers.
    dataframe["user"] = dataframe["user"].apply(lambda s:np.int64(s.replace("ur","")))
    # Convert the string elements of the "items" series into integers.
    dataframe["item"] = dataframe["item"].apply(lambda s:np.int64(s.replace("tt","")))
    # Conver the string elements of the "ratings" series into integers.
    dataframe["rating"] = dataframe["rating"].apply(lambda s:np.int64(s))
    # Convert the string elements of the "dates" series into datetime objects.
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe.to_pickle(pickle_file)

# Get the unique users in the dataset.
users = dataframe["user"].unique()
# Get the number of unique users.
users_num = len(users)
# Get the unique items in the dataset.
items = dataframe["item"].unique()
items_num = len(items)
# Get the total number of existing ratings.
ratings_num = dataframe.shape[0]
# Report the number of unique users and items in the dataset.
print("INITIAL DATASET: {0} number of unique users and {1} of unique items".format(users_num,items_num))
# Report the total number of existing ratings in the dataset.
print("INITIAL DATASET: {} total number of existing ratings".format(ratings_num))

# Set the pickle file that will store the ratings per user dataframe.
pickle_file = os.path.join(datafolder,"ratings_num_df.pkl")
# Check the existence of the previously defined pickle file.
if os.path.exists(pickle_file):
    # Load pickle file.
    ratings_num_df = pd.read_pickle(pickle_file)
else:
    # Create a new dataframe object featuring the unique users and the corresponding
    # number of total reviews in descebsing order.
    ratings_num_df =  dataframe.groupby("user")["rating"].count().sort_values(ascending=
                                    False).reset_index(name="ratings_num")
    # Save the previously created dataframe to pickle.
    ratings_num_df.to_pickle(pickle_file)

# Set the pickle file that will store the time span per user  dataframe.
pickle_file = os.path.join(datafolder,"ratings_span_df.pkl")
# Check the existence of the previously defined pickle file.
if os.path.exists(pickle_file):
    # Load the pickle file.
    ratings_span_df = pd.read_pickle(pickle_file)
else:
    # Create a new dataframe object featuring the unique users and the corrsponding
    # time span of their ratings in days.
    ratings_span_df = dataframe.groupby("user")["date"].apply(lambda date: 
    max(date)-min(date)).sort_values(ascending=False).reset_index(name="ratings_span")
    ratings_span_df.to_pickle(pickle_file)

# Create a new ratings dataframe the joining the previously defined dataframes.
ratings_df = ratings_num_df.join(ratings_span_df.set_index('user'),on='user')
# Covert time span to integer values.
ratings_df["ratings_span"] = ratings_df["ratings_span"].dt.days

# Set the threshold values for minimum and maximum ratings per user.
minimum_ratings = 100
maximum_ratings = 300
# Discard all users with more that ratings_threshold ratings.
reduced_ratings_df = ratings_df.loc[(ratings_df["ratings_num"] >= minimum_ratings) & 
                                    (ratings_df["ratings_num"] <= maximum_ratings)]
# Generate the frequency histogram for the number of ratings per user.
plot_histogram(reduced_ratings_df["ratings_num"],"Number of Ratings per User")
# Generate the frequency histogram for the time span of ratings per user.
plot_histogram(reduced_ratings_df["ratings_span"],"Time Span of Ratings per User")

# Get the final dataframe by excluding all users whose ratings fall outside the
# prespecified range.
final_df = dataframe.loc[dataframe["user"].isin(reduced_ratings_df["user"])].reset_index()
# Drop the links (indeces) to the original table.
final_df = final_df.drop("index",axis=1)

# Get the unique users and items in the final dataframe along with the final
# number of ratings.
final_users = final_df["user"].unique()
final_items = final_df["item"].unique()
final_users_num = len(final_users)
final_items_num = len(final_items)
final_ratings_num = len(final_df)

# Report the final number of unique users and items in the dataset.
print("REDUCED DATASET: {0} number of unique users and {1} of unique items".format(final_users_num,final_items_num))
# Report the final  number of existing ratings in the dataset.
print("REDUCED DATASET: {} total number of existing ratings".format(final_ratings_num))

# We need to reset the users and items ids in order to be able to construct the
# networks of users and items. Users and Items ids should be consecutive integers
# in the [1...final_users_num] and [1...final_items_num].
# Initialy, we need to acquire the sorted versions of the user and item ids.
sorted_final_users = np.sort(final_users)
sorted_final_items = np.sort(final_items)
# Generate the dictionary of final users as a mapping of the following form:
# sorted_final_users --> [0...final_users_num-1]
final_users_dict = dict(zip(sorted_final_users,list(range(0,final_users_num))))
# Generate the dictionary of final items as a mapping of the following form:
# sorted_final_items --> [0...final_items_num-1]
final_items_dict = dict(zip(sorted_final_items,list(range(0,final_items_num))))
# Apply the previously defined dictionary-based maps on the users and item 
# columns of the final dataframe.
final_df["user"] = final_df["user"].map(final_users_dict)
final_df["item"] = final_df["item"].map(final_items_dict)

# Get a grouped version of the final dataframe based on the unique final 
# users.
users_group_df = final_df.groupby("user")

# Initialize the adjacency matrix which stores the connection status for each
# pair of users in the recommendation network.
W = np.zeros((final_users_num,final_users_num))
# Initialize the matrix storing the number of commonly rated items for each pair
# of users.
CommonRatings = np.zeros((final_users_num,final_users_num))
# Initialize a the matrix of commmon ratings.
# Matrix W will be of size [final_users_num x final_users_num].
# Let U = {u1,u2,...,un} be the final set of users and I = {i1,i2,...,im} the
# final set of movie items. By considering the function Fi: U --> P(I) where
# P(I) is the powerset of I, Fi(u) returns the subset of items that have been
# rated by user u. In this context, the edge weight between any given pair of
# users (u,v) will be computed as:
#    
#          |Intersection(Fi(u),Fi(v))|
# W(u,v) = ----------------------------
#              |Union(Fi(u),Fi(v)))|

# In order to speed up the construction of the adjacency matrix for the users'
# ratings network, construct a dictionary object that will store a set of the
# rated items for each unique user.
user_items_dict = {}
for user in final_users:
    user_index = final_users_dict[user]
    user_items = set(users_group_df.get_group(user_index)["item"])
    user_items_dict[user_index] = user_items

# Sort the users' items dictionary by the user_index.
user_ids = list(user_items_dict.keys())
user_ids.sort()
# Generate the sorted version of the dictionary.
user_items_dict = {user_index:user_items_dict[user_index] for user_index in user_ids}

# Set the pickle file that will store the graph adjacency matrix W.
adjacency_numpy_file = os.path.join(datafolder,"W.npy")
common_ratings_numpy_file = os.path.join(datafolder,"CommonRatings.npy")
# Check the existence of the previously defined numpy file.
if os.path.exists(adjacency_numpy_file):
    # Load the numpy file.
    W = np.load(adjacency_numpy_file,allow_pickle=True)
    CommonRatings = np.load(common_ratings_numpy_file,allow_pickle=True)
else:
    # Loop through the rectangular grid of users that index the elements of matrix 
    # W.
    for source_user in user_items_dict.keys():
        for target_user in user_items_dict.keys():
            print("Pricessing users' pair ({0},{1})".format(source_user,target_user))
            intersection_items = user_items_dict[source_user].intersection(user_items_dict[target_user])
            union_items = user_items_dict[source_user].union(user_items_dict[target_user])
            W[source_user,target_user] = len(intersection_items) / len(union_items)
            CommonRatings[source_user,target_user] = len(intersection_items)
    # Save adjacency matrix to prespecified pickle file.
    np.save(adjacency_numpy_file,W)
    np.save(common_ratings_numpy_file,CommonRatings)

# Set the gml file that will store the network object in order to allow further
# processing with Gephi.
graph_file = os.path.join(datafolder,"recommendation_network.gml")
# Construct the networkx graph object.
G = nx.from_numpy_array(CommonRatings)
# Check the existence of the prespecified gml file.
if not os.path.exists(graph_file):
    # Save the graph object to the prespecified file.
    nx.write_gml(G,graph_file)