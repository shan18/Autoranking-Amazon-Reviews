import nltk
import numpy as np
import matplotlib.pyplot as plt

from get_data import parse_input
from preprocess_filter_data import preprocess_input, filter_input
from extract_features import extract_textual_features

# path containing the nltk data on the system
nltk.data.path.append('/home/shan/Packages/nltk_data')


INPUT_FILE = 'reviews_Video_Games_5.json.gz'

# Store the input in a dataframe
df = parse_input(INPUT_FILE)
print(df.head())

# Preprocessing and Filtering the dataset
df = preprocess_input(df)
df = filter_input(df)
print('\n\nPreprocessed and Filtered data:\n')
print(df.head())

# Separate data into training set and test set
df_train = df.sample(frac=0.8)
df_test = df.loc[~df.index.isin(df_train.index)]

# Preview training dataset
print('\n\nTraining dataset:\n')
print(df_train.head())

# Set the outcome variable (Denotes helpfulness of a review)
upvotes = np.array(df_train['upvotes'].values)
downvotes = np.array(df_train['downvotes'].values)
Y_train = upvotes / (upvotes + downvotes)

# Visualizing the distribution of helpfulness score across the data
plt.hist(Y_train, ec='black')
plt.xlabel('Helpfulness Score')
plt.show()


# Constructing feature vector

# Textual Features
X_train = extract_textual_features(df_train)
