import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from get_data import parse_input
from preprocess_filter_data import preprocess_input, filter_input
from extract_features import create_bag_of_words, create_tf_idf_vector
from create_input_output_vectors import get_xy_vectors

# path containing the nltk data on the system
# If the nltk_data directory is at the default path then comment out the line below
# otherwise set it to the desired location
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

# X_train: Input vector for the training set
# Y_train: Vector containing the results of the training set
X_train, Y_train = get_xy_vectors(df_train)

# X_test: Input vector for the test set
# Y_test: Vector containing the results of the test set
X_test, Y_test = get_xy_vectors(df_test)

# Bag of Words matrix
bow = create_bag_of_words(pd.concat([df_train, df_test]))

# tf-idf matrix
# matrix = create_tf_idf_vector(pd.concat([df_train, df_test]))

# Concat the bag of words matrix to the feature vectors
k = X_train.shape[0]
X_train = np.concatenate(
    (X_train, bow[:k, :]),
    axis=1
)
X_test = np.concatenate(
    (X_test, bow[k:, :]),
    axis=1
)

# Print the dimensions
print('Dimensions of training set vectors:')
print('X: ', X_train.shape)
print('Y: ', Y_train.shape)

print('\nDimensions of test set vectors:')
print('X: ', X_test.shape)
print('Y: ', Y_test.shape)

# Visualizing the distribution of helpfulness score across the data
plt.hist(Y_train, ec='black')
plt.xlabel('Helpfulness Score')
plt.show()

# Create  a Linear Regression Model
linear_model = LinearRegression(normalize=True)
linear_model.fit(X_train, Y_train)

Y_pred = linear_model.predict(X_test)

print('R^2 score: ', linear_model.score(X_test, Y_test))
