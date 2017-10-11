import nltk
import matplotlib.pyplot as plt

from get_data import parse_input
from preprocess_filter_data import preprocess_input, filter_input
from create_input_output_vectors import get_xy_vectors

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

# X_train: Input vector for the training set
# Y_train: Vector containing the results of the training set
X_train, Y_train = get_xy_vectors(df_train)

# Print the dimensions
print('Dimensions of X: ', X_train.shape)
print('Dimensions of Y: ', Y_train.shape)

# Visualizing the distribution of helpfulness score across the data
plt.hist(Y_train, ec='black')
plt.xlabel('Helpfulness Score')
plt.show()
