from get_data import parse_input
from preprocess_filter_data import preprocess_input, filter_input


INPUT_FILE = 'reviews_Video_Games_5.json.gz'

# Store the input in a dataframe
df = parse_input(INPUT_FILE)
print(df.head())

# Preprocessing and Filtering the dataset
df = preprocess_input(filter_input(df))
print('\n\nPreprocessed and Filtered data:\n')
print(df.head())
