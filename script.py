import json

from preprocess_input import format_input


INPUT_FILE = 'Video_Games_Reviews.json'
PREPROCESSED_INPUT_FILE = 'video_games_reviews.json'

# Preprocess the input file
format_input(INPUT_FILE, PREPROCESSED_INPUT_FILE)

# Load the preprocessed input file in a list
with open(PREPROCESSED_INPUT_FILE) as data_file:
    data = json.load(data_file)
