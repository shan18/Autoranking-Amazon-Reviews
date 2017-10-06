import pandas as pd

from get_data import parse_input

INPUT_FILE = 'reviews_Video_Games_5.json.gz'

df = parse_input('reviews_Video_Games_5.json.gz')
print(df['helpful'][0])
