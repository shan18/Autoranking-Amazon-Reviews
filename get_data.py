"""
This module loads the data in json file into a data frame.
"""

import gzip
import pandas as pd


# Extract and read the input file line by line
def read_input_file(path):
    g = gzip.open(path, 'rb')
    for line in g:
        yield eval(line)


# Store the data in input file in a dataframe
def parse_input(path):
    i = 0
    df = {}
    for d in read_input_file(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
