import numpy as np

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


# Textual Features
def extract_textual_features(data):
    # m: total number of training examples
    m = len(data['review'].values)

    # total number of characters
    text_length = np.array(data['review'].str.len()).reshape((m, 1))

    # total number of alphabetical characters
    character_count = np.array(
        data['review'].replace(regex=True, to_replace=r'[^a-zA-Z]', value=r'').str.len()
    ).reshape((m, 1))

    # Tokenized Sparse Matrix
    vectorizer = CountVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform(np.array(data['review'].values))
    matrix = np.array(matrix.todense())

    # the number of words in the review text
    word_count = np.sum(matrix, axis=1, keepdims=True)

    # the number of unique words in the review text
    unique_word_count = np.count_nonzero(matrix, axis=1).reshape((m, 1))

    # the number of sentences in the review text
    sentence_count = []
    for i in np.array(data['review'].values):
        s = sent_tokenize(i)
        sentence_count.append(len(s))
    sentence_count = np.array(sentence_count).reshape((m, 1))

    # Automated Readability Index
    ARI = 4.71 * (character_count / word_count) + 0.5 * (word_count / sentence_count)
    ARI = np.reshape(ARI, (m, 1))

    return np.concatenate(
        (text_length, character_count, word_count, unique_word_count, sentence_count, ARI),
        axis=1
    )


# Metadata Features
def extract_metadata_features(data):
    # m: total number of training examples
    m = len(data['overall'].values)

    # overall rating, the user gave to the product
    rating = np.array(data['overall'].values).reshape((m, 1))

    return rating


# Bag of Words
def create_bag_of_words(data):
    # Construct a bag of words matrix
    vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    matrix = vectorizer.fit_transform(np.array(data['review'].values))
    matrix = np.array(matrix.todense())

    return matrix
