import numpy as np

from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


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

    # the number of words, unique words, and sentences in the review text
    sentence_count, word_count, unique_word_count = [], [], []
    for review in data['review'].values:
        text_blob = TextBlob(review)
        word_count.append(len(text_blob.words))
        unique_word_count.append(len(set(text_blob.words)))
        sentence_count.append(len(text_blob.sentences))
    word_count = np.array(word_count).reshape((m, 1))
    unique_word_count = np.array(unique_word_count).reshape((m, 1))
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


# Create tf-idf representation of the review text
def create_tf_idf_vector(data):
    # Construct a tf-idf matrix
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    matrix = vectorizer.fit_transform(data['review'].values)

    return matrix.todense()


# Create bag of words
def create_bag_of_words(data):
    # Construct a bag of words matrix
    vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=3000)
    matrix = vectorizer.fit_transform(np.array(data['review'].values))

    return matrix.todense()
