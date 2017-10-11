import numpy as np

from extract_features import extract_textual_features, extract_metadata_features, create_bag_of_words


# Create the feature vector and the result vector
def get_xy_vectors(data):
    # Constructing feature vector
    X = np.concatenate(
        (extract_textual_features(data), extract_metadata_features(data), create_bag_of_words(data)),
        axis=1
    )

    # m: total number of training examples
    m = X.shape[0]

    # Set the outcome variable (Denotes helpfulness of a review)
    upvotes = np.array(data['upvotes'].values).reshape((m, 1))
    downvotes = np.array(data['downvotes'].values).reshape((m, 1))
    Y = upvotes / (upvotes + downvotes)

    return X, Y
