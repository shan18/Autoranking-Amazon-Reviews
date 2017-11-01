import numpy as np


# Preprocessing the input
def preprocess_input(data):

    # Merge summary and reviewText
    data['review'] = data['summary'] + ' ' + data['reviewText']

    # Separate out upvotes and downvotes from 'helpful'
    votes = list(zip(*list(data['helpful'].values)))
    data['upvotes'] = np.array(votes[0])
    data['downvotes'] = np.array(votes[1])

    # Remove unnecessary features
    del data['reviewTime'], data['unixReviewTime'], data['reviewerName'], data['summary'], data['reviewText'], data[
        'helpful']

    # Rearrange columns
    data = data[['reviewerID', 'asin', 'overall', 'upvotes', 'downvotes', 'review']]

    return data


# Filtering the dataset
def filter_input(data):

    # (1) Review should have atleast 5 votes
    data.drop(data[data.upvotes + data.downvotes <= 5].index, inplace=True)

    # (2) Each product should have more than 7 reviews
    product_review_count = data['asin'].value_counts()
    unpopular_products = product_review_count[product_review_count <= 7].index
    data.drop(data[data['asin'].isin(unpopular_products)].index, inplace=True)

    return data
