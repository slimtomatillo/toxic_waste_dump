# Imports
import pandas as pd
import pickle
from preprocess import preprocess
from preprocess import prep_text
from sklearn.feature_extraction.text import TfidfVectorizer

#Logging
import logging
logging.getLogger().setLevel(logging.INFO)

logging.info('Loading comments to classify...')

subreddits = ['AskWomen_10575.pkl', 'BlackPeopleTwitter_1293.pkl',
              'BullyingIsCool_121.pkl', 'Drugs_4329.pkl',
              'Feminism_3713.pkl', 'Fitness_10799.pkl',
              'ImGoingToHellForThis_12328.pkl', 'KotakuInAction_13109.pkl',
              'MensRights_9068.pkl', 'MGTOW_3034.pkl',
              'sex_8976.pkl', 'TheRedPill_9646.pkl']

reddit_comments = []

# Load in data
for subreddit in subreddits:
    curr_data = pd.read_pickle(f'reddit/{subreddit}')
    reddit_comments += curr_data

logging.info('Total number of comments:')
logging.info(len(reddit_comments))

def return_label(predicted_probs):
    """
    Function that takes in a list of 7 class
    probabilities and returns the labels
    with probabilities over a certain threshold.
    """
    threshold = 0.4
    labels = []
    classes = ['not toxic', 'toxic', 'severe toxic', 'obscene',
               'threat', 'insult', 'identity hate']

    i = 0
    while i < len(classes):
        if predicted_probs[i] > threshold:
            labels.append(classes[i])
        i += 1

    return (labels)


def predict_label(comment_str):
    """
    Function that takes in a comment in
    string form and returns the predicted
    class labels: not toxic, toxic, severe
    toxic, obscene, threat, insults, identity
    hate. May output multiple labels.
    """
    data = pd.DataFrame(data=comment_str, columns=['comment_text'])

    logging.info('Comments loaded.')
    logging.info('Preprocessing comments...')

    # Preprocess text
    X_to_predict = preprocess(data)

    # Identify data to make predictions from
    X_to_predict = X_to_predict['model_text']

    logging.info('Vectorizing comments...')

    # Load TF-IDF Vectorizer
    TF_IDF = pickle.load(open('model/TF_IDF/TF_IDF.pkl', 'rb'))

    #print(X_to_predict)

    # Fit / transform vectorizer
    X_to_predict = TF_IDF.transform(X_to_predict.values.astype('U'))

    logging.info('Comments ready for model.')
    logging.info('Loading model...')

    # Load OvR model from disk
    OvR_lr = pickle.load(open('model/OvR/OvR_lr.pkl', 'rb'))

    # Load MultiLableBinarizer from disk
    mlb = pickle.load(open('label_encoder.pkl', 'rb'))

    logging.info('Model loaded.')
    logging.info('Making prediction(s)...')

    # Make predictions
    preds = OvR_lr.predict(X_to_predict)

    # View comments and predicted labels
    for item, target in zip(data['comment_text'], mlb.inverse_transform(preds)):
        print('COMMENT:')
        print(item)
        print()
        print('LABEL:')
        print(target)
        print()

    logging.info('Finished.')

predict_label(reddit_comments)
