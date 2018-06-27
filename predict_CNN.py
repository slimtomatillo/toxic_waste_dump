# Imports
import pandas as pd
import pickle
from keras.models import load_model
from preprocess import preprocess
from preprocess import prep_text

#Logging
import logging
logging.getLogger().setLevel(logging.INFO)

logging.info('Loading comments to classify...')

# Enter comment to be classified below
comment_to_classify = ''

def return_label(predicted_probs):
    """
    Function that takes in a list of 7 class
    probabilities and returns the labels
    with probabilities over a certain threshold.
    """
    threshold = 0.4
    labels = []
    classes = ['clean', 'toxic', 'severe toxic', 'obscene',
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
    data = pd.DataFrame(data=[comment_str], columns=['comment_text'])

    logging.info('Comments loaded.')

    # Preprocess text
    X_to_predict = preprocess(data)

    # Identify data to make predictions from
    X_to_predict = X_to_predict['model_text']

    # Format data properly
    X_to_predict = prep_text(X_to_predict)

    logging.info('Loading model...')

    # Load CNN from disk
    cnn = load_model('model/CNN/binarycrossentropy_adam/model-04-0.9781.hdf5')

    logging.info('Model loaded.')
    logging.info('Making prediction(s)...')

    # Make predictions
    preds = cnn.predict(X_to_predict)

    for each_comment, prob in zip(data['comment_text'], preds):
        print('COMMENT:')
        print(each_comment)
        print()
        print('PREDICTION:')
        print(return_label(prob))
        print()

    logging.info('Finished.')

predict_label(comment_to_classify)
