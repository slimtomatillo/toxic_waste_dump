# Imports
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Logging
import logging
logging.getLogger().setLevel(logging.INFO)

def preprocess(data):
    """
    Function that takes in a pandas dataframe
    and performs preprocessing on the text,
    returning model-ready text.

    :param data: pandas.DataFrame

    :return: data: pandas.DataFrame
    """

    # View shape of data
    logging.info('Shape of data:')
    logging.info(data.shape)

    # Create new column of cleaned / preprocessed tokens
    data['model_text'] = data['comment_text'].apply(lambda row: clean_text(row).split())

    # Create new column of model-ready text in the form of a string
    data['model_text'] = data['model_text'].apply(lambda x: ' '.join(x))

    # Make clear that returned data is cleaned
    cleaned_data = data

    return(cleaned_data)

def clean_text(comment_string):
    """
    Function that takes in a comment in
    the form of a string, and preprocesses
    it, returning a clean string.

    Preprocessing includes:
    - lowercasing text
    - eliminating punctuation
    - dealing with edge case punctuation
      and formatting
    - replacing contractions with
      the proper full words
    - remove stopwords using gensim

    :param: comment_string: str

    :returns: cleaned_text: str
    """
    # Make text lowercase
    raw_text = comment_string.lower()

    # Get rid of newlines indicating contractions
    raw_text = raw_text.replace(r'\n', ' ')

    # Deal with anti- words
    raw_text = raw_text.replace('anti-', 'anti')

    # Standardize period formatting
    raw_text = raw_text.replace('.', ' ')

    # Replace exclamation point with space
    raw_text = raw_text.replace('!', ' ')

    # Replace slashes with space
    raw_text = raw_text.replace('/', ' ')

    # Replace questin marks with space
    raw_text = raw_text.replace('?', ' ')

    # Replace dashes with space
    raw_text = raw_text.replace('-', ' ')
    raw_text = raw_text.replace('—', ' ')

    # Replace ... with empty
    raw_text = raw_text.replace('…', '')
    raw_text = raw_text.replace('...', '')

    # Replace = with 'equals'
    raw_text = raw_text.replace('=', 'equals')

    # Replace commas with empty
    raw_text = raw_text.replace(',', '')

    # Replace ampersand with and
    raw_text = raw_text.replace('&', 'and')

    # Replace semi-colon with empty
    raw_text = raw_text.replace(';', '')

    # Replace colon with empty
    raw_text = raw_text.replace(':', '')

    # Get rid of brackets
    raw_text = raw_text.replace('[', '')
    raw_text = raw_text.replace(']', '')

    # Replace parentheses with empty
    raw_text = raw_text.replace('(', '')
    raw_text = raw_text.replace(')', '')

    # Replace symbols if they are by themselves
    raw_text = raw_text.replace(' $ ', ' ')
    raw_text = raw_text.replace(' ¢ ', ' ')
    raw_text = raw_text.replace(' @ ', ' ')
    raw_text = raw_text.replace(' # ', ' ')
    raw_text = raw_text.replace(' % ', ' ')
    raw_text = raw_text.replace(' * ', ' ')
    raw_text = raw_text.replace(' = ', ' ')
    raw_text = raw_text.replace(' + ', ' ')
    raw_text = raw_text.replace(' < ', ' ')
    raw_text = raw_text.replace(' > ', ' ')

    # Replace extra spaces with single space
    raw_text = raw_text.replace('   ', ' ')
    raw_text = raw_text.replace('  ', ' ')

    # Replace contractions with full words, organized alphabetically
    raw_text = raw_text.replace("can't", 'cannot')
    raw_text = raw_text.replace("couldn't", 'could not')
    raw_text = raw_text.replace("didn't", 'did not')
    raw_text = raw_text.replace("doesn't", 'does not')
    raw_text = raw_text.replace("don't", 'do not')
    raw_text = raw_text.replace("hasn't", 'has not')
    raw_text = raw_text.replace("he's", 'he is')
    raw_text = raw_text.replace("i'd", 'i would')
    raw_text = raw_text.replace("i'll", 'i will')
    raw_text = raw_text.replace("i'm", 'i am')
    raw_text = raw_text.replace("i've", 'i have')
    raw_text = raw_text.replace("isn't", 'is not')
    raw_text = raw_text.replace("it's", 'it is')
    raw_text = raw_text.replace("nobody's", 'nobody is')
    raw_text = raw_text.replace("she's", 'she is')
    raw_text = raw_text.replace("shouldn't", 'should not')
    raw_text = raw_text.replace("that'll", 'that will')
    raw_text = raw_text.replace("that's", 'that is')
    raw_text = raw_text.replace("there'd", 'there would')
    raw_text = raw_text.replace("they'd", 'they would')
    raw_text = raw_text.replace("they'll", 'they will')
    raw_text = raw_text.replace("they're", 'they are')
    raw_text = raw_text.replace("they've", 'they have')
    raw_text = raw_text.replace("there's", 'there are')
    raw_text = raw_text.replace("wasn't", 'was not')
    raw_text = raw_text.replace("we'd", 'we would')
    raw_text = raw_text.replace("we'll", 'we will')
    raw_text = raw_text.replace("we're", 'we are')
    raw_text = raw_text.replace("we've", 'we have')
    raw_text = raw_text.replace("won't", 'will not')
    raw_text = raw_text.replace("you'd", 'you would')
    raw_text = raw_text.replace("you'll", 'you will')
    raw_text = raw_text.replace("you're", 'you are')
    raw_text = raw_text.replace("you've", 'you have')

    # Fix other contractions / possessive
    raw_text = raw_text.replace("'s", '')

    # Replace quotes with nothing
    raw_text = raw_text.replace('“', '')
    raw_text = raw_text.replace('”', '')
    raw_text = raw_text.replace('"', '')
    raw_text = raw_text.replace("‘", "")
    raw_text = raw_text.replace("'", "")

    # Remove stopwords using gensim's remove_stopwords
    cleaned_text = remove_stopwords(raw_text)

    return(cleaned_text)

def prep_text(comments):
    """
    Function that takes in a comment
    and returns the text in model-ready
    form.

    :param comment: str
    :return: padded_comments: list
    """
    # Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(comments)

    # Integer encode the comments
    encoded_comments = tokenizer.texts_to_sequences(comments)

    # Pad the comments to the same length
    padded_comments = pad_sequences(encoded_comments, maxlen=300)

    return(padded_comments)
