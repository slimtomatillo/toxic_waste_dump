# Standard libraries
import pandas as pd
import numpy as np
import pickle

# Visualization
import matplotlib.pyplot as plt

# Other libraries
import gensim
import itertools
import time

# Keras
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint

# Logging
import logging
logging.getLogger().setLevel(logging.INFO)

logging.info('Loading data...')

# Load in data
data = pd.read_pickle('data_cleaned.pkl')

logging.info('Data loaded.')

# Set parameters
num_epochs = 8

# Create vocabulary
vocabulary = set(itertools.chain.from_iterable(data['cleaned']))
vocab_size = len(vocabulary)

logging.info('Vocab size:')
logging.info(vocab_size)
logging.info('Preprocessing text...')

# Choose X
X = data['model_text']

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

# Integer encode the comments
encoded_comments = tokenizer.texts_to_sequences(X)

# Dictionary of word to token number
word_index = tokenizer.word_index

# Vocabulary size
vocab_size = len(word_index) + 1

# Pad the comments to the same length
padded_comments = pad_sequences(encoded_comments, maxlen=300)
input_length = padded_comments.shape[1]

logging.info('Encoding classes...')

# Encode target classes
labels = ['clean', 'toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate']
mlb = MultiLabelBinarizer(classes=labels)

# Use label encoder to encode y
y = mlb.fit_transform(data['target_label'])

logging.info('Classes encoded.')
logging.info('Pickling MultiLabelBinarizer...')

# Save MultiLabelBinarizer model for making predictions later
pickle.dump(mlb, open('label_encoder.pkl', 'wb'))

logging.info('MultiLabelBinarizer pickled.')
logging.info('Classes:')
logging.info(mlb.classes_)

# Import Word2Vec
word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Create a weight matrix for words in training docs
embedding_weights = np.zeros((vocab_size, 300))
for word, index in word_index.items():
    embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(300)

# Create callbacks
filepath = 'model/model-{epoch:02d}-{val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='True'
)

callbacks_list = [checkpoint]

logging.info('Creating model...')

# Create model
model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[embedding_weights], input_length=input_length, trainable=False))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(7, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

logging.info('Training model...')

# Fit model
start = time.time()
model_info = model.fit(padded_comments, y, validation_split=0.3, epochs=num_epochs, callbacks=callbacks_list)
end = time.time()

logging.info('Finished.')

# Print how long it took to train the model
logging.info("Model took %0.2f seconds to train" %(end - start))

# Plot model accuracy and loss
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Summarize history of accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy', fontsize=20)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['Train', 'Validation'], loc='best')

    # Summarize history of loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss', fontsize=20)
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['Train', 'Validation'], loc='best')
    plt.show()
    fig.savefig('Model_History.png')

# Plot model history
plot_model_history(model_info)
