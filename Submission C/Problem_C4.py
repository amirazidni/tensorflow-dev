# =====================================================================================================
# PROBLEM C4 
#
# Build and train a classifier for the sarcasm dataset. 
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# 
# Do not use lambda layers in your model.
# 
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop


def splitter(data, training_portion):
    train = data[:training_portion]
    test = data[training_portion:]
    return train, test

def solution_C4():
    data_url = 'https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000


    with open('sarcasm.json', 'r') as f:
        data = f.readlines()
        data = list(map(json.loads, data))
        datastore = pd.DataFrame(data)

    # Splitting data
    train_labels, validation_labels = splitter(datastore['is_sarcastic'], training_size)
    train_sentences, validation_sentences = splitter(datastore['headline'], training_size)
    training_labels_final = np.array(train_labels)
    testing_labels_final = np.array(validation_labels)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])

    sequences = tokenizer.texts_to_sequences(train_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
    testing_sequences = tokenizer.texts_to_sequences(validation_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

    model.fit(
        padded,
        training_labels_final,
        epochs=10,
        validation_data=(testing_padded, testing_labels_final),
        verbose=2
    )
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_C4()
    model.save("model_C4.h5")