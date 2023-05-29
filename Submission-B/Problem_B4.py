# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

def splitter(data, training_portion):
    train = data[:training_portion]
    test = data[training_portion:]
    return train, test

def solution_B4():
    bbc = pd.read_csv('https://academy.blob.core.windows.net/picodiploma/Simulation/machine_learning/bbc-text.csv')
    
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8

    train_size = int(len(bbc) * training_portion)

    # Splitting data
    train_labels, validation_labels = splitter(bbc['category'], train_size)
    train_sentences, validation_sentences = splitter(bbc['text'], train_size)

    tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    x_train = tokenizer.texts_to_matrix(train_sentences)
    x_test = tokenizer.texts_to_matrix(validation_sentences)

    encoder = LabelEncoder()
    encoder.fit(train_labels)
    y_train = encoder.transform(train_labels)
    y_test = encoder.transform(validation_labels)

    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(vocab_size,)),
        tf.keras.layers.Dense(5, activation='softmax'),
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=2)

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
    model = solution_B4()
    model.save("model_B4.h5")
