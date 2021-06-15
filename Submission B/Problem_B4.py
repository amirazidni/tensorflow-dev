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
import tensorflow as tf
import pandas as pd


def solution_B4():
    bbc = pd.read_csv('https://dicodingacademy.blob.core.windows.net/picodiploma/Simulation/machine_learning/bbc-text.csv')
    
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE

    tokenizer =  # YOUR CODE HERE

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
    model = solution_B4()
    model.save("model_B4.h5")
