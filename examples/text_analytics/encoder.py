""" This script contains the `Encoder` class,
which makes index transformations for text
classification tasks with DeepSeries. 
Alphabet characters <-> integer ids <-> one hot  
"""


import string
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Encoder:

    def __init__(self, corpus):
        # Corpus is the string containing the entire text.
        # Vocabulary is a list of unique characters
        # (pretty much the alphabet plus " ")
        vocabulary = set(corpus)
        self.char2id = dict(zip(vocabulary, range(len(vocabulary))))
        self.id2char = dict(zip(range(len(vocabulary)), vocabulary))
        # convert full corpus to integer ids
        self.ids = np.array([self.char2id[char] for char in corpus])
        # Get the corpus in one-hot form
        self.fit_onehot()
        self.onehot_corpus = self.ids2onehot(self.ids)

    def fit_onehot(self):
        # one-hot encode
        self.enc = OneHotEncoder()
        self.enc.fit([[l] for l in set(self.ids)])

    def ids2onehot(self, id_list):
        id_array = np.array(id_list)
        onehot_labels = self.enc.transform(id_array.reshape(-1, 1)).toarray()
        return onehot_labels

    def onehot2ids(self, onehot_array):
        ''' this works too for transforming 
        softmax probs to ids '''
        id_array = np.argmax(onehot_array, 2)
        return id_array

    def ids2chars(self, id_array):
        chars = np.array([string.join([self.id2char[i]
                                       for i in x], '') for x in id_array])
        return chars

    def onehot2chars(self, onehot_array):
        ids = self.onehot2ids(onehot_array)
        return self.ids2chars(ids)

    def freestyle2onehot(self, sentence):
        ''' Converts a text string into one-hot encoding '''
        ids = np.array([self.char2id[char] for char in sentence])
        return np.array([self.ids2onehot(ids)])

    def talk(self, ds, lead_string, n_output):
        ''' Wrapper that takes a trained deepseries.Sequencer 
        instance and a custom lead string, and returns 
        the predicted output text'''
        enc_ = self.freestyle2onehot(lead_string)
        pred_ = ds.unroll(enc_, n_output)
        return self.onehot2chars(pred_)[0]
