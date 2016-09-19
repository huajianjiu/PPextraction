import numpy as np
import os.path
from nltk.corpus import wordnet as wn
import cPickle as pickle

from WordnetNet import remove_duplicate, remove_not_in_and_get_word_id

class PPDB(object):
    def __init__(self, relations_num=1, vocab="vocab.txt"):
        pass