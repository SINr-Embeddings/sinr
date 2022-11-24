# Adapted from glove-python originally written by Maciej Kula 
# https://github.com/maciejkula
# Cooccurrence matrix construction tools
# for fitting the GloVe model.
import numpy as np
import itertools
try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

from .cooccurrence_cython import construct_cooccurrence_matrix


class Cooccurrence(object):
    """
    Class for constructing a cooccurrence matrix
    from a corpus.

    A dictionnary mapping the vocabulary of the corpus in lexicographic order
    will be constructed as well as the cooccurrence matrix.

     """

    def __init__(self):

        self.dictionary = {}
        self.matrix = None

    def fit(self, corpus, window=2):
        """
        Perform a pass through the corpus to construct
        the cooccurrence matrix.

        :param corpus: List of lists of strings (words) from the corpus.
        :type corpus: `list(list())``
        :param window: The length of the (symmetric) context window used for cooccurrence, defaults to 2
        :type window: `int`
        """
        words_sorted = sorted(set(itertools.chain(*corpus)))
        self.dictionary   = dict(zip(words_sorted, range(len(words_sorted))))
        del words_sorted
        self.matrix = construct_cooccurrence_matrix(corpus,
                                                    self.dictionary,
                                                    int(window))

    def save(self, filename):
        """
        Save cooccurrence object to a pickle file.

        :param filename: Output path to the filename of the pickle file.
        :type filename: `str`
        """
        with open(filename, 'wb') as savefile:
            pickle.dump((self.dictionary, self.matrix),
                        savefile,
                        protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """
        Load Cooccurrence object from pickle.
        
        :param filename: Path to the pickle file.
        :type filename: str
        :return: An instance of the :class: Cooccurrence.
        :rtype: `community2vec.Cooccurrence`
        """
        instance = cls()

        with open(filename, 'rb') as savefile:
            instance.dictionary, instance.matrix = pickle.load(savefile)

        return instance
