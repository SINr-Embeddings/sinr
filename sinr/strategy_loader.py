# -*- coding: utf-8 -*-
from .logger import logger
import pickle as pk
import networkit as nk

def load_pkl_text(mat_path):
    """
    Load a cooccurrence matrix.

    :param cooc_mat_path: Path to coocurrence matrix.

    :type cooc_mat_path : str

    :return: The loaded cooccurrence matrix and the word index.

    :rtype: `tuple(dict(), scipy.sparse.coo_matrix)`
    """
    logger.info("Loading cooccurrence matrix and dictionary.")
    with open(mat_path, "rb") as file:
        word_to_idx, mat = pk.load(file)
    logger.info("Finished loading data.")
    return (word_to_idx, mat)

from scipy.sparse import coo_matrix
def load_adj_mat(matrix, labels=None):
    """
    Load a cooccurrence matrix.

    :param matrix: an adjacency matrix
    :param matrix: a dict matching labels with nodes

    :type matrix : csr_matrix
    :type labels : dict: string -> int

    :return: The loaded cooccurrence matrix and the word index.

    :rtype: `tuple(dict(), scipy.sparse.coo_matrix)`
    """
    if labels == None:
        labels = dict()
        for i in range(matrix.shape[0]):
            labels[i] = i
    return (labels, coo_matrix(matrix))