#Adapted from 
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import dok_matrix, triu, tril
from sklearn.metrics import pairwise_distances


def _as_diag(px, alpha):
    """
    Produces diagonal matrices from probabilities of words.

    :param px: Probability for each word.
    :type px: `numpy.ndarray`
    :param alpha: Smoothing factor.
    :type alpha: `float`

    :rtype: `scipy.sparse.diags`
    :return: A diagonal matrix of probabilities for px.
    """
    px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray([0 if v == 0 else 1 / (v + alpha) for v in px_diag.data[0]])
    return px_diag


def _logarithm_and_ppmi(exp_pmi, min_exp_pmi):
    """
    Applies logarithm and ppmi to exponential PMI matrix.

    :param exp_pmi: Exponential PMI values.
    :type exp_pmi: `scipy.csr_matrix`
    :param min_exp_pmi: Threshold for minimal PMI value.
    :type min_exp_pmi: `int`

    :rtype: `scipy.csr_matrix`
    :return: PMI matrix after applying logarithm and excluding values lower than min_exp_pmi.
    """
    n, m = exp_pmi.shape

    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    rows, cols = exp_pmi.nonzero()
    data = exp_pmi.data

    indices = np.where(data >= min_exp_pmi)[0]
    rows = rows[indices]
    cols = cols[indices]
    data = data[indices]

    # apply logarithm
    data = np.log(data)

    # new matrix
    exp_pmi_ = csr_matrix((data, (rows, cols)), shape=(n, m))
    return exp_pmi_

def pmi(X, py=None, min_pmi=0, alpha=0.0, beta=1):
    """
    :param X:  (word, word) sparse matrix
    :type X: `scipy.sparse.csr_matrix`
    :param py: (1, word) shape, probability of context words.
    :type py: `numpy.ndarray`
    :param min_pmi: Minimum value of PMI. all the values that smaller than min_pmi
        are reset to zero, defaults to 0
    :type min_pmi: `int`
    :param alpha: Smoothing factor. pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha)),
        defaults to  0.0
    :type alpha: `float`
    :param beta: Smoothing factor. pmi(x,y) = log ( Pxy / (Px x Py^beta) ),
        defaults to 1.0
    :type beta: `int`

    :return: A dictionary containing the PPMI matrix, the probability of words
    and the exponential PMI matrix '(pmi, px, py, exp_pmi)' . 
    (word, word) pmi value sparse matrix  if beta > 1 or beta < 0:
        raise ValueError("beta value {} is not in range ]0,1]".format(beta))
    if beta > 1 or beta < 0:
        raise ValueError("beta value {} is not in range ]0,1]".format(beta))

    :rtype: `list(scipy.sparse.csr_matrix, numpy.ndarray, numpy.ndarray, scipy.sparse.csr_matrix)`
    """

    assert 0 < beta <= 1
    # convert x to probability matrix & marginal probability
    px = np.asarray(((X.sum(axis=0)+(X.sum(axis=1).T)) / X.sum()).reshape(-1))
    if py is None:
        py = px
    if beta < 1:
        py = py ** beta
        py /= py.sum()
    pxy = X / X.sum()

    # transform px and py to diagonal matrix
    # using scipy.sparse.diags
    # pmi_alpha (x,y) = p(x,y) / ( p(x) x (p(y) + alpha) )
    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)
    pmi = _logarithm_and_ppmi(exp_pmi, min_exp_pmi)

    return pmi, px, py, exp_pmi


def pmi_filter(X, py=None, min_pmi=0, alpha=0.0, beta=1):
    """
    Filter a matrix (word, word) by computing the PMI. Exclude the records for which
    the PMI is lower than a thershold `min_pmi`.
    
    :param X:  (word, word) sparse matrix
    :type X: `scipy.sparse.csr_matrix`
    :param py: (1, word) shape, probability of context words.
    :type py: `numpy.ndarray`
    :param min_pmi: Minimum value of PMI. all the values that smaller than min_pmi
        are reset to zero, defaults to 0
    :type min_pmi: `int`
    :param alpha: Smoothing factor. pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha)),
        defaults to  0.0
    :type alpha: `float`
    :param beta: Smoothing factor. pmi(x,y) = log ( Pxy / (Px x Py^beta) ),
        defaults to 1.0
    :type beta: `int`

    :return: A dictionary containing the PPMI matrix, the probability of words
    and the exponential PMI matrix '(pmi, px, py, exp_pmi)' . 
    (word, word) pmi value sparse matrix  if beta > 1 or beta < 0:
        raise ValueError("beta value {} is not in range ]0,1]".format(beta))
    if beta > 1 or beta < 0:
        raise ValueError("beta value {} is not in range ]0,1]".format(beta))

    :rtype: `scipy.sparse.coo_matrix`

    """
    shape_X = X.shape
    pmi_X, _, _, _ = pmi(X, py, min_pmi, alpha, beta)
    pmi_X = pmi_X.tolil()
    filtering_mat = pmi_X > 0
    X = X.tolil()
    X = X.multiply(filtering_mat)
    assert shape_X == X.shape, "The shape of the matrix before and after pmi normalisation must be the same."
    #upper_tri = triu(X)
    #lower_tri = tril(X)
    #X = upper_tri + lower_tri.T
    return X.tocoo()


