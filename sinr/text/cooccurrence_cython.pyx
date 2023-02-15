#!python
# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
import scipy.sparse as sp
import itertools

from libc.stdlib cimport malloc, free

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector


cdef inline int int_min(int a, int b) nogil: return a if a <= b else b


cdef int binary_search(int* vec, int size, int first, int last, int x) nogil:
    """
    Binary seach in an array of ints
    """

    cdef int mid

    while (first < last):
        mid = (first + last) / 2
        if (vec[mid] == x):
            return mid
        elif vec[mid] > x:
            last = mid - 1
        else:
            first = mid + 1

    if (first == size):
        return first
    elif vec[first] > x:
        return first
    else:
        return first + 1


cdef struct SparseRowMatrix:
    vector[vector[int]] *indices
    vector[vector[float]] *data


cdef SparseRowMatrix* new_matrix():
    """
    Allocate and initialize a new matrix
    """

    cdef SparseRowMatrix* mat

    mat = <SparseRowMatrix*>malloc(sizeof(SparseRowMatrix))

    if mat == NULL:
        raise MemoryError()

    mat.indices = new vector[vector[int]]()
    mat.data = new vector[vector[float]]()

    return mat


cdef void free_matrix(SparseRowMatrix* mat) nogil:
    """
    Deallocate the data of a matrix

    :param mat: A matrix.
    :type mat: SparseRowMatrix*

    """

    cdef int i
    cdef int rows = mat.indices.size()

    for i in range(rows):
        deref(mat.indices)[i].clear()
        deref(mat.data)[i].clear()

    del mat.indices
    del mat.data

    free(mat)


cdef void increment_matrix(SparseRowMatrix* mat, int row, int col, float increment) nogil:
    """
    Increment the (row, col) entry of mat by increment.

    :param mat: A matrix.
    :type mat: SparseRowMatrix*
    :param row: A row.
    :type row: int
    """

    cdef vector[int]* row_indices
    cdef vector[float]* row_data
    cdef int idx
    cdef int col_at_idx

    # Add new row if necessary
    while row >= mat.indices.size():
        mat.indices.push_back(vector[int]())
        mat.data.push_back(vector[float]())

    row_indices = &(deref(mat.indices)[row])
    row_data = &(deref(mat.data)[row])

    # Find the column element, or the position where
    # a new element should be inserted
    if row_indices.size() == 0:
        idx = 0
    else:
        idx = binary_search(&(deref(row_indices)[0]), row_indices.size(),
                            0, row_indices.size(), col)

    # Element to be added at the end
    if idx == row_indices.size():
        row_indices.insert(row_indices.begin() + idx, col)
        row_data.insert(row_data.begin() + idx, increment)
        return

    col_at_idx = deref(row_indices)[idx]

    if col_at_idx == col:
        # Element to be incremented
        deref(row_data)[idx] = deref(row_data)[idx] + increment
    else:
        # Element to be inserted
        row_indices.insert(row_indices.begin() + idx, col)
        row_data.insert(row_data.begin() + idx, increment)


cdef int matrix_nnz(SparseRowMatrix* mat) nogil:
    """
    Get the number of nonzero entries in mat

    :param mat: A matrix.
    :type mat: SparseRowMatrix*
    """

    cdef int i
    cdef int size = 0

    for i in range(mat.indices.size()):
        size += deref(mat.indices)[i].size()

    return size


cdef matrix_to_coo(SparseRowMatrix* mat, int shape):
    """
    Convert to a shape by shape COO matrix.

    :param mat: A matrix.
    :type mat: SparseRowMatrix*
    :param shape: The shape (size of the vocabulary) of the matrix.
    :type shape: int

    :return: A matrix in scipy.sparse.coo_matrix format
    :rtype: scipy.sparse.coo_matrix
    """

    cdef int i, j
    cdef int row
    cdef int col
    cdef int rows = mat.indices.size()
    cdef int no_collocations = matrix_nnz(mat)

    # Create the constituent numpy arrays.
    row_np = np.empty(no_collocations, dtype=np.int32)
    col_np = np.empty(no_collocations, dtype=np.int32)
    data_np = np.empty(no_collocations, dtype=np.float64)
    cdef int[:] row_view = row_np
    cdef int[:] col_view = col_np
    cdef double[:] data_view = data_np

    j = 0

    for row in range(rows):
        for i in range(deref(mat.indices)[row].size()):

            row_view[j] = row
            col_view[j] = deref(mat.indices)[row][i]
            data_view[j] = deref(mat.data)[row][i]

            j += 1

    # Create and return the matrix.
    return sp.coo_matrix((data_np, (row_np, col_np)),
                         shape=(shape,
                                shape),
                         dtype=np.float64)


cdef int words_to_ids(list words, vector[int]& word_ids,
                      dictionary):
    """
    Convert a list of words into a vector of word ids, using either
    the supplied dictionary or by consructing a new one.

    If the dictionary was supplied, a word is missing from it,
    and we are not ignoring out-of-vocabulary (OOV) words, an
    error value of -1 is returned.

    If we have an OOV word and we do want to ignore them, we use
    a -1 placeholder for it in the word_ids vector to preserve
    correct context windows (otherwise words that are far apart
    with the full vocabulary could become close together with a
    filtered vocabulary).

    :param words: List of words (sentence).
    :type words: list[str]
    :param word_ids: An empty vector whose function is to store the ids of the words in the sentence
    :type word_ids: vector[int]
    :param dictionary: A dictionary mapping of word to their ids in the vocabulary (in lexicographic order).
    :type dictionary: dict

    :return: Value 0 if success else value -1
    :rtype: int
    """

    cdef int word_id

    word_ids.resize(0)
    
    
    for word in words:
       # Raise an error if the word
       # is missing from the supplied
       # dictionary.
        word_id = dictionary.get(word, -1)
        if word_id == -1:
            return -1

        word_ids.push_back(word_id)

    return 0


def construct_cooccurrence_matrix(corpus, dictionary, int window_size):
    """Construct the word-id dictionary and cooccurrence matrix for
    a given corpus, using a given window size.
    
    The dictionary is constructed by lexicographix order.
    The matrix accounts for the number of cooccurrences of words, no matter the
    order in which they appear in the corpus. Consequently, the cooccurrence
    matrix is upper triangular (undirected graph in SINr) .

    :param corpus: The sentences from the corpus.
    :type corpus: list[str]
    :param dictionary: A dictionary mapping of words to their ids in the vocabulary (in lexicographic order)
    :type dictionary: dictionary: dict[str:int]
    :param window_size: The size of the symmetric moving window.
    :type window_size: int
    :param int window_size: 
    :returns: The cooccurrence matrix built from the corpus.
    :rtype: scipy.sparse.coo

    """

    # Declare the cooccurrence map
    cdef SparseRowMatrix* matrix = new_matrix()

    # String processing variables.
    cdef list words
    cdef int i, j, outer_word, inner_word
    cdef int wordslen, window_stop, error
    cdef vector[int] word_ids

    # Pre-allocate some reasonable size
    # for the word ids vector.
    word_ids.reserve(1000)

    #Addition to use a lexicographic order dictionary and get a top right triangular matrix

    # Iterate over the corpus.
    for words in corpus:

        # Convert words to a numeric vector.
        error = words_to_ids(words, word_ids, dictionary)
        
        wordslen = word_ids.size()

        # Record co-occurrences in a moving window.
        for i in range(wordslen):
            outer_word = word_ids[i]

            # Continue if we have an OOD token.
            if outer_word == -1:
                continue

            window_stop = int_min(i + window_size + 1, wordslen)

            for j in range(i, window_stop):
                inner_word = word_ids[j]

                if inner_word == -1:
                    continue

                # Do nothing if the words are the same.
                if inner_word == outer_word:
                    continue
                else:
                # Each cooccurrence is counted as 1, cooccurrence(w_1, w_2) == cooccurrence(w_2, w_1) because the graph is undirected in SINr, subsequently, the matrix is upper triangular with cooccurrence(w_i, w_i) == 0
                    if inner_word < outer_word:
                        increment_matrix(matrix,
                                         inner_word,
                                         outer_word,
                                         2) # We increment the matrix by 2 because we record the cooccurrence of w_1 w_2 and the cooccurrence of w_2 w_1
                    else:
                        increment_matrix(matrix,
                                         outer_word,
                                         inner_word,
                                         2)
    mat = matrix_to_coo(matrix, len(dictionary))
    free_matrix(matrix)

    return mat
