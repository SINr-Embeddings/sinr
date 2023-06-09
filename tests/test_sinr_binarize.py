"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_binarize
"""

import pytest
import unittest

import numpy as np
import sinr.graph_embeddings as ge
from scipy.sparse import csr_matrix


class TestSinr_embeddings(unittest.TestCase):
    """Tests for `graph_embeddings` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

        row = np.array([0, 2, 2, 0, 2, 2, 0])
        col = np.array([2, 1, 0, 0, 2, 3, 3])
        data = np.array([2, 5, 4, 6, 1, 8, 3])
        mat_test = csr_matrix((data, (row, col)), shape=(3, 4))

        data = np.array([1, 1, 1, 1, 1, 1, 1])
        res = csr_matrix((data, (row, col)), shape=(3, 4))

        vec = ge.SINrVectors('') 
        vec.set_vectors(mat_test)
        
        self.sinr_vec = vec
        self.res = res

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_binarize(self):
        self.sinr_vec.binarize()
        self.assertTrue((self.sinr_vec.vectors.todense() == self.res.todense()).all())

if __name__ == '__main__':
    unittest.main()