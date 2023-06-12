"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_sparsify
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

        row = np.array([2, 0, 2, 0])
        col = np.array([1, 0, 3, 3])
        data = np.array([5, 6, 8, 3])

        v0 = ge.SINrVectors('')
        v10 = ge.SINrVectors('')
        v2 = ge.SINrVectors('')
        
        v0.set_vectors(mat_test)
        v10.set_vectors(mat_test)
        v2.set_vectors(mat_test) 
        
        res10 = mat_test
        res2 = csr_matrix((data, (row, col)), shape=(3, 4))
        
        self.v0 = v0
        self.v10 = v10
        self.v2 = v2

        self.res10 = res10
        self.res2 = res2
        

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_sparsify(self):
        self.v0.sparsify(0)
        self.v10.sparsify(10)
        self.v2.sparsify(2)

        #self.assertTrue((arr1 == arr2).all())        # k = 0
        self.assertTrue(self.v0.vectors.nnz == 0)
        # k > number of dimensions
        self.assertTrue((self.v10.vectors.todense() == self.res10.todense()).all())
        # 0 < k < number of dimensions
        self.assertTrue((self.v2.vectors.todense() == self.res2.todense()).all())

if __name__ == '__main__':
    unittest.main()