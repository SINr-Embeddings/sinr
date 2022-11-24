#!/usr/bin/env python

"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_nfm
"""


import unittest

from .. import nfm
from scipy.sparse import csr_matrix, coo_matrix
import networkit as nk


class TestSinr_embeddings(unittest.TestCase):
    """Tests for `sinr_embeddings` package."""
    

    def setUp(self):
        """Set up test fixtures, if any."""
        G = nk.Graph(5)
        G.addEdge(0,1)
        G.addEdge(0,2)
        G.addEdge(1,2)
        G.addEdge(2,3)
        G.addEdge(3,4)
        self.G = G
        self.adjacency = nk.algebraic.adjacencyMatrix(self.G, matrixType='sparse')
        self.vector=[0,0,0,1,1]
        self.membership_matrix = nfm.get_membership(self.vector)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_nnz(self):
        self.assertEqual(self.adjacency.nnz, 10)
    
    def test_adjacency(self):
        ref = csr_matrix([[0,1,1,0,0],[1,0,1,0,0],[1,1,0,1,0],[0,0,1,0,1], [0,0,0,1,0]])
        self.assertEqual((self.adjacency != ref).nnz, 0)
        
    def test_membership(self):
        ref = coo_matrix([[1,0],[1,0],[1,0],[0,1],[0,1]])
        self.assertEqual((self.membership_matrix != ref).nnz, 0)
        
    def test_nr(self):
        nr = nfm.compute_NR(self.adjacency, self.membership_matrix)
        for idx, val in enumerate(nr.data):
            nr.data[idx] = round(val, 2)
        ref = csr_matrix([[1,0],[1,0],[0.67, 0.33], [0.5, 0.5], [0,1]])
        self.assertEqual((nr != ref).nnz, 0)
        
    def test_community_weights(self):
        weights = nfm.get_community_weights(self.adjacency, self.membership_matrix)
        self.assertEqual(weights.item(0,0), 7)
        self.assertEqual(weights.item(0,1), 3)
        
    def test_np(self):
        np = nfm.compute_NP(self.adjacency, self.membership_matrix)
        #print(np.todense())
        for idx, val in enumerate(np.data):
            np.data[idx] = round(val, 2)
        ref = csr_matrix([[round(2/7, 2),0],[round(2/7, 2),0],[round(2/7, 2), round(1/3, 2)], 
                          [round(1/7, 2), round(1/3, 2)], [0,round(1/3, 2)]])
        self.assertEqual((np != ref).nnz, 0)


if __name__ == '__main__':
    unittest.main()