"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_dim_filter
"""

import pytest
import unittest

import numpy as np
import sinr.graph_embeddings as ge
from scipy.sparse import csr_matrix
import networkit as nk


class TestSinr_embeddings(unittest.TestCase):
    """Tests for `graph_embeddings` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

        G = nk.Graph(5)
        G.addEdge(1, 3)
        G.addEdge(2, 4)
        G.addEdge(1, 4)
        G.addEdge(1, 0)
        G.addEdge(1, 2)
        G.addEdge(3, 4)
        G.addEdge(2, 3)
        G.addEdge(4, 0)
        
        sinr = ge.SINr.load_from_graph(G)
        
        communities = sinr.detect_communities(gamma=50)
        sinr.extract_embeddings(communities)
        vec_remove=ge.InterpretableWordsModelBuilder(sinr, 'test', n_jobs=8, n_neighbors=25).build()
        
        self.vec_rm = vec_remove

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_remove_communities_dim_nnz(self):
        
        self.vec_rm.remove_communities_dim_nnz(3, 4)
        self.assertTrue((np.round(self.vec_rm.vectors.toarray(), 2) == np.round(np.array([[0.5, 0., 0., 0.5],
                                                                     [0., 0.25, 0.25, 0.25],
                                                                     [0.33333333, 0., 0.33333333, 0.33333333],
                                                                     [0.33333333, 0.33333333, 0., 0.33333333],
                                                                     [0.25, 0.25, 0.25, 0.]]), 2)).all())
        self.assertTrue(self.vec_rm.communities_sets == [{1}, {2}, {3}, {4}])
        self.assertTrue(self.vec_rm.community_membership == [1, 2, 3, 4])
                        
        self.vec_rm.remove_communities_dim_nnz(3, 3)
        self.assertTrue((np.round(self.vec_rm.vectors.toarray(), 2) == np.round(np.array([[0., 0.], 
                                                                      [0.25, 0.25],
                                                                      [0., 0.33333333],
                                                                      [0.33333333, 0.],
                                                                      [0.25, 0.25]]), 2)).all())
        self.assertTrue(self.vec_rm.communities_sets == [{2}, {3}])
        self.assertTrue(self.vec_rm.community_membership == [2, 3])


if __name__ == '__main__':
    unittest.main()