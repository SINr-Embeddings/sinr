"""Tests for `sinr_embeddings` package.

python -m tests.test_sinr_concordancier
"""

import pytest
import unittest

import sinr.graph_embeddings as ge
from networkit import Graph


class TestSinr_embeddings(unittest.TestCase):
    """Tests for `graph_embeddings` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

        G = Graph(weighted = True)
        G.addNodes(4)
        G.addEdge(0, 1, w = 0)
        G.addEdge(0, 2, w = 1)
        G.addEdge(0, 3, w = 0)
        G.addEdge(3, 1, w = 3)
        G.addEdge(2, 1, w = 1)

        sinr = ge.SINr.load_from_graph(G)
        communities = sinr.detect_communities(gamma=50)
        sinr.extract_embeddings(communities)
        vec=ge.InterpretableGraphModelBuilder(sinr, 'test', n_jobs=8, n_neighbors=25).build()
        
        self.sinrv = vec

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_get_sum_cooccurrences(self):
        self.assertEqual(self.sinrv.get_sum_cooccurrences(), 5)
        self.assertEqual(self.sinrv.get_sum_cooccurrences(2), 2)
    
    def test_get_p_i(self):
        self.assertEqual(self.sinrv.get_p_i(2), 0.4)
    
    def test_get_cooc(self):
        self.assertEqual(self.sinrv.get_cooc(3, 1), 3.0)
    
    def test_get_p_i_j(self):
        self.assertEqual(self.sinrv.get_p_i_j(3, 1), 0.6)
        
    def test_get_pmi(self):
        self.assertEqual(self.sinrv.get_pmi(3, 1), 1.25)
        
    def test_get_npmi(self):
        res = round(self.sinrv.get_npmi(3, 1), 3) 
        self.assertEqual(res, 2.447)

if __name__ == '__main__':
    unittest.main()