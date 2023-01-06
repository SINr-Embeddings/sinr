# -*- coding: utf-8 -*-

"""Tests for `graph_emn` package.

python -m sinr.tests.test_sinr_workflow
"""


import unittest

from .. import graph_embeddings
from ..text.cooccurrence import Cooccurrence
from ..text.pmi import pmi_filter

from ..graph_embeddings import NoCommunityDetectedException, NoEmbeddingExtractedException
from scipy.sparse import csr_matrix, coo_matrix
import networkit as nk

from sklearn.metrics import rand_score

from ..logger import logger
import logging

import numpy.testing as npt

from itertools import chain



class TestSinr_embeddings(unittest.TestCase):
    """Tests for `graph_embeddings` package."""
    

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
        self.sinr_from_graph = graph_embeddings.SINr.load_from_graph(self.G)
        self.sinr_from_mat = graph_embeddings.SINr.load_from_adjacency_matrix(self.adjacency)
        
        logger.setLevel(logging.CRITICAL)
        
        # Load your corpus as list of lists of tokens
        sentences = [["sinr", "is", "fun"], ["sinr", "is", "a", "python", "package"]]
        self.sentences = sentences
        
        # Build cooccurrence matrix
        c = Cooccurrence()
        c.fit(sentences, window=2)
        
        #Normalise cooccurrence matrix using PPMI
        c.matrix = pmi_filter(c.matrix)
        c.save("matrix.pk")
        
        self.sinr_from_cooc = graph_embeddings.SINr.load_from_cooc_pkl("matrix.pk")

        

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_load_voc(self):

        res = set(chain(*self.sentences))
        # ['a', 'fun', 'is', 'package', 'python', 'sinr']
        self.assertEquals(set(self.sinr_from_cooc.get_vocabulary()), res)
    
        self.assertEquals(set(self.sinr_from_graph.get_vocabulary()), set(self.G.iterNodes()))
        
        self.assertEquals(set(self.sinr_from_mat.get_vocabulary()), set(self.G.iterNodes()))
    
    def test_community_exception(self):
        self.assertRaises(NoCommunityDetectedException,self.sinr_from_graph.get_communities)
        
        
    def test_detect_communities(self):
        communities = self.sinr_from_graph.detect_communities(gamma=1, inspect=False)
        communities = communities.getVector()
        print(communities)
        self.assertAlmostEqual(rand_score([0,0,0,1,1], communities), 1)
        self.assertAlmostEqual(rand_score([0,0,0,1,1], self.sinr_from_graph.get_communities().getVector()), 1)
        
        communities = self.sinr_from_mat.detect_communities(gamma=1, inspect=False)
        communities = communities.getVector()
        self.assertAlmostEqual(rand_score([0,0,0,1,1], communities), 1)
        self.assertAlmostEqual(rand_score([0,0,0,1,1], self.sinr_from_mat.get_communities().getVector()), 1)
        
        communities = self.sinr_from_cooc.detect_communities(gamma=1, inspect=False)
        communities = communities.getVector()
        # ['a', 'fun', 'is', 'package', 'python', 'sinr']
        # [0    ,1     ,2    ,0         ,0      ,1]
        self.assertAlmostEqual(rand_score([0,1,2,0,0,1], communities), 1)
        self.assertAlmostEqual(rand_score([0,1,2,0,0,1], self.sinr_from_cooc.get_communities().getVector()), 1)
        
    def test_extract_embeddings(self):
        communities = self.sinr_from_graph.detect_communities(gamma=1, inspect=False)
        self.sinr_from_graph.extract_embeddings(communities)
        nr = self.sinr_from_graph.get_nr()
        ref = csr_matrix([[1,0],[1,0],[0.66666666, 0.33333333], [0.5, 0.5], [0,1]])
        try:
            npt.assert_array_almost_equal(nr.todense(), ref.todense())
        except AssertionError:
            self.fail("Nr not equals to what is expected")
        
        communities = self.sinr_from_mat.detect_communities(gamma=1, inspect=False)
        self.sinr_from_mat.extract_embeddings(communities)
        nr = self.sinr_from_mat.get_nr()
        try:
            npt.assert_array_almost_equal(nr.todense(), ref.todense())
        except AssertionError:
            self.fail("Nr not equals to what is expected")
            
        # Graphe : a-package-python, fun-sinr, is
        # "is" of community 2 is not connected
        ref = csr_matrix([[1, 0, 0],[0,1,0],[0, 0, 0], [1, 0, 0], [1, 0, 0], [0,1, 0]])
        communities = self.sinr_from_cooc.detect_communities(gamma=1, inspect=False)
        self.sinr_from_cooc.extract_embeddings(communities)
        nr = self.sinr_from_cooc.get_nr()
        try:
            npt.assert_array_almost_equal(nr.todense(), ref.todense())
        except AssertionError:
            self.fail("Nr not equals to what is expected")
    
    def test_nfm_exception(self):
        self.assertRaises(NoEmbeddingExtractedException,self.sinr_from_graph.get_nr)        
        
            

if __name__ == '__main__':
    unittest.main()