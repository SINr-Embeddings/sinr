"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_distRatio
"""

import pytest
import unittest

import sinr.graph_embeddings as ge
import urllib.request
import os


class TestSinr_embeddings(unittest.TestCase):
    """Tests for `graph_embeddings` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        
        sinr_vec = ge.SINrVectors.load_from_w2v("./w2v_for_distRatio.txt", "sinrvec")
        self.sinr_vec = sinr_vec
        
    def tearDown(self):
        """Tear down test fixtures, if any."""
    
    def test_get_topk(self):
        self.assertTrue(set(self.sinr_vec._get_topk(0, row=False)) == set({3, 4, 2, 5, 1}))
        
    def test_get_bottomk(self):
        self.assertTrue(set(self.sinr_vec._get_bottomk(0, topk=13, row=False)) == set({0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}))

    def test_get_union_topk(self):
        intruder_candidates = set({3, 4, 2, 5, 1, 8, 9, 0, 14, 16, 17, 18, 19, 10, 12})
        self.assertTrue((set(self.sinr_vec.get_union_topk(25)) == intruder_candidates))
        
    def test_pick_intruder(self):
        # un intru d'une dimension doit:
        # - être dans le bottom de la dimension
        # - être dans l'union des topk
        res = set()
        for ii in range(20):
            res.add(self.sinr_vec.pick_intruder(0))
        self.assertTrue((res - self.sinr_vec.get_union_topk(10)) == set() and (res - set(self.sinr_vec._get_bottomk(0, topk=10, row=False))) == set())


if __name__ == '__main__':
    unittest.main()