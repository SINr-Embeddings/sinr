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
        
        file = open("w2v_for_distRatio.txt", "w")
        file.write("a 0.2 0.7 0.0 0.0\nb 0.5 0.1 0.0 0.0\nc 0.7 0.5 0.0 0.0\nd 0.9 0.4 0.0 0.0\ne 0.8 0.0 0.1 0.0\nf 0.6 0.0 0.2 0.0\ng 0.4 0.0 0.3 0.0\nh 0.3 0.0 0.4 0.0\ni 0.0 0.9 0.3 0.0\nj 0.0 0.8 0.2 0.0\nk 0.0 0.3 0.5 0.0\nl 0.0 0.2 0.1 0.0\nm 0.0 0.4 0.0 0.9\nn 0.0 0.2 0.0 0.4\no 0.0 0.6 0.0 0.4\np 0.0 0.1 0.0 0.3\nq 0.0 0.0 0.9 0.5\nr 0.0 0.0 0.8 0.6\ns 0.0 0.0 0.7 0.7\nt 0.0 0.0 0.6 0.8")
        file.close()
        sinr_vec = ge.SINrVectors.load_from_w2v("w2v_for_distRatio.txt", "sinrvec")
        self.sinr_vec = sinr_vec
        
    def tearDown(self):
        """Tear down test fixtures, if any."""
        os.remove("w2v_for_distRatio.txt")
    
    def test_get_topk(self):
        self.assertTrue(set(self.sinr_vec._get_topk(0, row=False)) == set({3, 4, 2, 5, 1}))
        
    def test_get_bottomk(self):
        self.assertTrue(set(self.sinr_vec._get_bottomk(0, topk=13, row=False)) == set({0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}))

    def test_get_union_topk(self):
        intruder_candidates = set({3, 4, 2, 5, 1, 8, 9, 0, 14, 16, 17, 18, 19, 10, 12})
        self.assertTrue((set(self.sinr_vec.get_union_topk(25)) == intruder_candidates))
        
    def test_pick_intruder(self):
        res = set()
        for ii in range(20):
            res.add(self.sinr_vec.pick_intruder(0))
        self.assertTrue((res - self.sinr_vec.get_union_topk(10)) == set() and (res - set(self.sinr_vec._get_bottomk(0, topk=10, row=False))) == set())


if __name__ == '__main__':
    unittest.main()