"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_evaluate
"""

import pytest
import unittest

import sinr.graph_embeddings as ge
from sinr.text.evaluate import fetch_data_MEN, fetch_data_WS353, eval_similarity, similarity_MEN_WS353_SCWS
import urllib.request
import os


class TestSinr_embeddings(unittest.TestCase):
    """Tests for `graph_embeddings` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

        # Load SINrVectors from OANC

        file = open('oanc.pk','wb')

        with urllib.request.urlopen('https://lium-cloud.univ-lemans.fr/index.php/s/d2gQ5DTK37DJq6H/download/model_consist_oanc0.pk') as response:
            file.write(response.read())
            
        file.close()
        
        vectors = ge.SINrVectors('oanc')
        vectors.load()
        self.vectors = vectors

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_eval_similarity(self):
        res = round(eval_similarity(self.vectors, fetch_data_MEN()), 2) 
        self.assertGreater(res, 0.38)

    def test_similarity_MEN_WS353_SCWS(self):
        res = similarity_MEN_WS353_SCWS(self.vectors)
        self.assertGreater(round(res["MEN"],2), 0.38)
        self.assertGreater(round(res["WS353"],2), 0.40)
        self.assertGreater(round(res["SCWS"],2), 0.38)

if __name__ == '__main__':
    unittest.main()