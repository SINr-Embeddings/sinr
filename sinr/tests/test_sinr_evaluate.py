"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_evaluate
"""

#import sys
#sys.path.append("../text/")

import unittest

import sinr.graph_embeddings as ge
#from evaluate import fetch_data_MEN, fetch_data_WS353, eval_similarity, similarity_MEN_WS353
from ..text.evaluate import fetch_data_MEN, fetch_data_WS353, eval_similarity, similarity_MEN_WS353

class TestSinr_embeddings(unittest.TestCase):
    """Tests for `graph_embeddings` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

        # Load SINrVectors from OANC
        vectors = ge.SINrVectors('./models/oanc/oanc')
        vectors.load()
        self.vectors = vectors

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_eval_similarity(self):
        res = round(eval_similarity(self.vectors, fetch_data_MEN()), 2) 
        self.assertEqual(res, 0.39)

    def test_similarity_MEN_WS353(self):
        res = similarity_MEN_WS353(self.vectors)
        self.assertEqual(round(res["MEN"],2), 0.39)
        self.assertEqual(round(res["WS353"],2), 0.44)

if __name__ == '__main__':
    unittest.main()