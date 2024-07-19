"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_evaluate
"""

import pytest
import unittest

import sinr.graph_embeddings as ge
from sinr.text.evaluate import fetch_data_MEN, fetch_data_WS353, eval_similarity, similarity_MEN_WS353_SCWS, vectorizer, clf_fit, clf_score
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
        
        vectors = ge.SINrVectors('oanc.pk')
        vectors.load('oanc.pk')
        self.vectors = vectors
        
        # datas for classification 
        X_train = [['goodbye', 'please', 'love'],[],['no', 'yes', 'friend', 'family', 'happy'],['a', 'the'],['beautiful','small','a']]
        y_train = [0,0,1,0,1]
        X_test = [['goodbye', 'family', 'friend'],['beautiful', 'small'],['please','happy','love']]
        y_test = [0,1,0]
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

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
    
    def test_vectorize(self):
        X, y = vectorizer(self.vectors, self.X_train, y=self.y_train)
        self.assertTrue(len(X) == len(y))
    
    def test_clf_fit_and_score(self):
        X_train, y_train = vectorizer(self.vectors, self.X_train, y=self.y_train)
        X_test, y_test = vectorizer(self.vectors, self.X_test, y=self.y_test)
        
        clf = clf_fit(X_train, y_train)
        score = clf_score(clf, X_test, y_test)
        
        self.assertTrue(score <= 1 and score >= 0)
        

if __name__ == '__main__':
    unittest.main()