"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_evaluate
"""

import pytest
import unittest

import sinr.graph_embeddings as ge
from sinr.text.evaluate import fetch_data_MEN, fetch_data_WS353, eval_similarity, similarity_MEN_WS353_SCWS, vectorizer, clf_fit, clf_score, normalize, identify_gender_direction_sinr, calc_direct_bias_sinr, load_config
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
        
class MockSINrVectors:
    def __init__(self, vocab, vectors):
        self.vocab = vocab  
        self.vectors = vectors

    def get_my_vector(self, word):
        if word in self.vocab:
            return self.vectors[self.vocab.index(word)]
        raise ValueError(f"Word '{word}' not found in vocab.")

class TestBiasFunctions(unittest.TestCase):
    def setUp(self):
        self.vocab = ["men", "woman", "father", "mother", "male", "female", "actor", "actress"]
        vectors = [np.random.rand(300) for _ in range(len(self.vocab))]
        self.sinr_vec = MockSINrVectors(self.vocab, vectors)

        self.config = {
            "gender": {
                "definitional_pairs": [["men", "woman"], ["father", "mother"]]
            },
            "professions": ["actor", "actress"]
        }

    def test_gender_direction(self):
        direction = identify_gender_direction_sinr(
            self.sinr_vec, 
            self.config["gender"]["definitional_pairs"], 
            method="pca"
        )
        self.assertEqual(direction.shape[0], 300)

    def test_direct_bias(self):
        direction = identify_gender_direction_sinr(
            self.sinr_vec, 
            self.config["gender"]["definitional_pairs"]
        )
        bias = calc_direct_bias_sinr(
            self.sinr_vec, 
            self.config["professions"], 
            direction
        )
        self.assertTrue(0 <= bias <= 1)

class TestIndirectBiasFunctions(unittest.TestCase):

    def setUp(self):
        self.vocab = ['father', 'mother', 'he', 'she']
        np.random.seed(42)  
        vectors = [np.random.rand(300) for _ in range(len(self.vocab))]
        self.model = MockSINrVectors(self.vocab, vectors)
        self.gender_direction = np.random.rand(300)

    def test_project_vector(self):
        v = np.array([3, 4])
        u = np.array([1, 0])
        projected_vector = project_vector(v, u)
        self.assertTrue(np.allclose(projected_vector, np.array([3, 0])))

    def test_reject_vector(self):
        v = np.array([3, 4])
        u = np.array([1, 0])
        rejected_vector = reject_vector(v, u)
        self.assertTrue(np.allclose(rejected_vector, np.array([0, 4])))

    def test_calc_indirect_bias_sinr(self):
        similarity = calc_indirect_bias_sinr(self.model, 'father', 'mother', self.gender_direction)
        self.assertIsInstance(similarity, float)

    def test_calc_indirect_bias_sinr_edge_case(self):
        similarity = calc_indirect_bias_sinr(self.model, 'father', 'father', self.gender_direction)
        self.assertEqual(similarity, 0.0)  

        
if __name__ == '__main__':
    unittest.main()