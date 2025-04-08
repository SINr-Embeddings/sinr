"""Tests for `sinr_embeddings` package.

python -m sinr.tests.test_sinr_evaluate
"""

import pytest
import unittest
import numpy as np

import sinr.graph_embeddings as ge
from sinr.text.evaluate import fetch_data_MEN, fetch_data_WS353, eval_similarity, similarity_MEN_WS353_SCWS, vectorizer, clf_fit, clf_score, calcul_analogy_normalized, calcul_analogy_sparsified_normalized, calcul_analogy_value_zero, calc_direct_bias_sinr, calc_indirect_bias_sinr, project_vector, reject_vector, identify_gender_direction_sinr, eval_analogy_k, best_predicted_word_k
import urllib.request
import os
import tempfile

from scipy.sparse import csr_matrix


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
    """Mock SINrVectors class for testing analogy functions."""
    def __init__(self, vocab, vectors):
        self.vocab = vocab
        self.vectors = csr_matrix(vectors)
        
class TestAnalogyFunctions(unittest.TestCase):
    """Tests for analogy functions."""
    def setUp(self):
        """Set up a simple vocabulary and vectors for testing."""
        vocab = ["king", "queen", "man", "woman"]
        vectors = np.array([
            [0.8, 0.2, 0.0],  # "king"
            [0.7, 0.3, 0.0],  # "queen"
            [0.6, 0.4, 0.0],  # "man"
            [0.5, 0.5, 0.0],  # "woman"
        ])
        self.sinr_vec = MockSINrVectors(vocab, vectors)

    def test_best_predicted_word_correct(self):
        word_a = "king"
        word_b = "queen"
        word_c = "man"
        expected = "woman"

        result = calcul_analogy_normalized(self.sinr_vec, word_a, word_b, word_c)

        self.assertEqual(result, expected)

    def test_best_predicted_word_exclusion(self):
        result = calcul_analogy_normalized(self.sinr_vec, "king", "queen", "king")
        self.assertNotEqual(result, "king")

    def test_best_predicted_word_invalid_word(self):
        word_a = "dog"
        word_b = "queen"
        word_c = "man"
        expected = None

        result = calcul_analogy_normalized(self.sinr_vec, word_a, word_b, word_c)

        self.assertIsNone(result)        
        
class TestCalculAnalogySparsifiedNormalized(unittest.TestCase):
    """Tests for `calcul_analogy_sparsified_normalized` function."""
    def setUp(self):
        self.vocab = ["king", "queen", "man", "woman", "child"]
        self.vectors = np.array([
            [0.8, 0.1, 0.1, 0.0],  # king
            [0.7, 0.2, 0.1, 0.0],  # queen
            [0.6, 0.3, 0.1, 0.0],  # man
            [0.5, 0.4, 0.1, 0.0],  # woman
            [0.2, 0.1, 0.7, 0.0],  # child
        ])
        self.sinr_vec = MockSINrVectors(self.vocab, self.vectors)

    def test_correct_analogy(self):
        result = calcul_analogy_sparsified_normalized(self.sinr_vec, "king", "queen", "man", n=2)
        self.assertEqual(result, "woman")

    def test_nonexistent_word(self):
        result = calcul_analogy_sparsified_normalized(self.sinr_vec, "dog", "queen", "man", n=2)
        self.assertIsNone(result)

    def test_small_n(self):
        result = calcul_analogy_sparsified_normalized(self.sinr_vec, "king", "queen", "man", n=1)
        self.assertEqual(result, "woman")

    def test_large_n(self):
        result = calcul_analogy_sparsified_normalized(self.sinr_vec, "king", "queen", "man", n=10)
        self.assertEqual(result, "woman")

class TestCalculAnalogyValueZero(unittest.TestCase):
    """Tests for `calcul_analogy_value_zero` function."""
    def setUp(self):
        self.vocab = ["king", "queen", "man", "woman", "child"]
        self.vectors = np.array([
            [0.8, 0.1, 0.1, 0.0],  # king
            [0.7, 0.2, 0.1, 0.0],  # queen
            [0.6, 0.3, 0.1, 0.0],  # man
            [0.5, 0.4, 0.1, 0.0],  # woman
            [0.2, 0.1, 0.7, 0.0],  # child
        ])
        self.sinr_vec = MockSINrVectors(self.vocab, self.vectors)

    def test_correct_analogy(self):
        result = calcul_analogy_value_zero(self.sinr_vec, "king", "queen", "man")
        self.assertEqual(result, "woman")

    def test_nonexistent_word(self):
        result = calcul_analogy_value_zero(self.sinr_vec, "dog", "queen", "man")
        self.assertIsNone(result)

    def test_exclusion_of_words(self):
        result = calcul_analogy_value_zero(self.sinr_vec, "king", "queen", "king")
        self.assertNotEqual(result, "king")
        self.assertNotEqual(result, "queen")

    def test_vector_clamping_to_zero(self):
        result = calcul_analogy_value_zero(self.sinr_vec, "woman", "man", "queen")
        self.assertIn(result, self.vocab)
        
class TestBestPredictedWordK(unittest.TestCase):
    """Tests for `best_predicted_word_k` function."""
    def setUp(self):
        self.vocab = ["king", "queen", "man", "woman"]
        self.vectors = [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
        ]
        self.sinr_vec = MockSINrVectors(self.vocab, self.vectors)

    def test_analogy_correct(self):
        result = best_predicted_word_k(self.sinr_vec, "king", "queen", "man", k=1)
        self.assertIn("woman", result)

    def test_invalid_word(self):
        result = best_predicted_word_k(self.sinr_vec, "foo", "queen", "man", k=1)
        self.assertIsNone(result)

    def test_exclusion_from_results(self):
        result = best_predicted_word_k(self.sinr_vec, "king", "queen", "king", k=1)
        self.assertNotIn("king", result)
        
class TestEvalAnalogyK(unittest.TestCase):
    """Tests for `eval_analogy_k` function."""
    def setUp(self):
        self.vocab = ["king", "queen", "man", "woman"]
        self.vectors = [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
        ]
        self.sinr_vec = MockSINrVectors(self.vocab, self.vectors)

        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.txt')
        self.temp_file.write(": capital-common-countries\n")
        self.temp_file.write("king queen man woman\n")
        self.temp_file.write("man woman king queen\n")
        self.temp_file.flush()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_eval_analogy(self):
        error_rate = eval_analogy_k(self.sinr_vec, self.temp_file.name, best_predicted_word_k, k=1)
        self.assertTrue(0 <= error_rate <= 1)

class MockClassSINr:
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
        self.sinr_vec = MockClassSINr(self.vocab, vectors)

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
        self.sinr_vec = MockClassSINr(self.vocab, vectors)
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
        similarity = calc_indirect_bias_sinr(self.sinr_vec, 'father', 'mother', self.gender_direction)
        self.assertIsInstance(similarity, float)

    def test_calc_indirect_bias_sinr_edge_case(self):
        similarity = calc_indirect_bias_sinr(self.sinr_vec, 'father', 'father', self.gender_direction)
        self.assertEqual(similarity, 0.0)

if __name__ == '__main__':
    unittest.main()