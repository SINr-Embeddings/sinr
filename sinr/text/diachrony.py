import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt

class Diachronic:
    def __init__(self, model1, model2):
        """
        Analyzer for computing semantic changes between two transfered and aligned SINrVectors models.
        :param model1: SINrVectors model at time t1
        :param model2: SINrVectors model at time t2
        """
        self.model1 = model1
        self.model2 = model2

    @staticmethod
    def cosine_similarity(model, word1, word2):
        """Compute cosine similarity between two words in the same model.
        :param model: SINrVectors model to use
        :param word1: first word
        :param word2: second word
        
        :return: cosine similarity between the two words, or NaN if either word is not in the model
        :rtype: float
        """
        if word1 not in model.vocab or word2 not in model.vocab:
            return np.nan
        v1 = model.get_my_vector(word1)
        v2 = model.get_my_vector(word2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def difference_vector(self, word):
        """Compute the change vector of a word between the two models.
        :param word: word to compute change vector for
        
        :return: change vector (model2 - model1) or None if word not in both models
        :rtype: np.ndarray or None
        """
        if word not in self.model1.vocab or word not in self.model2.vocab:
            return None
        v1 = self.model1.get_my_vector(word)
        v2 = self.model2.get_my_vector(word)
        if v1.shape != v2.shape:
            print(f"Dimension mismatch for '{word}': {v1.shape} vs {v2.shape}")
            return None
        return v2 - v1

    def get_change_vectors(self, words=None):
        """Return the dictionary of change vectors for a list of words.
        :param words: list of words to compute change vectors for, if None, use all common words
        
        :return: dictionary of words and their change vectors
        :rtype: dict
        """
        if words is None:
            vocab = sorted(set(self.model1.vocab) & set(self.model2.vocab))
        else:
            vocab = [w for w in words if w in self.model1.vocab and w in self.model2.vocab]
        return {w: self.difference_vector(w) for w in vocab}

    def compute_change_norms(self, words):
        """Compute L2 norms of change vectors for a list of words.
        :param words: list of words to compute norms for
        
        :return: dictionary of words and their change norms, sorted by norm
        :rtype: OrderedDict
        """
        results = []
        for w in tqdm(words, desc="Computing change norms"):
            diff = self.difference_vector(w)
            if diff is None:
                results.append((w, None))
            else:
                results.append((w, float(np.linalg.norm(diff))))
        valid = {w: v for w, v in results if v is not None}
        missing = [w for w, v in results if v is None]
        sorted_valid = OrderedDict(sorted(valid.items(), key=lambda x: x[1], reverse=True))
        for w in missing:
            sorted_valid[w] = None
        return sorted_valid

    @staticmethod
    def compute_similarity_matrix(change_vectors):
        """Compute cosine similarity matrix between change vectors.
        :param change_vectors: dictionary of word-change vector pairs
        
        :return: list of words and their similarity matrix
        :rtype: (list, np.ndarray)
        """
        valid = [(w, v) for w, v in change_vectors.items() if v is not None]
        if not valid:
            print("No valid change vectors.")
            return [], np.array([])
        words, vecs = zip(*valid)
        V = np.stack(vecs)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        nz = norms[:, 0] > 0
        V[nz] /= norms[nz]
        S = V.dot(V.T)
        return list(words), S

    def plot_topk_similarity_matrix(self, sinr_model, file_name="heatmap", change_norms=None, topk=10):
        """Plot heatmaps of the top-k words with the highest change norms.
        :param sinr_model: SINrVectors model to use for similarity computation
        :param file_name: base name for output files
        :param change_norms: precomputed change norms, if None, compute them
        :param topk: number of top words to consider
        
        :return: None, saves heatmaps to files
        """
        if change_norms is None:
            change_norms = self.compute_change_norms(sinr_model.vocab)
    
        top_k = [w for w, v in change_norms.items() if v is not None][:topk]
    
        for word in top_k:
            try:
                res = sinr_model.most_similar(word)
                neighbors = res.get("neighbors", res.get("neighbors ", []))
            except KeyError:
                print(f"No neighbors found for '{word}', skipping.")
                continue
    
            neigh_words = [w for w, _ in neighbors][:topk]
            vecs = self.get_change_vectors([word] + neigh_words)
            words, S = self.compute_similarity_matrix(vecs)
    
            if len(words) <= 1:
                print(f"Not enough valid neighbors for '{word}', skipping heatmap.")
                continue
    
            plt.figure(figsize=(6, 6))
            plt.imshow(S, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(label='Cosine similarity')
            plt.xticks(range(len(words)), words, rotation=90)
            plt.yticks(range(len(words)), words)
            plt.title(f"Heatmap: '{word}' + neighbors")
            plt.savefig(f"{file_name}_{word}.png")
            plt.show()
