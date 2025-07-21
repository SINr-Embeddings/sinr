import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt

def cosine_similarity(model, word1, word2):
    """
    Return the cosine similarity between two words in the given model.
    :param model: the model containing the word vectors
    :param word1: the first word
    :param word2: the second word
    
    :return: cosine similarity if both words are in the model, NaN otherwise.
    """
    if word1 not in model.vocab or word2 not in model.vocab:
        return np.nan
    v1 = model.get_my_vector(word1)
    v2 = model.get_my_vector(word2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def difference_vector_common_space(model1, model2, word):
    """
    Return the difference vector of word in the common space of model1 and model2.
    
    :param model1: model1 from which to get the vector for word
    :param model2: model2 from which to get the vector for word
    :param word: the word for which to compute the difference vector
    
    :return: the difference vector if both models have the word, None otherwise.
    """
    v1 = model1.get_my_vector(word)
    v2 = model2.get_my_vector(word)
    if v1.shape != v2.shape:
        print(f"Dimension mismatch for '{word}': {v1.shape} vs {v2.shape}")
        return None
    return v2 - v1

def compute_change_norms(model1, model2, words):
    """
    Compute the change norms for a list of words between two models.
    :param model1: the first model
    :param model2: the second model
    :param words: a list of words to compute the change norms for
    
    :return: an OrderedDict with words as keys and their change norms as values.
    If a word is not present in either model, its value will be None.
    """
    results = []
    for w in tqdm(words):
        if w not in model1.vocab or w not in model2.vocab:
            results.append((w, None)); continue
        diff = difference_vector_common_space(model1, model2, w)
        if diff is None:
            results.append((w, None)); continue
        results.append((w, float(np.linalg.norm(diff))))
    valid = {w: v for w, v in results if v is not None}
    missing = [w for w, v in results if v is None]
    sorted_valid = OrderedDict(sorted(valid.items(), key=lambda x: x[1], reverse=True))
    for w in missing:
        sorted_valid[w] = None
    return sorted_valid

def get_change_vectors(model1, model2, words=None):
    """
    Get change vectors for a list of words between two models.
    :param model1: the first model
    :param model2: the second model
    :param words: a list of words to compute the change vectors for. If None, uses the intersection of vocabularies.
    
    :return: a dictionary with words as keys and their change vectors as values.
    """
    if words is None:
        vocab = sorted(set(model1.vocab) & set(model2.vocab))
    else:
        vocab = [w for w in words if w in model1.vocab and w in model2.vocab]
    return {w: difference_vector_common_space(model1, model2, w) for w in vocab}

def compute_similarity_matrix(change_vectors):
    """
    Compute the similarity matrix from change vectors.
    :param change_vectors: a dictionary with words as keys and their change vectors as values.
    
    :return: a list of words and a similarity matrix.
    If no valid change vectors are found, returns an empty list and an empty array.
    """
    valid = [(w, v) for w, v in change_vectors.items() if v is not None]
    if not valid:
        print("No valid change vectors."); return [], np.array([])
    words, vecs = zip(*valid)
    V = np.stack(vecs)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    nz = norms[:, 0] > 0
    V[nz] /= norms[nz]
    S = V.dot(V.T)
    return list(words), S

def plot_topk_similarity_matrix(small1, small2, sinr1, file_name="heatmap", change_norms=None, topk=10):
    """
    Plot the top-k similarity matrix for change vectors between two models.
    :param small1: the first model
    :param small2: the second model
    :param sinr1: the model containing the vocabulary for similarity computation
    :param file_name: the base name for saving the heatmap images
    :param change_norms: precomputed change norms, if None they will be computed
    :param topk: the number of top words to consider for the heatmap
    
    :return: None, plots heatmaps for the top-k words and their neighbors.
    """
    if change_norms is None:
        change_norms = compute_change_norms(small1, small2, sinr1.vocab)

    top_k = [w for w, v in change_norms.items() if v is not None][:topk]

    # heatmaps
    for word in top_k:
        neigh = [w for w, _ in sinr1.most_similar(word)["neighbors "]][:topk]
        vecs = get_change_vectors(small1, small2, [word] + neigh)
        words, S = compute_similarity_matrix(vecs)

        plt.figure(figsize=(6, 6))
        plt.imshow(S, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine similarity')
        plt.xticks(range(len(words)), words, rotation=90)
        plt.yticks(range(len(words)), words)
        plt.title(f"Heatmap: '{word}' + neighbors")
        plt.savefig(f"{file_name}_{word}.png")
        plt.show()