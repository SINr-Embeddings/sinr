import numpy as np
from numpy.linalg import norm
import scipy
from scipy import stats
from sklearn.datasets._base import Bunch

def eval_sim_bunch(sinr_vec, dataset):
    """Evaluate similarity with Spearman correlation
    
    :param dataset: sklearn.datasets.base.Bunch
                    dictionary-like object. Keys of interest:
                    'X': matrix of 2 words per column,
                    'y': vector with scores
    
    :return: Spearman correlation between cosine similarity and human rated similarity
    :rtype: float
    
    """
    
    scores = list()
    cosine_sim = list()
    
    vocab = sinr_vec.vocab
    missing_words = 0
    
    # Mean vector
    vec_mean = np.ravel(sinr_vec.vectors.mean(axis=0))

    for i in range(len(dataset.X)):

        # Words into vectors
        # Missing words replaced by mean vector
        
        if dataset.X[i][0] not in vocab:
            vec1 = vec_mean
            missing_words += 1
        else:
            vec1 = sinr_vec._get_vector(sinr_vec._get_index(dataset.X[i][0]))
            
        if dataset.X[i][1] not in vocab:
            vec2 = vec_mean
            missing_words += 1
        else:
            vec2 = sinr_vec._get_vector(sinr_vec._get_index(dataset.X[i][1]))
        
        # Cosine similarity
        cosine_sim.append(np.dot(vec1,vec2)/(norm(vec1)*norm(vec2)))
        #cosine_sim.append(1 - spatial.distance.cosine(vec1, vec2))
        scores.append(dataset.y[i])
    
    print(str(missing_words) + ' missing words')
    
    return scipy.stats.spearmanr(cosine_sim, scores).correlation

def eval_sim_f(sinr_vec, ratings_file):
    """Evaluate similarity with Spearman correlation
    
    :param ratings_file: txt file with "word1 word2 score" for each line
    
    :return: Spearman correlation between cosine similarity and human rated similarity
    :rtype: float
    
    """
    
    scores = list()
    cosine_sim = list()
    
    vocab = sinr_vec.vocab
    missing_words = 0

    tmp = list()
    
    # Mean vector
    vec_mean = np.ravel(sinr_vec.vectors.mean(axis=0))

    file = open(ratings_file, "r")

    for line in file:
        tmp = line.split()
        word1 = tmp[0]
        word2 = tmp[1]
        
        # Words into vectors
        # Missing words replaced by mean vector
        if word1 not in vocab:
            vec1 = vec_mean
            missing_words += 1
        else:
            vec1 = sinr_vec._get_vector(sinr_vec._get_index(word1))
        if word2 not in vocab:
            vec2 = vec_mean
            missing_words += 1
        else:
            vec2 = sinr_vec._get_vector(sinr_vec._get_index(word2))
        
        # Cosine similarity
        cosine_sim.append(np.dot(vec1,vec2)/(norm(vec1)*norm(vec2)))
        #cosine_sim.append(1 - spatial.distance.cosine(vec1, vec2))
        scores.append(tmp[2])

    file.close()
    
    print(str(missing_words) + ' missing words')
    
    return scipy.stats.spearmanr(cosine_sim, scores).correlation