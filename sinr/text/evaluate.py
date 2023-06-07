import numpy as np
from numpy.linalg import norm
import scipy
from scipy import stats
from sklearn.datasets._base import Bunch
import pandas as pd
import urllib.request
import os

def fetch_data_MEN():
    """Fetch MEN dataset for testing relatedness similarity
    
    :return: dictionary-like object. Keys of interest:
             'X': matrix of 2 words per column,
             'y': vector with scores
    :rtype: sklearn.datasets.base.Bunch
    
    """
    
    file = open('dataset.txt','wb')

    with urllib.request.urlopen('https://www.dropbox.com/s/b9rv8s7l32ni274/EN-MEN-LEM.txt?dl=1') as response:
        file.write(response.read())
    
    file.close()

    data = pd.read_csv('dataset.txt', header=None, sep=" ")
    
    os.remove('dataset.txt')
    
    # Remove last two chars from first two columns (-n, -a, -v)
    data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])

    data = Bunch(X=data.values[:, 0:2].astype("object"), y=(data.values[:, 2:].astype(float) / 5.0).ravel())

    return data

def fetch_data_WS353():
    """Fetch WS353 dataset for testing relatedness similarity
    
    :return: dictionary-like object. Keys of interest:
             'X': matrix of 2 words per column,
             'y': vector with scores
    :rtype: sklearn.datasets.base.Bunch
                    
    
    """
    
    file = open('dataset.txt','wb')

    with urllib.request.urlopen('https://www.dropbox.com/s/eqal5qj97ajaycz/EN-WS353.txt?dl=1') as response:
        file.write(response.read())

    file.close()
    
    data = pd.read_csv('dataset.txt', header=None, sep="\t")
    
    os.remove('dataset.txt')

    # Select the words pairs columns and the scores column
    X = data.values[1:, 0:2]
    y = data.values[1:, 2].astype(float)

    data = Bunch(X=X.astype("object"), y=y)  
    
    return data

def eval_similarity(sinr_vec, dataset):
    """Evaluate similarity with Spearman correlation
    
    :param sinr_vec: SINrVectors object
    
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
        scores.append(dataset.y[i])
    
    print(str(missing_words) + ' missing words')
    
    return scipy.stats.spearmanr(cosine_sim, scores).correlation

def similarity_MEN_WS353(sinr_vec):
    """Evaluate similarity with MEN and WS353 datasets

    :param sinr_vec: SINrVectors object
    
    :return: Spearman correlation for MEN and WS353 datasets
    :rtype: dict
    
    """
    
    sim_MEN = eval_similarity(sinr_vec, fetch_data_MEN())
    sim_WS353 = eval_similarity(sinr_vec, fetch_data_WS353())

    return {"MEN": sim_MEN, "WS353" : sim_WS353}
