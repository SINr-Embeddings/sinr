import numpy as np
from numpy.linalg import norm
import scipy
from scipy import stats
from sklearn.datasets._base import Bunch
import pandas as pd
import urllib.request
import os
from tqdm.auto import tqdm
import time

def fetch_data_MEN():
    """Fetch MEN dataset for testing relatedness similarity
    
    :return: dictionary-like object. Keys of interest:
             'X': matrix of 2 words per column,
             'y': vector with scores
    :rtype: sklearn.datasets.base.Bunch
    
    """
    
    file_name = 'dataset' + str(round(time.time()*1000)) + '.txt'
    
    file = open(file_name,'wb')

    with urllib.request.urlopen('https://www.dropbox.com/s/b9rv8s7l32ni274/EN-MEN-LEM.txt?dl=1') as response:
        file.write(response.read())
    
    file.close()

    data = pd.read_csv(file_name, header=None, sep=" ")
    
    os.remove(file_name)
    
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
    
    file_name = 'dataset' + str(round(time.time()*1000)) + '.txt'
    
    file = open(file_name,'wb')

    with urllib.request.urlopen('https://www.dropbox.com/s/eqal5qj97ajaycz/EN-WS353.txt?dl=1') as response:
        file.write(response.read())

    file.close()
    
    data = pd.read_csv(file_name, header=None, sep="\t")
    
    os.remove(file_name)

    # Select the words pairs columns and the scores column
    X = data.values[1:, 0:2]
    y = data.values[1:, 2].astype(float)

    data = Bunch(X=X.astype("object"), y=y)  
    
    return data

def fetch_data_SCWS():
    """Fetch SCWS dataset for testing relatedness similarity
    
    :return: dictionary-like object. Keys of interest:
             'X': matrix of 2 words per column,
             'y': vector with scores
    :rtype: sklearn.datasets.base.Bunch
    
    """
    
    file_name = 'dataset' + str(round(time.time()*1000)) + '.txt'
    
    file = open(file_name,'wb')

    with urllib.request.urlopen('https://raw.githubusercontent.com/jjlastra/HESML/master/HESML_Library/WN_Datasets/SCWS1994_dataset.csv') as response:
        file.write(response.read())
    
    file.close()

    data = pd.read_csv(file_name, header=None, sep=";")
    
    os.remove(file_name)

    data = Bunch(X=data.values[:, 0:2].astype("object"), y=(data.values[:, 2:].astype(float) / 5.0).ravel())

    return data

def fetch_SimLex(which="665"):
    """Fetch SimLex datasets for testing relatedness similarity
    
    :param which: dataset (default value = "665")
    :type which: str
    
    :return: dictionary-like object. Keys of interest:
             'X': matrix of 2 words per column,
             'y': vector with scores
    :rtype: sklearn.datasets.base.Bunch
    
    """
    
    file_name = 'dataset' + str(round(time.time()*1000)) + '.txt'
    
    file = open(file_name,'wb')
    
    # Nouns
    if which=="665":
        with urllib.request.urlopen('https://raw.githubusercontent.com/jjlastra/HESML/master/HESML_Library/WN_Datasets/SimLex665_dataset.csv') as response:
            file.write(response.read())
    
    # Adjectives, nouns and verbs
    elif which=="999":
        with urllib.request.urlopen('https://raw.githubusercontent.com/jjlastra/HESML/master/HESML_Library/WN_Datasets/SimLex999_dataset.csv') as response:
            file.write(response.read())
    
    # Verbs
    elif which=="222":
        with urllib.request.urlopen('https://raw.githubusercontent.com/jjlastra/HESML/master/HESML_Library/WN_Datasets/SimLex222_verbs_dataset.csv') as response:
            file.write(response.read())
    
    # Adjectives
    elif which=="111":
        with urllib.request.urlopen('https://raw.githubusercontent.com/jjlastra/HESML/master/HESML_Library/WN_Datasets/SimLex111_adjectives_dataset.csv') as response:
            file.write(response.read())

    else:
        RuntimeError("Not recognised which parameter")
        
    file.close()
    
    data = pd.read_csv(file_name, header=None, sep=";")
    
    os.remove(file_name)
        
    data = Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2].astype(float))
    
    return data

def eval_similarity(sinr_vec, dataset, print_missing=True):
    """Evaluate similarity with Spearman correlation
    
    :param sinr_vec: SINrVectors object
    
    :param dataset: sklearn.datasets.base.Bunch
                    dictionary-like object. Keys of interest:
                    'X': matrix of 2 words per column,
                    'y': vector with scores

    :param print_missing: boolean (default : True)
    
    :return: Spearman correlation between cosine similarity and human rated similarity
    :rtype: float
    
    """
    
    scores = list()
    cosine_sim = list()
    
    vocab = sinr_vec.vocab
    missing_words = list()
    
    # Mean vector
    vec_mean = np.ravel(sinr_vec.vectors.mean(axis=0))

    for i in tqdm(range(len(dataset.X)), desc = 'eval similarity', leave = False):

        # Words into vectors
        # Missing words replaced by mean vector
        
        if dataset.X[i][0] not in vocab:
            if dataset.X[i][0].lower() in vocab:
                vec1 = sinr_vec._get_vector(sinr_vec._get_index(dataset.X[i][0].lower()))
            else:
                vec1 = vec_mean
                if dataset.X[i][0] not in missing_words:
                    missing_words.append(dataset.X[i][0])
        else:
            vec1 = sinr_vec._get_vector(sinr_vec._get_index(dataset.X[i][0]))
            
        if dataset.X[i][1] not in vocab:
            if dataset.X[i][1].lower() in vocab:
                vec2 = sinr_vec._get_vector(sinr_vec._get_index(dataset.X[i][1].lower()))
            else:
                vec2 = vec_mean
                if dataset.X[i][1] not in missing_words:
                    missing_words.append(dataset.X[i][0])
        else:
            vec2 = sinr_vec._get_vector(sinr_vec._get_index(dataset.X[i][1]))
        
        # Cosine similarity
        cosine_sim.append(np.dot(vec1,vec2)/(norm(vec1)*norm(vec2)))
        scores.append(dataset.y[i])
    if print_missing == True:
        print(str(len(missing_words)) + ' missing words')
    
    return scipy.stats.spearmanr(cosine_sim, scores).correlation

def similarity_MEN_WS353_SCWS(sinr_vec, print_missing=True):
    """Evaluate similarity with MEN, WS353 and SCWS datasets

    :param sinr_vec: SINrVectors object

    :param print_missing: boolean (default : True)
    
    :return: Spearman correlation for MEN, WS353 and SCWS datasets
    :rtype: dict
    
    """
    
    sim_MEN = eval_similarity(sinr_vec, fetch_data_MEN(), print_missing=print_missing)
    sim_WS353 = eval_similarity(sinr_vec, fetch_data_WS353(), print_missing=print_missing)
    sim_SCWS = eval_similarity(sinr_vec, fetch_data_SCWS(), print_missing=print_missing)

    return {"MEN": sim_MEN, "WS353" : sim_WS353, "SCWS" : sim_SCWS}
