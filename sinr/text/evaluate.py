import numpy as np
from numpy.linalg import norm
import scipy
from scipy import stats
from sklearn.datasets._base import Bunch
import sklearn.metrics as metrics
import pandas as pd
import urllib.request
import os
from tqdm.auto import tqdm
import time
import xgboost as xgb

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
                    missing_words.append(dataset.X[i][1])
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

def dist_ratio(sinr_vec, union=None, prctbot=50, prcttop=10, nbtopk=5, dist=True):
    """DistRatio of the model
    
    :param sinr_vec: SINrVectors object
    :type sinr_vec: SINrVectors
    :param union: ids of words that are among the top prct of at least one dimension (defaults to None)
    :type union: int list
    :param prctbot: bottom prctbot to pick (defaults to 50)
    :type prctbot: int
    :param prcttop: top prcttop to pick (defaults to 10)
    :type prcttop: int
        
    :returns: DisRatio of the model
    :rtype: float
        
    """
    ratio = 0
    if union == None:
        union = sinr_vec.get_union_topk(prct = prcttop)
    nb_dims = sinr_vec.get_number_of_dimensions()
    for dim in tqdm(range(nb_dims)):
        ratio += dist_ratio_dim(sinr_vec, dim, union=union, prctbot=prctbot, prcttop=prcttop, nbtopk=nbtopk)
    return ratio / nb_dims


def dist_ratio_dim(sinr_vec, dim, union=None, prctbot=50, prcttop=10, nbtopk=5, dist=True):
    """DistRatio for one dimension of the model
    
    :param sinr_vec: SINrVectors object
    :type sinr_vec: SINrVectors
    :param dim: the index of the dimension for which to get the DistRatio
    :type dim: int
    :param union: ids of words that are among the top prct of at least one dimension (defaults to None)
    :type union: int list
    :param prctbot: bottom prctbot to pick (defaults to 50)
    :type prctbot: int
    :param prcttop: top prcttop to pick (defaults to 10)
    :type prcttop: int
    :param nbtopk: number of top words to pick (defaults to 5)
    :type nbtopk: int
    :param dist: set to True (default) to use cosine distance and False to use cosine similarity
    :type dist: boolean
        
    :returns: DistRatio for dimension `dim`
    :rtype: float
        
    """
    intruder = sinr_vec.pick_intruder(dim, union, prctbot, prcttop)
    topks = sinr_vec._get_topk(dim, topk = nbtopk, row=False)
    intra = sinr_vec.intra_sim(topks, dist)
    inter = sinr_vec.inter_sim(intruder, topks, dist)
    if dist:
        return inter / intra
    else:
        if inter == 0:
            print("dimension",dim,"inter nulle", topks)
            return 0
        return intra / inter

def vectorizer(sinr_vec, X, y=[]):
    """Vectorize preprocessed documents to sinr embeddings
    
    :param sinr_vec: SINrVectors object
    :type sinr_vec: SINrVectors
    :param X: preprocessed documents
    :type X: text (list(list(str))): A list of documents containing words
    :param y: documents labels
    :type y: numpy.ndarray
    
    :returns: list of vectors
    """
    
    if len(y) > 0 and len(X) != len(y):
        raise ValueError("X and y must be the same size")
    
    indexes = set()
    vectors = list()
    
    # Doc to embedding
    for i, doc in enumerate(X):
        doc_vec = [sinr_vec._get_vector(sinr_vec._get_index(token)) for token in doc if token in sinr_vec.vocab]
        if len(doc_vec) == 0:
            indexes.add(i)
        else:
            vectors.append(np.mean(doc_vec, axis=0))
        
    # Delete labels of:
    #- empty documents
    #- documents with only unknown vocabulary
    if len(y) > 0:
        y = np.delete(y, list(indexes))
        y = list(map(int,y))
          
    return vectors, y

def clf_fit(X_train, y_train, clf=xgb.XGBClassifier()):
    """Fit a classification model according to the given training data.
    :param X_train: training data
    :type X_train: list of vectors
    :param y_train: labels
    :type y_train: numpy.ndarray
    :param clf: classifier
    :type clf: classifier (ex.: xgboost.XGBClassifier, sklearn.svm.SVC)
    
    :returns: Fitted classifier
    :rtype: classifier
    """
    clf.fit(X_train, y_train)
    return clf

def clf_score(clf, X_test, y_test, scoring='accuracy', params={}):
    """Evaluate classification on given test data.
    :param clf: classifier
    :type clf: classifier (ex.: xgboost.XGBClassifier, sklearn.svm.SVC)
    :param X_test: test data
    :type X_test: list of vectors
    :param y_test: labels
    :type y_test: numpy.ndarray
    :param scoring: scikit-learn scorer object, default='accuracy'
    :type scoring: str
    :param params: parameters for the scorer object
    :type params: dictionary
    
    :returns: Score
    :rtype: float
    """
    score = getattr(metrics, scoring+'_score')
    y_pred = clf.predict(X_test)
    return score(y_test, y_pred, **params)

def clf_xgb_interpretability(sinr_vec, xgb, interpreter,topk_dim=10, topk=5, importance_type='gain'):
    """Interpretability of main dimensions used by the xgboost classifier
    :param sinr_vec: SINrVectors object from which datas were vectorized
    :type sinr_vec: SINrVectors
    :param xgb: fitted xgboost classifier
    :type xgb: xgboost.XGBClassifier
    :param interpreter: whether stereotypes or descriptors are requested
    :type interpreter: str
    :param topk_dim: Number of features requested among the main features used by the classifier (Default value = 10)
    :type topk_dim: int
    :param topk: `topk` value to consider on each dimension (Default value = 5)
    :type topk: int
    :param importance_type: ‘weight’: the number of times a feature is used to split the data across all trees,
                            ‘gain’: the average gain across all splits the feature is used in,
                            ‘cover’: the average coverage across all splits the feature is used in,
                            ‘total_gain’: the total gain across all splits the feature is used in
                            ‘total_cover’: the total coverage across all splits the feature is used in
    :type importance_type: str
    
    :returns: Interpreters of dimensions, importance of dimensions
    :rtype: list of set of object, list of tuple

    """
    
    features = xgb.get_booster().get_score(importance_type=importance_type)
    features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True))
    features_index = [int(f[1:]) for f in list(features.keys())[:topk_dim]]
    features_importance = list(features.items())[:topk_dim]
    
    if interpreter=='descriptors':
        dim = [sinr_vec.get_dimension_descriptors_idx(index, topk=topk) for index in features_index]
    elif interpreter=='stereotypes':
        dim = [sinr_vec.get_dimension_stereotypes_idx(index, topk=topk) for index in features_index]
    
    return dim, features_importance
