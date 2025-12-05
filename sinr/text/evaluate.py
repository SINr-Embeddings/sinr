import shutil
import zipfile
import numpy as np
from numpy.linalg import norm
import scipy
from scipy import stats
from sklearn.datasets._base import Bunch
import sklearn.metrics as metrics
import pandas as pd
import urllib.request
import os
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import time
import xgboost as xgb
import json
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import matplotlib.pyplot as plt
from pathlib import Path

import sinr.graph_embeddings as ge
from joblib import Parallel, delayed
from statistics import mean 


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

def find_txt_files(directory):
    """Find all text files in a directory and its subdirectories.
    
    :param directory: path to the directory
    :type directory: str
    
    :return: list of text files
    :rtype: list
    
    """
    
    return Path(directory).rglob('*.txt')

def remove_invalid_lines(content):
    """Remove invalid lines from the content.
    
    :param content: content of the file
    :type content: str
    
    :return: cleaned content
    :rtype: str
    
    """
    
    lines = content.splitlines()
    return '\n'.join(line.strip() for line in lines if '/' not in line)

def format_lines(content):
    """Format lines of the content.
    
    :param content: content of the file
    :type content: str
    
    :return: formatted content
    :rtype: str
    
    """
    
    words = content.split()
    formatted_lines = []
    for i in range(0, len(words), 4):
        if i + 4 <= len(words):
            formatted_lines.append(' '.join(words[i:i+4]))
    return '\n'.join(formatted_lines)

def fetch_analogy(langage):
    """Fetch dataset for testing analogies
    
    :param langage: language of the dataset
    :type langage: str
    
    :return: dictionary-like object. Keys of interest:
                'X': matrix of 4 words per column
    :rtype: sklearn.datasets.base.Bunch
    
    """
    
    if langage == 'fr':
        file_url = 'https://p-lux4.pcloud.com/D4ZTKr4DFZbBl110ZZZTojfXkZ2ZZosLZkZxKzZ17ZzkZf7ZOn0J7Z9zGco5qAJ3J0iMfwMoEtUFhyrsmy/BATS_3.0.zip'
    elif langage == 'en':
        file_url = 'https://p-lux4.pcloud.com/D4ZTKr4DFZbBl110ZZZTojfXkZ2ZZosLZkZxKzZ17ZzkZf7ZOn0J7Z9zGco5qAJ3J0iMfwMoEtUFhyrsmy/BATS_3.0.zip'
    else:
        raise ValueError("Language is not recognized.")

    file_name = 'dataset' + str(round(time.time() * 1000))
    file_path = file_name + '.txt'  # Name of download file (TXT or CSV)
    output_dir = file_name
    data = file_name + '_merged_cleaned.txt'

    with urllib.request.urlopen(file_url) as response:
        with open(file_path, 'wb') as file:
            file.write(response.read())

    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        os.remove(file_path)
        print(f"Temporary file deleted : {file_path}")

        txt_files = find_txt_files(output_dir)

        if txt_files:
            with open(data, 'w', encoding='utf-8') as merged_file:
                for txt_file in txt_files:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        cleaned_content = remove_invalid_lines(content)
                        formatted_content = format_lines(cleaned_content)
                        if formatted_content.strip():
                            merged_file.write(formatted_content)
                            merged_file.write('\n')

            print(f"Clean merge completed: file created -> {data}")
        else:
            raise RuntimeError("No text files found in the ZIP archive.")

        shutil.rmtree(output_dir)
        print(f"Temporary folder deleted : {output_dir}")
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned_content = remove_invalid_lines(content)
            formatted_content = format_lines(cleaned_content)
            
            if formatted_content.strip():
                with open(data, 'w', encoding='utf-8') as merged_file:
                    merged_file.write(formatted_content)

        print(f"Text file processing complete: file created -> {data}")

    try:
        data_df = pd.read_csv(data, sep='\s+', header=None, names=['A', 'B', 'C', 'D'])
    except Exception as e:
        raise RuntimeError(f"Error reading the merged file: {e}")

    data = Bunch(
        X=data_df.values.tolist()
    )

    return data

def normalize_vector(vector):
    """
    Normalize a vector.
    
    :param vector: vector to normalize
    :type vector: numpy.ndarray
    
    :return: normalized vector
    :rtype: numpy.ndarray
    """
    return vector / np.linalg.norm(vector)

def best_predicted_word(sinr_vec, word_a, word_b, word_c):
    """Solve analogy of the type A is to B as C is to D

    :param sinr_vec: SINrVectors object
    
    :param word_a: string
    :param word_b: string
    :param word_c: string
    
    :return: best predicted word of the dataset (word D) or None if not in the vocab.
    """
    
    if word_a in sinr_vec.vocab and word_b in sinr_vec.vocab and word_c in sinr_vec.vocab:
        vector_a = sinr_vec.get_my_vector(word_a)
        vector_b = sinr_vec.get_my_vector(word_b)
        vector_c = sinr_vec.get_my_vector(word_c)

        result_vector = normalize_vector(vector_b) - normalize_vector(vector_a) + normalize_vector(vector_c)

        similarities = cosine_similarity(result_vector.reshape(1, -1), sinr_vec.vectors).flatten()
        excluded_indices = [sinr_vec.vocab.index(word) for word in [word_a, word_b, word_c]]
        for idx in excluded_indices:
            similarities[idx] = -np.inf
    
        best_index = np.argmax(similarities)
        return sinr_vec.vocab[best_index]
    else:
        return None

def eval_analogy(sinr_vec, dataset, analogy_func):
    """Compare the predicted with the expected word.
    
    :param sinr_vec: SINrVectors object
    :param dataset: sklearn.datasets.base.Bunch
                    dictionary-like object. Keys of interest:
                    'X': matrix of 2 words per column,
                    'y': vector with scores
    
    :return: error rate
    :rtype: float
    """
    with open(dataset, 'r') as f:
        lines = f.readlines()

    valid_analogies = []

    for line in lines:
        line = line.strip()
        if line.startswith(':'):
            continue
        
        words = line.split()

        word_a, word_b, word_c, expected = words

        if (word_a in sinr_vec.vocab 
            and word_b in sinr_vec.vocab
            and word_c in sinr_vec.vocab
            and expected in sinr_vec.vocab):
            valid_analogies.append(line)

    total_analogies = 0
    correct_count = 0
    incorrect_count = 0

    for line in valid_analogies:
        word_a, word_b, word_c, expected = line.split()
        
        predicted_word = analogy_func(sinr_vec, word_a, word_b, word_c)
        if predicted_word is not None:
            total_analogies += 1
            if predicted_word == expected:
                correct_count += 1
            else:
                incorrect_count += 1
            
    error_rate = incorrect_count / total_analogies if total_analogies > 0 else 0

    print("\n=== Summary ===")
    print(f"Total valid analogies processed: {total_analogies}")
    print(f"Correct analogies: {correct_count}")
    print(f"Incorrect analogies: {incorrect_count}")
    print(f"Error rate: {error_rate:.2%}")

    return error_rate

def compute_analogy_value_zero(sinr_vec, word_a, word_b, word_c):
    """Solve analogy of the type A is to B as C is to D with only positives values in the resulting vector

    :param sinr_vec: SINrVectors object

    :param word_a: string
    :param word_b: string
    :param word_c: string

    :return: best predicted word of the dataset (word D) or None if not in the vocab.
    :rtype: string
    """
    if word_a in sinr_vec.vocab and word_b in sinr_vec.vocab and word_c in sinr_vec.vocab:

        vector_a = sinr_vec.vectors[sinr_vec.vocab.index(word_a)]
        vector_b = sinr_vec.vectors[sinr_vec.vocab.index(word_b)]
        vector_c = sinr_vec.vectors[sinr_vec.vocab.index(word_c)]

        result_vector = vector_c - vector_a + vector_b

        result_vector[result_vector < 0] = 0

        similarities = cosine_similarity(result_vector.reshape(1, -1), sinr_vec.vectors).flatten()

        excluded_indices = [sinr_vec.vocab.index(word) for word in [word_a, word_b, word_c]]
        for idx in excluded_indices:
            similarities[idx] = 0

        best_index = np.argmax(similarities)
        return sinr_vec.vocab[best_index]
    
    return None

def compute_analogy_normalized(sinr_vec, word_a, word_b, word_c):
    """Solve analogy of the type A is to B as C is to D with normalized values

    :param sinr_vec: SINrVectors object

    :param word_a: string
    :param word_b: string
    :param word_c: string

    :return: best predicted word of the dataset
    :rtype: string
    """
    if word_a in sinr_vec.vocab and word_b in sinr_vec.vocab and word_c in sinr_vec.vocab:

        vector_a = sinr_vec.vectors[sinr_vec.vocab.index(word_a)]
        vector_b = sinr_vec.vectors[sinr_vec.vocab.index(word_b)]
        vector_c = sinr_vec.vectors[sinr_vec.vocab.index(word_c)]

        result_vector = vector_c - vector_a + vector_b

        result_vector[result_vector < 0] = 0

        result_vector = result_vector / np.sum(result_vector)

        similarities = cosine_similarity(result_vector.reshape(1, -1), sinr_vec.vectors).flatten()

        excluded_indices = [sinr_vec.vocab.index(word) for word in [word_a, word_b, word_c]]
        for idx in excluded_indices:
            similarities[idx] = 0

        best_index = np.argmax(similarities)
        return sinr_vec.vocab[best_index]
    
    return None

def best_predicted_word_k(sinr_vec, word_a, word_b, word_c, k=1):
    """Predict the best word for the analogy A is to B as C is to D with k best words.
    
    :param sinr_vec: SINrVectors object
    
    :param word_a: string
    :param word_b: string
    :param word_c: string
    :param k: int, number of best words to return (default is 1)
    
    :return: list of k best predicted words of the dataset or None if not in the vocab.
    :rtype: list of strings
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    if word_a in sinr_vec.vocab and word_b in sinr_vec.vocab and word_c in sinr_vec.vocab:
        vector_a = sinr_vec.get_my_vector(word_a)
        vector_b = sinr_vec.get_my_vector(word_b)
        vector_c = sinr_vec.get_my_vector(word_c)

        result_vector = normalize_vector(vector_b) - normalize_vector(vector_a) + normalize_vector(vector_c)
        similarities = cosine_similarity(result_vector.reshape(1, -1), sinr_vec.vectors).flatten()
        
        excluded_indices = [sinr_vec.vocab.index(word) for word in [word_a, word_b, word_c]]
        for idx in excluded_indices:
            similarities[idx] = -np.inf
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        predicted_words = [sinr_vec.vocab[i] for i in top_indices]
        return predicted_words
    else:
        return None
    
def eval_analogy_k(sinr_vec, dataset, analogy_func, k=1):
    """Compare the predicted with the expected word with k best words.
    
    :param sinr_vec: SINrVectors object
    :param dataset: sklearn.datasets.base.Bunch
                    dictionary-like object. Keys of interest:
                    'X': matrix of 2 words per column,
                    'y': vector with scores
    
    :return: error rate
    :rtype: float
    """
    with open(dataset, 'r') as f:
        lines = f.readlines()

    valid_analogies = []

    for line in lines:
        line = line.strip()
        if line.startswith(':'):
            continue
        
        words = line.split()

        word_a, word_b, word_c, expected = words

        if (word_a in sinr_vec.vocab 
            and word_b in sinr_vec.vocab
            and word_c in sinr_vec.vocab
            and expected in sinr_vec.vocab):
            valid_analogies.append(line)

    total_analogies = 0
    correct_count = 0
    incorrect_count = 0

    for line in valid_analogies:
        word_a, word_b, word_c, expected = line.split()
        
        predicted_words = analogy_func(sinr_vec, word_a, word_b, word_c, k)
        if predicted_words is not None:
            total_analogies += 1
            if expected in predicted_words:
                correct_count += 1
            else:
                incorrect_count += 1
            
    error_rate = incorrect_count / total_analogies if total_analogies > 0 else 0

    print("\n=== Category ===")
    print(f"Total validated analogies: {total_analogies}")
    print(f"Correct analogies: {correct_count}")
    print(f"Incorrect analogies: {incorrect_count}")
    print(f"Error rate: {error_rate:.2%}")

    return error_rate


def eval_analogy_by_category_k(sinr_vec, dataset, analogy_func, k=1):
    """Evaluate analogy by category with k best words
    
    :param sinr_vec: SINrVectors object
    :param dataset: sklearn.datasets.base.Bunch
    :param analogy_func: function to use for analogy prediction
    :param k: int, number of best words to return (default is 1)
    
    :return: dictionary with categories as keys and error rates as values
    :rtype: dict
    """
    with open(dataset, 'r') as f:
        lines = f.readlines()

    current_category = "default"
    categories = {}

    for line in lines:
        line = line.strip()
        if line.startswith(':'):
            current_category = line[1:].strip()
            categories[current_category] = []
        else:
            words = line.split()
            if len(words) != 4:
                continue
            word_a, word_b, word_c, expected = [w.lower() for w in words]
            if (word_a in sinr_vec.vocab and 
                word_b in sinr_vec.vocab and 
                word_c in sinr_vec.vocab and 
                expected in sinr_vec.vocab):
                categories[current_category].append((word_a, word_b, word_c, expected))

    results = {}
    for category, analogies in categories.items():
        total = 0
        correct = 0
        incorrect = 0
        for word_a, word_b, word_c, expected in analogies:
            predicted_words = analogy_func(sinr_vec, word_a, word_b, word_c, k)
            if predicted_words is not None:
                total += 1
                if expected in predicted_words:
                    correct += 1
                else:
                    incorrect += 1
        error_rate = incorrect / total if total > 0 else 0
        results[category] = {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'error_rate': error_rate
        }
        print("\n=== Category : {} ===".format(category))
        print("Total validated analogies :", total)
        print("Correct analogies :", correct)
        print("Incorrect analogies :", incorrect)
        print("Error rate : {:.2%}".format(error_rate))
    return results

def plot_global_error_rates(sinr_vec, file_path, best_predicted_word_k, ks):
    """ Plot global error rates for different values of k
    
    :param sinr_vec: SINrVectors object
    :param file_path: path to the dataset file
    :param best_predicted_word_k: function to use for analogy prediction
    :param ks: list of k values to evaluate - [1, 2, 5, 10]
    """
    global_error_rates = []

    for k_val in ks:
        err_rate = eval_analogy_k(sinr_vec, file_path, best_predicted_word_k, k=k_val)
        global_error_rates.append(err_rate)

    global_error_rates_pct = [rate * 100 for rate in global_error_rates]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, global_error_rates_pct, marker='o')
    plt.xlabel("k")
    plt.ylabel("Global error rate (%)")
    plt.title("Global error rate k values")
    plt.xticks(ks)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_category_error_rates(sinr_vec, file_path, best_predicted_word_k, ks):
    """ Plot error rates by category for different values of k
    
    :param sinr_vec: SINrVectors object
    :param file_path: path to the dataset file
    :param best_predicted_word_k: function to use for analogy prediction
    :param ks: list of k values to evaluate - [1, 2, 5, 10]
    
    """
    category_error_rates = {}

    for k_val in ks:
        cat_results = eval_analogy_by_category_k(sinr_vec, file_path, best_predicted_word_k, k=k_val)
        for cat, metrics in cat_results.items():
            if cat not in category_error_rates:
                category_error_rates[cat] = []
            category_error_rates[cat].append(metrics['error_rate'] * 100)

    categories = list(category_error_rates.keys())
    n_categories = len(categories)
    x = np.arange(n_categories)
    width = 0.15

    plt.figure(figsize=(12, 6))
    for i, k_val in enumerate(ks):
        error_vals = [category_error_rates[cat][i] for cat in categories]
        plt.bar(x + i * width, error_vals, width, label=f'k={k_val}')

    plt.xlabel("Category")
    plt.ylabel("Global error rate (%)")
    plt.title("Global error by category for k values")
    plt.xticks(x + width * (len(ks) - 1) / 2, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

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

def project_vector(v,u):
    normalize_u = normalize_vector(u)
    return np.dot(v, normalize_u) * normalize_u

def reject_vector(v, u):
    "Compute the orthogonal projection of a vector onto a given direction."
    return v - project_vector(v,u)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def identify_gender_direction_sinr(sinr_vec, definitional_pairs, method="pca", positive_end="brother", negative_end="sister"):
    """
    Identifies the gender direction in a SINr model.

    Parameters:
    - sinr_vec: SINr model.
    - positive_end: word representing the masculine gender.
    - negative_end: word representing the feminine gender.
    - definitional_pairs: list of word pairs defining gender.
    - method: method used to compute the gender direction ('single', 'sum', 'pca').

    Returns:
    - A vector representing the gender direction.
    """
    if method == "single":
        return normalize_vector(sinr_vec.get_my_vector(positive_end) - sinr_vec.get_my_vector(negative_end))
    elif method == "sum":
        group1 = np.sum([sinr_vec.get_my_vector(w1) for w1, w2 in definitional_pairs], axis=0)
        group2 = np.sum([sinr_vec.get_my_vector(w2) for w1, w2 in definitional_pairs], axis=0)
        return normalize_vector(group1 - group2)
    elif method == "pca":
        matrix = np.array([sinr_vec.get_my_vector(w1) - sinr_vec.get_my_vector(w2) for w1, w2 in definitional_pairs])
        pca = PCA(n_components=1)
        pca.fit(matrix)
        return normalize_vector(pca.components_[0])
    else:
        raise ValueError("Invalid method. Use 'single', 'sum' or 'pca'.")

def compute_direct_bias_sinr(sinr_vec, word_list, gender_direction, c=1):
    """
    Computes the direct bias of a set of words with respect to the gender direction
    using cosine similarity.

    Args:  
        sinr_vec: SINr model.  
        word_list: List of words to analyze. (professions in config.json)
        gender_direction: Gender direction vector.  
        c: Exponent applied to cosine similarity (default is c=1).  

    Returns:  
        float: Direct bias value.
    """
    word_vectors = [sinr_vec.get_my_vector(word) for word in word_list if word in sinr_vec.vocab]
    if not word_vectors:
        return 0.0
    word_vectors = np.array(word_vectors)
    gender_direction = gender_direction.reshape(1, -1)
    cos_similarities = np.abs(cosine_similarity(word_vectors, gender_direction)).flatten()
    return np.mean(cos_similarities ** c)


def compute_indirect_bias_sinr(sinr_vec, word1, word2, direction):
    """
        Compute the indirect bias SINr model.

        :param sinr_vec: SINr model.
        :param word1: The first word.
        :param word2: The second word.
        :param direction: The gender direction.
        :return: The gender component of the similarity between the two words.
    """

    vector1 = normalize_vector(sinr_vec.get_my_vector(word1))
    vector2 = normalize_vector(sinr_vec.get_my_vector(word2))


    gender_component1 = np.dot(vector1, direction) * direction
    gender_component2 = np.dot(vector2, direction) * direction


    orthogonal_vector1 = reject_vector(vector1, direction)
    orthogonal_vector2 = reject_vector(vector2, direction)


    inner_product = np.dot(vector1, vector2)

    orthogonal_vector1_2d = orthogonal_vector1.reshape(1, -1)
    orthogonal_vector2_2d = orthogonal_vector2.reshape(1, -1)
    

    orthogonal_similarity = cosine_similarity(orthogonal_vector1_2d, orthogonal_vector2_2d)[0][0]

    indirect_bias = ((inner_product - orthogonal_similarity) / inner_product)

    return indirect_bias

def compute_analogy_sparse_normalized(sinr_vec, word_a, word_b, word_c, n=100):
    """Solve analogy of the type A is to B as C is to D with sparsification and normalization.

    :param sinr_vec: SINrVectors object
    :param word_a: string
    :param word_b: string
    :param word_c: string
    :param n: int, number of dimensions to keep after sparsification

    :return: best predicted word of the dataset (word D) or None if not in the vocab.
    :rtype: string
    """
    if word_a in sinr_vec.vocab and word_b in sinr_vec.vocab and word_c in sinr_vec.vocab:

        vector_a = sinr_vec.vectors[sinr_vec.vocab.index(word_a)].toarray().flatten()
        vector_b = sinr_vec.vectors[sinr_vec.vocab.index(word_b)].toarray().flatten()
        vector_c = sinr_vec.vectors[sinr_vec.vocab.index(word_c)].toarray().flatten()


        result_vector = vector_c - vector_a + vector_b

        #Sparsification: keep only the n highest values
        sparse_vector = np.zeros_like(result_vector)
        top_indices = np.argsort(result_vector)[-n:]
        sparse_vector[top_indices] = result_vector[top_indices]

        #Normalization: ensure the sum is 1
        sparse_vector = np.maximum(sparse_vector, 0)
        sparse_vector = sparse_vector / np.sum(sparse_vector) if np.sum(sparse_vector) > 0 else sparse_vector

        #print(f"Sparse Vector (normalized): {sparse_vector}")
        #print(f"Sum of dimensions: {np.sum(sparse_vector)}")

        similarities = cosine_similarity(sparse_vector.reshape(1, -1), sinr_vec.vectors).flatten()
        excluded_indices = [sinr_vec.vocab.index(word) for word in [word_a, word_b, word_c]]
        for idx in excluded_indices:
            similarities[idx] = 0

        best_index = np.argmax(similarities)
        return sinr_vec.vocab[best_index]
    return None


def varnn(set1, set2, k=25):
    """
    Computes varnn metrics from two neighbor sets.
    
    :param set1: set of neighbors (e.g. from model1)
    :type set1: set
    :param set2: set of neighbors (e.g. from model2)
    :type set2: set
    :param k: number of neighbors used to compute the sets (default 25)
    :type k: int
    :returns: pierrejean score and flat score
    :rtype: tuple
    """
    intersection_len = len(set1.intersection(set2))
    metric_pierrejean = 1 - (intersection_len / len(set1))
    metric_flats = k - intersection_len
    
    return (metric_pierrejean, metric_flats)

def varnn_from_models(model1, model2, word, k=25, nn1=None, nn2=None):
    """Computes varnn for each word is model1 also present in model2
    
    :param model1: a SINrVectors object
    :type model1: SINrVectors
    :param model2: a second SINrVectors object
    :type model2: SINrVectors:
    :param word: an item in the vocabulary of the first model
    :type word: str
    :param k: number of neighbors used to compute the sets (default 25)
    :type k: int
    :param nn1: NearestNeighbors object for model1 (default None)
    :type nn1: NearestNeighbors
    :param nn2: NearestNeighbors object for model2 (default None)
    :type nn2: NearestNeighbors
    :returns: Results of varnn between model1 and model2 on a single word as pierrejean score, the 1 - the proportion of shared neighbors, and a flat score the number of different neighbors
    :rtype: tuple

    """
    index1 = model1._get_index(word)
    index2 = model2._get_index(word)
    
    if nn1 is None:
        nn1 = NearestNeighbors(n_neighbors=k, metric='cosine').fit(model1.vectors)
    if nn2 is None:
        nn2 = NearestNeighbors(n_neighbors=k, metric='cosine').fit(model2.vectors)

    _, idxs1 = nn1.kneighbors(model1.vectors[index1])
    _, idxs2 = nn2.kneighbors(model2.vectors[index2])

    neighbors1 = set(model1.vocab[i] for i in idxs1.flatten()[1:])
    neighbors2 = set(model2.vocab[i] for i in idxs2.flatten()[1:])
    
    return varnn(neighbors1, neighbors2, k)

def varnn_across_models(model1, model2, k=25):
    """Computes varnn for each word is model1 also present in model2
    
    :param model1: a SINrVectors object
    :type model1: SINrVectors
    :param model2: a second SINrVectors object
    :type model2: SINrVectors:
    :param k: number of neighbors used to compute the sets (default 25)
    :type k: int
    :returns: Results of varnn on the total vocabulary of model1
    :rtype: list of tuples

    """
    # Determining common vocabulary
    shared_vocabulary = set(model1.vocab).intersection(set(model2.vocab))
    
    nn1 = NearestNeighbors(n_neighbors=k, metric='cosine').fit(model1.vectors)
    nn2 = NearestNeighbors(n_neighbors=k, metric='cosine').fit(model2.vectors)
    
    res = Parallel(n_jobs=-1)(
        delayed(varnn_from_models)(model1, model2, w, k, nn1, nn2) for w in tqdm(shared_vocabulary, desc="varnn calc", unit="word")
    )
    return res

def scores_varnn(model1, model2, k=25):
    """Computes mean of pierrejean score on the shared vocabulary of model1 and model2
    
    :param model1: a SINrVectors object
    :type model1: SINrVectors
    :param model2: a second SINrVectors object
    :type model2: SINrVectors:
    :returns: A score of varnn between the two models
    :rtype: float
    """
    res_per_word=varnn_across_models(model1, model2, k)
    res_per_word_pierrejean = [i[0] for i in res_per_word]
    return mean(res_per_word_pierrejean)


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
