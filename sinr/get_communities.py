from sinr.text.cooccurrence import Cooccurrence
from sinr.text.pmi import pmi_filter
from sinr.text.preprocess import extract_text
from os.path import isfile

import sinr.text.evaluate as ev
import sinr.graph_embeddings as ge
import sys, pickle

if __name__=="__main__":

    if (len(sys.argv)!=3):
        print(f"Usage: {sys.argv[0]} path_to_big_corpus path_to_small_corpus")
        exit(1)

    # Generating corpus name and path to matrix
    corpus_1 = sys.argv[1]
    path_to_corpus_1 = "/".join(corpus_1.split("/")[:-1])+"/"
    corpus_1_name = (corpus_1.split("/")[-1]).split(".")[0]
    path_to_matrix_1 = path_to_corpus_1+corpus_1_name+"_matrix.pk"
    path_to_communities_1 = path_to_corpus_1+corpus_1_name+"_communities.pk"
    
    corpus_2 = sys.argv[2]
    path_to_corpus_2 = "/".join(corpus_2.split("/")[:-1])+"/"
    corpus_2_name = (corpus_2.split("/")[-1]).split(".")[0]
    path_to_matrix_2 = path_to_corpus_2+corpus_2_name+"_matrix.pk"
    
    # Creating the cooccurrence matrix if not existing
    if not isfile(path_to_matrix_1):
        c = Cooccurrence()
        sentences = extract_text(corpus_1, lemmatize=True, lower_words=True, number=False,punct=False, en='chunking', min_freq=50, alpha=True, min_length_word=3)
        c.fit(sentences, window=100)
        c.matrix = pmi_filter(c.matrix)
        c.save(path_to_matrix_1)

    if not isfile(path_to_matrix_2):
        c = Cooccurrence()
        sentences = extract_text(corpus_2, lemmatize=True, lower_words=True, number=False,punct=False, en='chunking', min_freq=20, alpha=True, min_length_word=3)
        c.fit(sentences, window=100)
        c.matrix = pmi_filter(c.matrix)
        c.save(path_to_matrix_2)

    # Saving the embeddings for the sake of performance
    if not isfile(corpus_1_name+".pk"):
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix_1)
        communities = sinr.detect_communities(gamma=50)
        sinr.extract_embeddings()
        sinr_vectors_1 = ge.InterpretableWordsModelBuilder(sinr, corpus_1_name, n_jobs=40, n_neighbors=4).build()
        sinr_vectors_1.save()
    else:
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix_1)
        sinr_vectors_1 = ge.SINrVectors(corpus_1_name)
        sinr_vectors_1.load()

    if not isfile(corpus_2_name+".pk"):
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
        communities = sinr.detect_communities(gamma=50)
        sinr.extract_embeddings()
        sinr_vectors_2 = ge.InterpretableWordsModelBuilder(sinr, corpus_2_name, n_jobs=40, n_neighbors=4).build()
        sinr_vectors_2.save()
    else:
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
        sinr_vectors_2 = ge.SINrVectors(corpus_2_name)
        sinr_vectors_2.load()

    sinr_vectors_2.sparsify(100)
    similarity = ev.similarity_MEN_WS353_SCWS(sinr_vectors_2)
    print(f"{corpus_2_name}: {similarity}")

    sinr.transfert_communities_labels(sinr_vectors_1.get_communities_as_labels_sets())
    sinr.extract_embeddings()
    sinr_vectors_transferred = ge.InterpretableWordsModelBuilder(sinr, corpus_2_name, n_jobs=40, n_neighbors=4).build()

    similarity = ev.similarity_MEN_WS353_SCWS(sinr_vectors_transferred)
    print(f"{corpus_2_name} transferred: {similarity}")

    # Giving the precomputed communities to label propagation to see if this helps
    initial_partition = sinr.get_communities()
    similarity = ev.similarity_MEN_WS353_SCWS(sinr_vectors_transferred)
    print(f"{corpus_2_name} transferred: {similarity}")

    while(True):
        s = input()
        print(sinr_vectors_transferred.most_similar(s))
