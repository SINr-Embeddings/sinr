import transfert as tf
import glob, sys, pickle
import logging

from networkit import Graph, components, community, setNumberOfThreads, getCurrentNumberOfThreads, getMaxNumberOfThreads, Partition

import networkit
import sinr.text.evaluate as ev
import sinr.graph_embeddings as ge
from os.path import isfile
from sinr.text.cooccurrence import Cooccurrence
from sinr.text.pmi import pmi_filter

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', filename="smallcorpus.log", encoding='utf-8', level=logging.DEBUG)

min_frequences = { 
        "BNC": 50,
        "OANC_raw": 20
        }

def load_cooc_matrix(sentences,matrix,corpus_name,min_frequences):
    """ Utility function computing the cooccurrence matrix from the corpus, if not done yet.

    :param matrix: Path to the file containing the cooccurrence matrix
    :param corpus: Path to the file containing the corpus (.VRT)

    """
    if not isfile(matrix):
        c = Cooccurrence()
        c.fit(sentences, window=100)
        c.matrix = pmi_filter(c.matrix)
        c.save(matrix)

if __name__ == "__main__":
    for corpus_2 in glob.glob(sys.argv[2]+"*.pkl"):
        if (len(sys.argv)!=3):
            logger.error(f"Usage: {sys.argv[0]} path_to_big_corpus path_to_small_corpus")
            exit(1)

        corpus_1 = sys.argv[1]

        # Generating corpus name and path to matrix
        corpus_1_name, path_to_matrix_1 = tf.parse_path(corpus_1)
        corpus_2_name, path_to_matrix_2 = tf.parse_path(corpus_2)

        # Creating the cooccurrence matrix if not existing
        #logger.debug(f"Loading cooccurrence matrices for {corpus_1_name} and {corpus_2_name}")
        tf.create_cooc_matrix(path_to_matrix_1, corpus_1, corpus_1_name, min_frequences)
        with open(corpus_2, "rb") as f:
            load_cooc_matrix(pickle.load(f), path_to_matrix_2, corpus_2_name, min_frequences)
        #tf.create_cooc_matrix(path_to_matrix_2, corpus_2, corpus_2_name, min_frequences)

        # Saving the embeddings for the sake of performance
        sinr_1, sinr_vectors_1 = tf.compute_communities_and_vectors(path_to_matrix_1, corpus_1_name, ".pk")
        sinr_2, sinr_vectors_2 = tf.compute_communities_and_vectors(path_to_matrix_2, corpus_2_name, ".pk")

        # Evaluating the similarity for the small corpus using communities learned on itself
        #logger.debug(f"Evaluating similarity for {corpus_2_name}")
        logger.debug(tf.compute_similarity(sinr_vectors_2,corpus_2_name))

        # Evaluating the similarity for the small corpus using communities learned on the big one
        corpus_transferred = corpus_2_name+"_transferred"
        sinr_transferred = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
        sinr_transferred.transfert_communities_labels(sinr_vectors_1.get_communities_as_labels_sets())
        sinr_transferred.extract_embeddings()
        sinr_vectors_transferred = ge.InterpretableWordsModelBuilder(sinr_transferred, corpus_transferred, n_jobs=40, n_neighbors=4).build()
        transferred_communities = sinr_transferred.get_communities()
        #logger.debug(f"{corpus_transferred} has {len([i for i in transferred_communities.subsetSizes() if i==1])} singleton communities for {transferred_communities.numberOfSubsets()} communities")

        #logger.debug(f"Evaluating similarity for {corpus_transferred}")
        logger.debug(tf.compute_similarity(sinr_vectors_transferred, corpus_transferred))

        # Giving the precomputed communities as a seed to label propagation to see if this helps
        corpus_refined = corpus_2_name+"_transferred_refined"
        initial_partition = sinr_transferred.get_communities()
        refined_communities = tf.refine(sinr_transferred.get_cooc_graph(), initial_partition, "louvain")
        #logger.debug(f"{corpus_refined} has {len([i for i in refined_communities.subsetSizes() if i==1])} singleton communities for {refined_communities.numberOfSubsets()} communities")
       
        sinr_refined = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
        sinr_refined.extract_embeddings(refined_communities)
        sinr_vectors_refined = ge.InterpretableWordsModelBuilder(sinr_refined, corpus_refined, n_jobs=40, n_neighbors=4).build()

        #logger.debug(f"Evaluating similarity for {corpus_refined}")
        logger.debug(tf.compute_similarity(sinr_vectors_refined,corpus_refined))

