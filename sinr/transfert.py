from sinr.text.cooccurrence import Cooccurrence
from sinr.text.pmi import pmi_filter
from sinr.text.preprocess import extract_text
from os.path import isfile
from networkit import Graph, components, community, setNumberOfThreads, getCurrentNumberOfThreads, getMaxNumberOfThreads, Partition

import networkit
import sinr.text.evaluate as ev
import sinr.graph_embeddings as ge
import sys, pickle

min_frequences = { 
        "BNC": 50,
        "OANC_raw": 20
        }

def diff_voc(voc_1, voc_2):
    """Compute the set of words that are in voc_2 but not in voc_1

    :param voc_1: 
    :param voc_2:

    """

    return voc_2 - voc_1

def singletons(voc, sinr_object, communities):
    number_of_communities = communities.numberOfSubsets()
    result = list()
    for community in range(number_of_communities):
        members = communities.getMembers(community)
        if(len(members)==1):
            member = members[0]
            if member in voc:
                result.append(voc)
    return result

def parse_path(path):
    """ Utility function parsing the relative/absolute path given as argument to 
        extract the corpus' name and the path to the file saving the cooccurrence matrix

        :param path: Path to the file saving the cooccurrence matrix
    
    """
    corpus = path
    path_to_corpus = "/".join(corpus.split("/")[:-1])+"/"
    corpus_name = (corpus.split("/")[-1]).split(".")[0]
    path_to_matrix = path_to_corpus+corpus_name+"_matrix.pk"
    return corpus_name, path_to_matrix

def create_cooc_matrix(matrix,corpus,corpus_name,min_frequences):
    """ Utility function computing the cooccurrence matrix from the corpus, if not done yet.

    :param matrix: Path to the file containing the cooccurrence matrix
    :param corpus: Path to the file containing the corpus (.VRT)

    """
    if not isfile(matrix):
        c = Cooccurrence()
        sentences = extract_text(corpus, exceptions_path="../sinr/text/exception_for_similarity.txt", lemmatize=True, lower_words=True, number=False,punct=False, en='chunking', min_freq=min_frequences[corpus_name], alpha=True, min_length_word=3)
        c.fit(sentences, window=100)
        c.matrix = pmi_filter(c.matrix)
        c.save(matrix)

# TODO --- split into compute_communities and compute_vectors for better modularity 
# (also change the name of path_to_matrix, maybe not adapted)
def compute_communities_and_vectors(path_to_matrix, corpus_name, extension):
    """Returns SINr and SINrVectors objects created from the cooccurrence matrix given as argument

    :param path_to_matrix: Path to the pickle file containing the cooccurrence matrix
    """
    if not isfile(corpus_name+extension):
        # TODO -- passer le logger en argument
        #logger.info(f"Computing communities and embeddings for {corpus_name}")
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix)
        communities = sinr.detect_communities(gamma=50)
        sinr.extract_embeddings()
        sinr_vectors = ge.InterpretableWordsModelBuilder(sinr, corpus_name, n_jobs=40, n_neighbors=4).build()
        sinr_vectors.save()
        initial_partition = sinr.get_communities()
        #logger.info(f"{corpus_name} has {len([i for i in initial_partition.subsetSizes() if i==1])} singleton communities for {initial_partition.numberOfSubsets()} communities")
    else:
        #logger.warning(f"Loading communities and embeddings of {corpus_name} for the sake of performance: SHOULD BE COMPUTED AT ANY RUN")
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix)
        sinr_vectors = ge.SINrVectors(corpus_name)
        sinr_vectors.load()

    return sinr, sinr_vectors

def compute_similarity(sinr_vectors, corpus_name):
    sinr_vectors.sparsify(100)
    similarity = ev.similarity_MEN_WS353_SCWS(sinr_vectors)
    return f"{corpus_name}: {similarity}"

def refine(graph, communities, algorithm):
    def _leiden():
        from cdlib import algorithms, NodeClustering
        networkx_graph = networkit.nxadapter.nk2nx(graph)
        
        coms = algorithms.leiden(networkx_graph, weights='weight', initial_membership=communities.getVector())

        # Compute a map node<->list of communities
        community_map = NodeClustering.to_node_community_map(coms)

        refined_communities = networkit.structures.Partition(len(networkx_graph.nodes))
        refined_communities.allToSingletons()
        for node, community in community_map.items():
            refined_communities.moveToSubset(community[0],node)

        return refined_communities

    def _louvain():
        from cdlib import algorithms, NodeClustering
        networkx_graph = networkit.nxadapter.nk2nx(graph)

        list_of_communities = [list() for _ in range(communities.numberOfSubsets())]
        for index,community in enumerate(communities.getVector()):
            list_of_communities[community].append(index)

        nc = NodeClustering(list_of_communities, networkx_graph, algorithm)
        coms = algorithms.louvain(networkx_graph, weight='weight', resolution=25, partition=nc)

        # Compute a map node<->list of communities
        community_map = NodeClustering.to_node_community_map(coms)

        refined_communities = networkit.structures.Partition(len(networkx_graph.nodes))
        refined_communities.allToSingletons()
        for node, community in community_map.items():
            refined_communities.moveToSubset(community[0],node)

        return refined_communities

    algorithms = {
            "leiden": _leiden, 
            "louvain": _louvain,
            }

    return algorithms[algorithm]()    

if __name__=="__main__":

    if (len(sys.argv)!=3):
        logger.error(f"Usage: {sys.argv[0]} path_to_big_corpus path_to_small_corpus")
        exit(1)

    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', stream=sys.stderr, encoding='utf-8', level=logging.INFO)
    corpus_1, corpus_2 = sys.argv[1], sys.argv[2]

    # Generating corpus name and path to matrix
    corpus_1_name, path_to_matrix_1 = parse_path(corpus_1)
    corpus_2_name, path_to_matrix_2 = parse_path(corpus_2)

    # Creating the cooccurrence matrix if not existing
    logger.info(f"Loading cooccurrence matrices for {corpus_1_name} and {corpus_2_name}")
    create_cooc_matrix(path_to_matrix_1, corpus_1, corpus_1_name, min_frequences)
    create_cooc_matrix(path_to_matrix_2, corpus_2, corpus_2_name, min_frequences)

    # Saving the embeddings for the sake of performance
    sinr_1, sinr_vectors_1 = compute_communities_and_vectors(path_to_matrix_1, corpus_1_name, ".pk")
    sinr_2, sinr_vectors_2 = compute_communities_and_vectors(path_to_matrix_2, corpus_2_name, ".pk")

    # Evaluating the similarity for the small corpus using communities learned on itself
    #logger.info(f"Evaluating similarity for {corpus_2_name}")
    #logger.info(compute_similarity(sinr_vectors_2,corpus_2_name))

    # Evaluating the similarity for the small corpus using communities learned on the big one
    corpus_transferred = corpus_2_name+"_transferred"
    sinr_transferred = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
    sinr_transferred.transfert_communities_labels(sinr_vectors_1.get_communities_as_labels_sets())
    sinr_transferred.extract_embeddings()
    sinr_vectors_transferred = ge.InterpretableWordsModelBuilder(sinr_transferred, corpus_transferred, n_jobs=40, n_neighbors=4).build()
    transferred_communities = sinr_transferred.get_communities()
    logger.info(f"{corpus_transferred} has {len([i for i in transferred_communities.subsetSizes() if i==1])} singleton communities for {transferred_communities.numberOfSubsets()} communities")

    #logger.info(f"Evaluating similarity for {corpus_transferred}")
    #logger.info(compute_similarity(sinr_vectors_transferred, corpus_transferred))

    # Giving the precomputed communities as a seed to label propagation to see if this helps
    corpus_refined = corpus_2_name+"_transferred_refined"
    initial_partition = sinr_transferred.get_communities()
    refined_communities = refine(sinr_transferred.get_cooc_graph(), initial_partition, "louvain")
    logger.info(f"{corpus_refined} has {len([i for i in refined_communities.subsetSizes() if i==1])} singleton communities for {refined_communities.numberOfSubsets()} communities")
   
    sinr_refined = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
    sinr_refined.extract_embeddings(refined_communities)
    sinr_vectors_refined = ge.InterpretableWordsModelBuilder(sinr_refined, corpus_refined, n_jobs=40, n_neighbors=4).build()

    logger.info(f"Evaluating similarity for {corpus_refined}")
    logger.info(compute_similarity(sinr_vectors_refined,corpus_refined))
