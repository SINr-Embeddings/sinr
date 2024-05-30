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

gamma = { 
        "BNC": 80,
        "OANC_raw": 50,
        "oanc_extracted": 50,
        "acl_extracted": 50
        }

def parse_corpus_path(path, extension=".pk"):
    """ Utility function parsing the relative/absolute path given as argument (the large corpus)
        to extract the corpus' name and the path to the file saving the cooccurrence matrix
        The folder text/ contains the result of sinr.extract_text and matrix contains 
        the saved cooccurence matrix.

        :param path: Path to the file saving the cooccurrence matrix
    
    """
    from os import makedirs
    
    corpus = path
    path_to_corpus = "/".join(corpus.split("/")[:-2])+"/"
    corpus_name = (corpus.split("/")[-1]).split(".")[0]

    finalpath = path_to_corpus

    path_to_matrix = finalpath+"matrix/"
    path_to_text = finalpath+"text/"

    makedirs(path_to_matrix, exist_ok=True)

    return corpus_name, path_to_text+corpus_name+extension, path_to_matrix+corpus_name+extension

def parse_corpuses_path(path, corpus, extension=".pk"):
    """ Utility function parsing the relative/absolute path given as argument (the small corpuses) 
        to extract the corpuses' names and the path to the file saving the cooccurrence matrix
        The folder text/ contains the result of sinr.extract_text and matrix contains 
        the saved cooccurence matrix.

        :param path: Path to the file saving the cooccurrence matrix
    
    """
    from os import makedirs
   
    corpus_name = corpus.split("/")[-2]
    path_to_matrix = path+"matrix/"
    path_to_text = path+"text/"

    makedirs(path_to_matrix, exist_ok=True)
    
    return corpus_name, path_to_text+corpus_name+extension, path_to_matrix+corpus_name+extension

def create_cooc_matrix(path_to_text,path_to_matrix,corpus_name):
    """ Utility function computing the cooccurrence matrix from the extracted text and 
        saving it as a pickle file, if not done yet. 

    :param path_to_text: Path to the file containing the extracted text
    :param path_to_matrix: Path to the file containing the cooccurrence matrix
    :param corpus_name: The name of the corpus

    """
    if not isfile(path_to_matrix):
        c = Cooccurrence()
        with open(path_to_text, "rb") as f:
            c.fit(pickle.load(f), window=100)
            c.matrix = pmi_filter(c.matrix)
            c.save(path_to_matrix)

# TODO --- split into compute_communities and compute_vectors for better modularity 
# (also change the name of path_to_matrix, maybe not adapted)
def compute_communities_and_vectors(path_to_matrix, corpus_name, gamma=50, extension=".pk", logger=None):
    """Returns SINr and SINrVectors objects created from the cooccurrence matrix given as argument

    :param path_to_matrix: Path to the pickle file containing the cooccurrence matrix
    :param corpus_name: The name of the corpus
    :param extension: String representing the extension of the matrices (".pk" by default)
    :param gamma: Resolution parameter for Louvain’s algorithm (default to 50)
    """
    if not isfile(corpus_name+extension):
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix)
        communities = sinr.detect_communities(gamma=gamma)
        sinr.extract_embeddings()
        vectors = ge.InterpretableWordsModelBuilder(sinr, corpus_name, n_jobs=40, n_neighbors=4).build()
        # TODO -- ugly trick to avoid extracting embeddings for the larger BNC corpus every time
        if ("BNC" in corpus_name):
            vectors.save()
    else: 
        # TODO -- should not enter here unless the corpus is BNC
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix)
        vectors = ge.SINrVectors(corpus_name)
        vectors.load()

    return sinr, vectors

def compute_similarity(vectors, corpus_name):
    """Compute the similarity for a given set of embeddings

    :param vectors: Embeddings of the corpus
    :typev vectors: SINrVectors object
    :param corpus_name: The name of the corpus

    """
    vectors.sparsify(100)
    similarity = ev.similarity_MEN_WS353_SCWS(vectors)
    output = {dataset: f"{value:9.4f}" for dataset, value in similarity.items()}
    return f"{corpus_name} {output}"

def refine(graph, initial_membership, gamma=50):
    """Refining communities for a given graph starting from an initial membership partition

    :param graph: The graph to compute communities for
    :typev graph: networkit object
    :param initial_membership: The initial membership partition
    :typev initial_membership: networkit.Partition object
    :param gamma: Resolution parameter for Louvain’s algorithm (default to 50)

    """

    from cdlib import algorithms, NodeClustering
    networkx_graph = networkit.nxadapter.nk2nx(graph)

    list_of_communities = [list() for _ in range(initial_membership.numberOfSubsets())]
    for index,community in enumerate(initial_membership.getVector()):
        list_of_communities[community].append(index)

    nc = NodeClustering(list_of_communities, networkx_graph, algorithm)
    coms = algorithms.louvain(networkx_graph, weight='weight', resolution=gamma, partition=nc)

    # Compute a map node<->list of communities
    community_map = NodeClustering.to_node_community_map(coms)

    refined_communities = networkit.structures.Partition(len(networkx_graph.nodes))
    refined_communities.allToSingletons()
    for node, community in community_map.items():
        refined_communities.moveToSubset(community[0],node)

    return refined_communities

if __name__=="__main__":

    import logging, glob
    import calendar, time

    # Current GMT time in a tuple format
    current_GMT = time.gmtime()
    # ts stores timestamp
    ts = calendar.timegm(current_GMT) 

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(message)s', filename="stats/run-"+str(ts)+".txt", encoding='utf-8', level=logging.WARNING, filemode="w")
    
    if (len(sys.argv)!=4):
        logger.error(f"Usage: {sys.argv[0]} path_to_big_corpus path_to_small_corpuses (this is probably not a word but you get the idea) number_of_desired_strategies (should be a list)")
        exit(1)

    def _log_communities(logger, communities, corpus):
        sizes = communities.subsetSizes()
        logger.info(f"{corpus} has {len(sizes)} communities of maximum size {max(sizes)} and of average size {sum(sizes)/len(sizes)}")

    strategies = range(1,int(sys.argv[3])+1)
    strategies_names = {
            1: "adding existing edges", 
            2: "adding all edges",
            3: "adding non existing edges"
            }
    # TODO -- uniformize extensions or have them as arguments
    for corpus_2 in glob.glob(sys.argv[2]+"text/*.pk*"):

        corpus_1 = sys.argv[1]

        # Generating corpus name and path to matrix
        corpus_1_name, path_to_text_1, path_to_matrix_1 = parse_corpuses_path(sys.argv[1],corpus_1, ".pkl")
        corpus_2_name, path_to_text_2, path_to_matrix_2 = parse_corpus_path(corpus_2, ".pkl")

        # Creating the cooccurrence matrix if not existing
        logger.info(f"Loading cooccurrence matrices for {corpus_1_name} and {corpus_2_name}")
        create_cooc_matrix(path_to_text_1, path_to_matrix_1, corpus_1_name)
        create_cooc_matrix(path_to_text_2, path_to_matrix_2, corpus_2_name)

        # Saving the embeddings for the sake of performance
        sinr_1, vectors_1 = compute_communities_and_vectors(path_to_matrix_1, corpus_1_name, ".pk", gamma[corpus_1_name])
        sinr_2, vectors_2 = compute_communities_and_vectors(path_to_matrix_2, corpus_2_name, ".pk", 50)
        
        logger.warning(f"=============== {corpus_2_name} ===============")
        #_log_communities(logger, sinr_2.get_communities(), "original louvain: ")
        logger.warning(compute_similarity(vectors_2, "original louvain: "))

        #corpus_weights_transferred = corpus_2_name+"_with_weights_transferred"
        for strategy in strategies:  
            logger.warning(f"=============== {strategies_names[strategy]} ===============")
            # [BEGIN] Computing communities on the small corpus after transferring the weights of the large corpus
            corpus_with_weights_transferred = f"louvain with weigths transferred: "
            sinr_with_weights_transferred = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
            sinr_with_weights_transferred.ponderate_graph(sinr_1, strategy)

            communities_with_weights_transferred = sinr_with_weights_transferred.detect_communities(gamma=50)
            sinr_with_weights_transferred.extract_embeddings()
            vectors_with_weights_transferred = ge.InterpretableWordsModelBuilder(sinr_with_weights_transferred, corpus_2_name, n_jobs=40, n_neighbors=4).build()

            logger.info(f"{corpus_with_weights_transferred} has {len([i for i in communities_with_weights_transferred.subsetSizes() if i==1])} singleton communities for {communities_with_weights_transferred.numberOfSubsets()} communities")
           
            _log_communities(logger,communities_with_weights_transferred,corpus_with_weights_transferred)
            logger.warning(compute_similarity(vectors_with_weights_transferred,corpus_with_weights_transferred))
            # [END] Computing communities on the small corpus after transferring the weights of the large corpus

            # [BEGIN] Evaluating the similarity for the small corpus using weights and communities learned on the big one
            #corpus_communities_weights_transferred = corpus_2_name+"_transferred"
            corpus_communities_and_weights_transferred = "communities and weights transferred: "
            sinr_communities_and_weights_transferred = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
            sinr_communities_and_weights_transferred.ponderate_graph(sinr_1, strategy)

            sinr_communities_and_weights_transferred.transfert_communities_labels(vectors_1.get_communities_as_labels_sets())
            sinr_communities_and_weights_transferred.extract_embeddings()

            vectors_communities_and_weights_transferred = ge.InterpretableWordsModelBuilder(sinr_communities_and_weights_transferred, corpus_communities_and_weights_transferred, n_jobs=40, n_neighbors=4).build()
            transferred_communities = sinr_communities_and_weights_transferred.get_communities()

            print("TRANSFERRED COMMUNITIES")
            print(networkit.community.inspectCommunities(transferred_communities, sinr_communities_and_weights_transferred.get_cooc_graph()))
            
            _log_communities(logger, transferred_communities, corpus_communities_and_weights_transferred)
            logger.warning(compute_similarity(vectors_communities_and_weights_transferred, corpus_communities_and_weights_transferred)+"\n")
            # [END] Evaluating the similarity for the small corpus using weights and communities learned on the big one
        
            # [BEGIN] Giving the precomputed communities as a seed to louvain on re-weighted graph 
            '''corpus_communities_and_weights_transferred_refined = "communities and weights transferred and refined: "
            initial_partition = sinr_transferred.get_communities()
            refined_communities = refine(sinr_communities_and_weights_transferred_transferred.get_cooc_graph(), initial_partition, "louvain")
            logger.info(f"{corpus_refined} has {len([i for i in refined_communities.subsetSizes() if i==1])} singleton communities for {refined_communities.numberOfSubsets()} communities")
           
            sinr_communities_and_weights_transferred_refined = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
            sinr_communities_and_weights_transferred_refined.extract_embeddings(refined_communities)
            vectors_communities_and_weights_transferred_refined = ge.InterpretableWordsModelBuilder(sinr_communities_and_weights_transferred_refined, corpus_refined, n_jobs=40, n_neighbors=4).build()
                
            _log_communities(logger,refined_communities,corpus_refined)
            logger.warning(compute_similarity(vectors_communities_and_weights_transferred_refined,corpus_communities_and_weights_transferred_refined))'''
            # [END] Giving the precomputed communities as a seed to louvain on re-weighted graph 
