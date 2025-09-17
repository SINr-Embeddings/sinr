from sinr.text.cooccurrence import Cooccurrence
from sinr.text.pmi import pmi_filter
from sinr.text.preprocess import extract_text
from os.path import isfile
from networkit import Graph, components, community, setNumberOfThreads, getCurrentNumberOfThreads, getMaxNumberOfThreads, Partition

from multiprocessing import Pool

import networkit
import sinr.text.evaluate as ev
import sinr.graph_embeddings as ge
import sys, pickle

import warnings
warnings.filterwarnings("ignore")

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

def preprocess(path, corpus_name):
    return extract_text(path, min_freq=min_frequences[corpus_name])

def parse_corpus_path(path, extension=".pk"):
    """ Utility function parsing the relative/absolute path given as argument (the large corpus)
        to extract the corpus' name and the path to the file saving the cooccurrence matrix
        The folder text/ contains the result of sinr.extract_text and matrix/ contains 
        the saved cooccurence matrix.

        :param path: Path to the file saving the cooccurrence matrix
    
    """
    from os import makedirs
    
    path_to_corpus = "/".join(path.split("/")[:-2])+"/"
    corpus_name = (path.split("/")[-1]).split(".")[0]

    path_to_matrix = path_to_corpus+"matrix/"
    path_to_text = path_to_corpus+"text/"

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
        vectors.save()
        #if ("BNC" in corpus_name):
        #    vectors.save()
    else: 
        print("loading vectors")
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

def refine(graph, communities, algorithm, gamma=50):
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
        coms = algorithms.louvain(networkx_graph, weight='weight', resolution=gamma, partition=nc)

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

def make_transfert(sinr_large_corpus, small_corpus, small_corpus_name, strategy=1, logger=None):
    def process(strategy):
        result = f""
        # [BEGIN] Computing communities on the small corpus after transferring the weights of the large corpus
        corpus_with_weights_transferred = f"{small_corpus_name}_louvain_weights_transferred_{strategy}: "
        sinr_with_weights_transferred = ge.SINr.load_from_cooc_pkl(small_corpus)
        sinr_with_weights_transferred.ponderate_graph(sinr_large_corpus, strategy)
        weighted_graph = sinr_with_weights_transferred.get_cooc_graph()

        communities_with_weights_transferred = sinr_with_weights_transferred.detect_communities(gamma=50)
        #logger.info(f"{corpus_with_weights_transferred} has {len([i for i in communities_with_weights_transferred.subsetSizes() if i==1])} singleton communities for {communities_with_weights_transferred.numberOfSubsets()} communities")
        #_log_communities(logger,communities_with_weights_transferred,corpus_with_weights_transferred)

        sinr_with_weights_transferred.extract_embeddings()
        vectors_with_weights_transferred = ge.InterpretableWordsModelBuilder(sinr_with_weights_transferred, small_corpus_name, n_jobs=40, n_neighbors=4).build()
        #logger.warning(compute_similarity(vectors_with_weights_transferred,corpus_with_weights_transferred))
        result += compute_similarity(vectors_with_weights_transferred,corpus_with_weights_transferred)+"\n"
        # [END] Computing communities on the small corpus after transferring the weights of the large corpus

        # [BEGIN] Evaluating the similarity for the small corpus using communities learned on the big one
        corpus_communities_transferred = f"{small_corpus_name}_communities_transferred_{strategy}"
        sinr_communities_transferred = ge.SINr.load_from_cooc_pkl(small_corpus)

        sinr_communities_transferred.transfert_communities_labels(vectors_1.get_communities_as_labels_sets())
        transferred_communities = sinr_communities_transferred.get_communities()
        #_log_communities(logger, transferred_communities, corpus_communities_and_weights_transferred)
        
        sinr_communities_transferred.extract_embeddings()
        vectors_communities_transferred = ge.InterpretableWordsModelBuilder(sinr_communities_transferred, corpus_communities_transferred, n_jobs=40, n_neighbors=4).build()
        
        #logger.warning(compute_similarity(vectors_communities_and_weights_transferred, corpus_communities_and_weights_transferred)+"\n")
        result += compute_similarity(vectors_communities_transferred, corpus_communities_transferred)+"\n"
        # [END] Evaluating the similarity for the small corpus using weights and communities learned on the big one

        # [BEGIN] Evaluating the similarity for the small corpus using weights and communities learned on the big one
        corpus_communities_and_weights_transferred = f"{small_corpus_name}_communities_and_weights_transferred_{strategy}"
        sinr_communities_and_weights_transferred = ge.SINr.load_from_cooc_pkl(small_corpus)
        sinr_communities_and_weights_transferred.set_cooc_graph(weighted_graph)

        sinr_communities_and_weights_transferred.transfert_communities_labels(vectors_1.get_communities_as_labels_sets())
        transferred_communities = sinr_communities_and_weights_transferred.get_communities()
        #_log_communities(logger, transferred_communities, corpus_communities_and_weights_transferred)
        
        sinr_communities_and_weights_transferred.extract_embeddings()
        vectors_communities_and_weights_transferred = ge.InterpretableWordsModelBuilder(sinr_communities_and_weights_transferred, corpus_communities_and_weights_transferred, n_jobs=40, n_neighbors=4).build()
        
        #logger.warning(compute_similarity(vectors_communities_and_weights_transferred, corpus_communities_and_weights_transferred)+"\n")
        result += compute_similarity(vectors_communities_and_weights_transferred, corpus_communities_and_weights_transferred)+"\n"
        # [END] Evaluating the similarity for the small corpus using weights and communities learned on the big one
   
        # [BEGIN] Giving the precomputed communities as a seed to louvain on re-weighted graph 
        corpus_communities_and_weights_transferred_refined = f"{small_corpus_name}_communities_and_weights_transferred_refined_{strategy}"
        #_log_communities(logger,refined_communities,corpus_communities_and_weights_transferred_refined)
      
        # Transferring communities from large corpus to small one
        sinr_communities_and_weights_transferred_refined = ge.SINr.load_from_cooc_pkl(small_corpus)
        sinr_communities_and_weights_transferred_refined.transfert_communities_labels(vectors_1.get_communities_as_labels_sets())

        # Using such communities as a seed on the original graph
        refined_communities = refine(sinr_communities_and_weights_transferred_refined.get_cooc_graph(), sinr_communities_and_weights_transferred_refined.get_communities(), "leiden")
        sinr_communities_and_weights_transferred_refined.extract_embeddings(refined_communities)
        vectors_communities_and_weights_transferred_refined = ge.InterpretableWordsModelBuilder(sinr_communities_and_weights_transferred_refined, corpus_communities_and_weights_transferred_refined, n_jobs=40, n_neighbors=4).build()
            
        #logger.warning(compute_similarity(vectors_communities_and_weights_transferred_refined,corpus_communities_and_weights_transferred_refined))
        result += compute_similarity(vectors_communities_and_weights_transferred_refined,corpus_communities_and_weights_transferred_refined)+"\n"
        # [END] Giving the precomputed communities as a seed to louvain on re-weighted graph

        return result

    return process(strategy)

class Transfert:
    def __init__(self, sinr_large_corpus, strategy, logger):
        self.sinr_large_corpus = sinr_large_corpus
        self.strategy = strategy
        self.logger = logger

    def __call__(self, small_corpus_path):
        result = f""
        # Generating corpus name and path to matrix
        corpus_2_name, path_to_text_2, path_to_matrix_2 = parse_corpus_path(small_corpus_path, ".pkl")
        sinr_2, vectors_2 = compute_communities_and_vectors(path_to_matrix_2, corpus_2_name, extension=".pkl", gamma=50)
        
        #logger.warning(compute_similarity(vectors_2, corpus_2_name))
        result += compute_similarity(vectors_2, corpus_2_name)+"\n"

        # Note -- case self.strategy==0 is computing classical SINr vectors
        if (self.strategy>0):
            # Creating the cooccurrence matrix if not existing
            create_cooc_matrix(path_to_text_2, path_to_matrix_2, corpus_2_name)
            result += make_transfert(self.sinr_large_corpus, path_to_matrix_2, corpus_2_name, self.strategy, self.logger)
        return result

if __name__=="__main__":

    import logging, glob
    import calendar, time

    # Current GMT time in a tuple format
    current_GMT = time.gmtime()
    # ts stores timestamp
    ts = calendar.timegm(current_GMT) 

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s:%(message)s', filename="stats.log", encoding='utf-8', level=logging.WARNING, filemode="w")
    
    if (len(sys.argv)!=4):
        print(f"Usage: {sys.argv[0]} path_to_big_corpus path_to_small_corpuses number_of_desired_strategies (should be a list)", file=sys.stderr)
        exit(1)

    def _log_communities(logger, communities, corpus):
        sizes = communities.subsetSizes()
        logger.warning(f"{corpus} has {len(sizes)} communities of maximum size {max(sizes)} and of average size {sum(sizes)/len(sizes)}")

    strategies = range(1,int(sys.argv[3])+1)
    strategies_names = {
            1: "adding existing edges", 
            2: "adding all edges",
            3: "adding non existing edges"
            }

    # TODO -- was it useful to compute this for every small corpus?
    corpus_1 = sys.argv[1]
    # Generating corpus name and path to matrix
    corpus_1_name, path_to_text_1, path_to_matrix_1 = parse_corpuses_path(sys.argv[1],corpus_1, ".pk")
    # Pre-processing OANC
    if (not isfile(path_to_matrix_1)): 
        sentences = preprocess(corpus_1+corpus_1_name+".vrt", corpus_1_name)
        c = Cooccurrence()
        c.fit(sentences, window=5)
        c.matrix = pmi_filter(c.matrix)
        c.save(path_to_matrix_1)

    else:
        create_cooc_matrix(path_to_text_1, path_to_matrix_1, corpus_1_name)
        sinr_1, vectors_1 = compute_communities_and_vectors(path_to_matrix_1, corpus_1_name, extension=".pk", gamma=gamma[corpus_1_name])

        # TODO -- uniformize extensions or have them as arguments
        small_corpuses = glob.glob(sys.argv[2]+"*.pk*")

        try:
            pool = Pool()
            transfert = Transfert(sinr_1, 1, logger)
            results = pool.map(transfert, small_corpuses)
            #print(results, file=sys.stderr)
            print("\n".join(results), file=sys.stderr)
        finally:
            pool.close()
            pool.join()
