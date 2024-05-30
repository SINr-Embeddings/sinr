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
    from os import makedirs
    """ Utility function parsing the relative/absolute path given as argument to 
        extract the corpus' name and the path to the file saving the cooccurrence matrix

        :param path: Path to the file saving the cooccurrence matrix
    
    """
    corpus = path
    path_to_corpus = "/".join(corpus.split("/")[:-2])+"/"
    corpus_name = (corpus.split("/")[-1]).split(".")[0]

    finalpath = path_to_corpus

    path_to_matrix = finalpath+"matrix/"
    path_to_text = finalpath+"text/"

    makedirs(path_to_matrix, exist_ok=True)

    return corpus_name, path_to_text+corpus_name+extension, path_to_matrix+corpus_name+extension

def parse_corpuses_path(path_to_corpuses, corpus, extension=".pk"):
    from os import makedirs
   
    corpus_name = corpus.split("/")[-2]
    path_to_matrix = path_to_corpuses+"matrix/"
    path_to_text = path_to_corpuses+"text/"

    makedirs(path_to_matrix, exist_ok=True)
    
    return corpus_name, path_to_text+corpus_name+extension, path_to_matrix+corpus_name+extension

def create_cooc_matrix(path_to_text,path_to_matrix,corpus_name):
    """ Utility function computing the cooccurrence matrix from the corpus, if not done yet.

    :param matrix: Path to the file containing the cooccurrence matrix
    :param corpus: Path to the file containing the corpus (.VRT)

    """
    if not isfile(path_to_matrix):
        c = Cooccurrence()
        with open(path_to_text, "rb") as f:
            c.fit(pickle.load(f), window=100)
            c.matrix = pmi_filter(c.matrix)
            c.save(path_to_matrix)

# TODO --- split into compute_communities and compute_vectors for better modularity 
# (also change the name of path_to_matrix, maybe not adapted)
def compute_communities_and_vectors(path_to_matrix, corpus_name, extension, gamma, logger=None):
    """Returns SINr and SINrVectors objects created from the cooccurrence matrix given as argument

    :param path_to_matrix: Path to the pickle file containing the cooccurrence matrix
    """
    if not isfile(corpus_name+extension):
        #logger.info(f"Computing communities and embeddings for {corpus_name}")
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix)
        communities = sinr.detect_communities(gamma=gamma)
        sinr.extract_embeddings()
        sinr_vectors = ge.InterpretableWordsModelBuilder(sinr, corpus_name, n_jobs=40, n_neighbors=4).build()
        sinr_vectors.save()
        initial_partition = sinr.get_communities()
        if logger is not None:
            sizes = initial_partition.subsetSizes()
            logger.warning(f"{corpus_name} has {len(sizes)} communities of maximum size {max(sizes)} and of average size {sum(sizes)/len(sizes)}")

    else:
        # TODO -- need to pickle the graph to avoid problems
        if logger is not None:
            logger.debug(f"Loading communities and embeddings of {corpus_name} for the sake of performance: SHOULD BE COMPUTED AT ANY RUN")
        sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix)
        sinr_vectors = ge.SINrVectors(corpus_name)
        sinr_vectors.load()

    return sinr, sinr_vectors

def compute_similarity(sinr_vectors, corpus_name):
    sinr_vectors.sparsify(100)
    similarity = ev.similarity_MEN_WS353_SCWS(sinr_vectors)
    return f"{corpus_name}: {similarity}"

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

if __name__=="__main__":

    import logging, glob
    if (len(sys.argv)!=3):
        logger.error(f"Usage: {sys.argv[0]} path_to_big_corpus path_to_small_corpuses (this is probably not a word but you get the idea)")
        exit(1)

    # TODO -- uniformize extensions or have them as arguments
    for corpus_2 in glob.glob(sys.argv[2]+"text/*.pk*"):

        corpus_1 = sys.argv[1]

        def _log_communities(logger, communities, corpus):
            sizes = communities.subsetSizes()
            logger.warning(f"{corpus} has {len(sizes)} communities of maximum size {max(sizes)} and of average size {sum(sizes)/len(sizes)}")

        logger = logging.getLogger(__name__)
        logging.basicConfig(format='%(levelname)s:%(message)s', filename="stats.log", encoding='utf-8', level=logging.WARNING, filemode="w")

        # Generating corpus name and path to matrix
        corpus_1_name, path_to_text_1, path_to_matrix_1 = parse_corpuses_path(sys.argv[1],corpus_1, ".pkl")
        corpus_2_name, path_to_text_2, path_to_matrix_2 = parse_corpus_path(corpus_2, ".pkl")

        # Creating the cooccurrence matrix if not existing
        logger.info(f"Loading cooccurrence matrices for {corpus_1_name} and {corpus_2_name}")
        create_cooc_matrix(path_to_text_1, path_to_matrix_1, corpus_1_name)
        create_cooc_matrix(path_to_text_2, path_to_matrix_2, corpus_2_name)

        # Saving the embeddings for the sake of performance
        sinr_1, sinr_vectors_1 = compute_communities_and_vectors(path_to_matrix_1, corpus_1_name, ".pk", gamma[corpus_1_name])
        sinr_2, sinr_vectors_2 = compute_communities_and_vectors(path_to_matrix_2, corpus_2_name, ".pk", 50)
        
        _log_communities(logger, sinr_2.get_communities(), corpus_2_name)
        logger.warning(compute_similarity(sinr_vectors_2, corpus_2_name))

        # Evaluating the similarity for the small corpus using communities learned on the big one
        corpus_transferred = corpus_2_name+"_transferred"
        sinr_transferred = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
        sinr_transferred.ponderate_graph(sinr_1)

        corpus_ponderated = corpus_2_name+"_ponderated"
        ponderated_communities = sinr_transferred.detect_communities(gamma=50)
        sinr_transferred.extract_embeddings()
        sinr_vectors_ponderated = ge.InterpretableWordsModelBuilder(sinr_transferred, corpus_2_name, n_jobs=40, n_neighbors=4).build()

        ponderated_communities = sinr_transferred.get_communities()
        logger.info(f"{corpus_ponderated} has {len([i for i in ponderated_communities.subsetSizes() if i==1])} singleton communities for {ponderated_communities.numberOfSubsets()} communities")
       
        _log_communities(logger,ponderated_communities,corpus_ponderated)
        #logger.warning(f"Evaluating similarity for {corpus_refined}")
        logger.warning(compute_similarity(sinr_vectors_ponderated,corpus_ponderated))

        sinr_transferred.transfert_communities_labels(sinr_vectors_1.get_communities_as_labels_sets())
        sinr_transferred.extract_embeddings()
        sinr_vectors_transferred = ge.InterpretableWordsModelBuilder(sinr_transferred, corpus_transferred, n_jobs=40, n_neighbors=4).build()
        transferred_communities = sinr_transferred.get_communities()
        
        _log_communities(logger, transferred_communities, corpus_transferred)
        logger.warning(compute_similarity(sinr_vectors_transferred, corpus_transferred))
        
        # Giving the precomputed communities as a seed to label propagation to see if this helps
        corpus_refined = corpus_2_name+"_transferred_refined"
        initial_partition = sinr_transferred.get_communities()
        refined_communities = refine(sinr_transferred.get_cooc_graph(), initial_partition, "louvain")
        logger.info(f"{corpus_refined} has {len([i for i in refined_communities.subsetSizes() if i==1])} singleton communities for {refined_communities.numberOfSubsets()} communities")
       
        sinr_refined = ge.SINr.load_from_cooc_pkl(path_to_matrix_2)
        sinr_refined.extract_embeddings(refined_communities)
        sinr_vectors_refined = ge.InterpretableWordsModelBuilder(sinr_refined, corpus_refined, n_jobs=40, n_neighbors=4).build()
            
        _log_communities(logger,refined_communities,corpus_refined)
        #logger.warning(f"Evaluating similarity for {corpus_refined}")
        logger.warning(compute_similarity(sinr_vectors_refined,corpus_refined))
