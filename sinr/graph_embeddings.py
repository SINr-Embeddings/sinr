import pickle as pk

from networkit import Graph, components, community, setNumberOfThreads, getCurrentNumberOfThreads, getMaxNumberOfThreads, Partition
import networkit.graphtools as nkg
from numpy import argpartition, argsort, asarray, where, nonzero, concatenate, repeat, mean, nanmax, int64, shape, delete, nanmean
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
from scipy.sparse import csr_matrix
import sklearn.preprocessing as skp
from tqdm.auto import tqdm
from random import randint
from functools import partialmethod
from math import ceil
import time
import os
from itertools import combinations

from . import strategy_loader
from .logger import logger
from .nfm import get_nfm_embeddings
import sinr.text.evaluate as ev


class SINr(object):
    """Object that can be used to extract word or graph embeddings using the SINr approach.
    This object cannot then be used to inspect the resulting vectors. Instead, using the ModelBuilder class, a SINrVectors object should be created that will allow to use the resulting vectors.
    
    ...


    Attributes
    ----------
    Attributes should not be read
    """

    @classmethod
    def load_from_cooc_pkl(cls, cooc_matrix_path, n_jobs=-1):
        """Build a sinr object from a co-occurrence matrix stored as a pickle : useful to deal with textual data.
        Co-occurrence matrices should for instance be generated using sinr.text.cooccurrence

        :param cooc_matrix_path: Path to the cooccurrence matrix generated using sinr.text.cooccurrence : the file should be a pickle
        :type cooc_matrix_path: string
        :param n_jobs: Number of jobs that should be used The default is -1.
        :type n_jobs: int, optional

        
        """
        logger.info("Building Graph.")

        word_to_idx, matrix = strategy_loader.load_pkl_text(cooc_matrix_path)
        graph = get_graph_from_matrix(matrix)
        graph, word_to_idx = get_compact_lgcc(graph, word_to_idx)
        out_of_LgCC = get_lgcc(graph)
        logger.info("Finished building graph.")
        return cls(graph, out_of_LgCC, word_to_idx)

    @classmethod
    def load_from_adjacency_matrix(cls, matrix_object, labels=None, n_jobs=-1):
        """Build a sinr object from an adjacency matrix as a sparse one (csr)

        :param matrix_object: Matrix describing the graph.
        :type matrix_object: csr_matrix
        :param labels:  (Default value = None)
        :param n_jobs: Number of jobs that should be used The default is -1.
        :type n_jobs: int, optional

        
        """
        logger.info("Building Graph.")
        word_to_idx, matrix = strategy_loader.load_adj_mat(matrix_object, labels)
        graph = get_graph_from_matrix(matrix)
        graph, word_to_idx = get_compact_lgcc(graph, word_to_idx)
        out_of_LgCC = get_lgcc(graph)
        logger.info("Finished building graph.")
        return cls(graph, out_of_LgCC, word_to_idx)

    @classmethod
    def load_from_graph(cls, graph, n_jobs=-1):
        """Build a sinr object from a networkit graph object

        :param graph: Networkit graph object.
        :type graph: networkit
        :param n_jobs: Number of jobs that should be used The default is -1.
        :type n_jobs: int, optional

        
        """
        word_to_idx = dict()
        idx = 0
        for u in graph.iterNodes():
            word_to_idx[u] = idx
            idx += 1
        graph, word_to_idx = get_compact_lgcc(graph, word_to_idx)
        out_of_LgCC = get_lgcc(graph)
        logger.info("Finished building graph.")
        return cls(graph, out_of_LgCC, word_to_idx)

    def run(self, algo=None):
        """Runs the training of the embedding, i.e. community detection + vectors extraction

        :param algo: Community detection algorithm. The default, None allorws to run a Louvain algorithm
        :type algo: networkit.algo.community, optional

        
        """
        self.communities = self.detect_communities(self.cooc_graph, algo=algo)
        self.extract_embeddings(communities=self.communities)

    def detect_communities(self, gamma=1, algo=None, inspect=True, par="balanced"):
        """Runs community detection on the graph

        :param gamma: For Louvain algorithm which is the default algorithm (ignore this parameter if param algo is used), allows to control the size of the communities. The greater it is, the smaller the communities. The default is 1.
        :type gamma: int, optional
        :param algo: Community detection algorithm. The default, None allorws to run a Louvain algorithm
        :type algo: networkit.algo.community, optional
        :param inspect: Whether or not one wants to get insight about the communities extracted. The default is True.
        :type inspect: boolean, optional
        :param par: Parallelisation strategy for networkit community detection (Louvain), see https://networkit.github.io/dev-docs/python_api/community.html#networkit.community.PLM for more details, "none randomized" allows randomness in Louvain in single thread mode. To force determinism pass the "none" parallelisation strategy. The default is balanced.

        
        """
        logger.info("Detecting communities.")
        print(f"Gamma for louvain : {gamma}")
        if getMaxNumberOfThreads() == 1 and par!="none randomized":
            logger.warning(f"""The current number of threads is set to {getMaxNumberOfThreads()} with parallelization strategy {par}. Nodes will not be randomized in Louvain. Consider using more threads by resetting the setNumberOfThreads parameter of networkit or use the 'none randomized' parallelization strategy.""")
        if algo is None:
            algo = community.PLM(self.cooc_graph, refine=False, gamma=gamma, turbo=True, recurse=False, par=par)
        communities = community.detectCommunities(self.cooc_graph, algo=algo, inspect=inspect)
        communities.compact(useTurbo=True)  # Consecutive communities from 0 to number of communities - 1
        self.communities = communities
        logger.info("Finished detecting communities.")
        return communities

    def size_of_voc(self):
        """Returns the size of the vocabulary."""
        return len(self.idx_to_wrd)

    def transfert_communities_labels(self, community_labels, refine=False):
        """Transfer communities computed on one graph to another, used mainly with co-occurence graphs.

        :param community_labels: a list of communities described by sets of labels describing the nodes
        :typev community_labels: list[set[str]]
        :param refine:  (Default value = False)
        :type refine: bool
        :returns: Initializes a partition where nodes are all singletons. Then, when communities in parameters contain labels
        that are in the graph at hand, these communities are transferred.

        """
        self.communities = Partition(self.size_of_voc())
        self.communities.allToSingletons()
        for com in community_labels:
            new_com = []
            # Check if labels in communities passed as parameters are in the graph at hand, if so -> transfer the community
            for word in com:
                if word in self.wrd_to_idx:
                    new_com.append(self.wrd_to_idx[word])
            # Transferring the community to the graph at hand
            if len(new_com) > 1:
                subset_id = self.communities.subsetOf(new_com[0])
                for idx in range(1, len(new_com)):
                    self.communities.moveToSubset(subset_id, new_com[idx])
        # Compating the community ids
        self.communities.compact()
        if refine:
            self._refine_transfered_communities()


    def extract_embeddings(self, communities=None):
        """Extract the embeddings based on the graph and the partition in communities previously detected.

        :param communities: Community structures (Default value = None)
        :type communities: networkit.Partition

        
        """
        logger.info("Extracting embeddings.")

        if communities is not None:
            self.communities = communities

        logger.info("Applying NFM.")
        np, nr, nfm = get_nfm_embeddings(self.cooc_graph, self.communities.getVector(), self.n_jobs)
        self.np = np
        self.nr = nr
        self.nfm = nfm
        logger.info("NFM successfully applied.")
        logger.info("Finished extracting embeddings.")

    def __init__(self, graph, lgcc, wrd_to_idx, n_jobs=-1):
        """Should not be used ! Some factory methods below starting with "load_" should be used instead.

        Parameters
        ----------
        graph : networkit graph
        lgcc : networkit graph
            the largest connected component of graph
        wrd_to_idx : dict
            A matching between a vocabulary and ids. Useful for text. Otherwise, the vocabulary and the ids are the same
        n_jobs : int, optional
            Number of jobs that should be runned. The default is -1, all available threads

        Returns
        -------
        None.

        """
        self.nfm = None
        self.nr = None
        self.np = None
        self.communities = None
        self.n_jobs = n_jobs
        self.wrd_to_idx = wrd_to_idx
        self.idx_to_wrd = _flip_keys_values(self.wrd_to_idx)
        self.cooc_graph = graph
        self.out_of_LgCC = lgcc

        assert type(n_jobs) == int, "n_jobs must be of type int"
        assert n_jobs == -1 or n_jobs>0, "Value for n_jobs must be -1 or greater than 0"

        if n_jobs > 0:
            setNumberOfThreads(n_jobs)

        self.wd_before, self.wd_after = None, None

    def get_out_of_LgCC_coms(self, communities):
        """Get communities that are not in the Largest Connected Component (LgCC).

        :param communities: Partition object of the communities as obtained by calling a Networkit community detection algorithm
        :type communities: Partition
        :returns: Indices of the comunnities outside the LgCC
        :rtype: list[int]

        """
        set_out_of_LgCC = set(self.out_of_LgCC)
        out_of_LgCC_coms = []
        for com in communities.getSubsetIds():
            if set(communities.getMembers(com)) & set_out_of_LgCC != {}:
                out_of_LgCC_coms.append(com)
        return out_of_LgCC_coms

    def get_cooc_graph(self):
        """Return the graph. """
        return self.cooc_graph

    def get_nr(self):
        """Return the NR matrix. """
        if self.nr is None:
            raise NoEmbeddingExtractedException
        else:
            return self.nr

    def get_np(self):
        """Return the NP matrix. """
        if self.np is None:
            raise NoEmbeddingExtractedException
        else:
            return self.np

    def get_nfm(self):
        """Return the NFM matrix. """
        if self.nfm is None:
            raise NoEmbeddingExtractedException
        else:
            return self.nfm

    def get_vocabulary(self):
        """Return the vocabulary. """
        return list(self.idx_to_wrd.values())

    def get_wrd_to_id(self):
        """Return the word to index map. """
        return self.wrd_to_idx

    def get_communities(self):
        """Return the `networkit.Patrtion`community object. """
        if self.communities is None:
            raise NoCommunityDetectedException
        else:
            return self.communities


def _flip_keys_values(dictionary):
    """Flip keys and values in a dictionnary.

    :param dictionary: The dictionnary to invert

    """
    return dict((v, k) for k, v in dictionary.items())


def get_lgcc(graph):
    """Return the nodes that are outside the Largest Connected Component (LgCC) of the graph.

    :param graph: The graph for which to retrieve out of LgCC nodes
    :type graph: networkit graph

    
    """
    out_of_LgCC = set(graph.iterNodes()) - set(components.ConnectedComponents.extractLargestConnectedComponent(
        graph).iterNodes())  # Extract out of largest connected component vocabulary
    return out_of_LgCC


def get_graph_from_matrix(matrix):
    """Build a graph from a sparse adjacency matrix.

    :param matrix: A sparse matrix describing a graph
    :type matrix: scipy.sparse.coo_matrix

    """
    graph = Graph(weighted=True)
    rows, cols = matrix.row, matrix.col
    weights = matrix.data
    for row, col, weight in zip(rows, cols, weights):
        graph.addEdge(u=row, v=col, w=weight, addMissing=True)
    return graph

def get_compact_lgcc(graph, word_to_idx):
    """Get a compacted graph with only nodes inside the largest connected component. Get the words with ids corresponding to the new node ids.
    
    :param graph: The input graph
    :type graph: networkit graph
    :param word_to_idx: The words mapped to their initial ids
    :type word_to_idx: dictionnary
    
    :returns: The new graph and dictionnary of words
    :rtype: networkit graph, dictionnary
    
    """
    
    # search isolated nodes
    isolated_nodes = list()
    for u in graph.iterNodes():
        if graph.degree(u) == 0:
            isolated_nodes.append(u)
            
    if len(isolated_nodes) != 0:
        # remove nodes and corresponding words from the graph and dict of words
        idx_to_word = _flip_keys_values(word_to_idx)
        for u in isolated_nodes:
            graph.removeNode(u)
            del idx_to_word[u]
        word_to_idx = _flip_keys_values(idx_to_word)
        # change nodes ids to continuous ids
        idx_map = nkg.getContinuousNodeIds(graph)
        graph = nkg.getCompactedGraph(graph, idx_map)
        # change words ids to continuous ids
        word_to_idx = {k: idx_map[v] for k, v in word_to_idx.items()}
        
    return graph, word_to_idx


class NoCommunityDetectedException(Exception):
    """Exception raised when no community detection has been performed thus leaving `self.communities` to its default value `None`. """
    pass


class NoEmbeddingExtractedException(Exception):
    """Exception raised when no embedding extraction has been performed thus leaving `self.nr` and `self.np`and `self.nfm` to their default value `None`. """
    pass


class ModelBuilder:
    """Object that should be used after the training of word or graph embeddings using the `SINr` object.
    The `ModelBuilder` will make use of the `SINr` object to build a `SINrVectors` object that will allow to use the resulting vectors efficiently.
    ..


    Attributes
    ----------
    Attributes should not be read
    """

    def __init__(self, sinr, name, n_jobs=-1, n_neighbors=31):
        """
        Creating a ModelBuilder object to build a `SINrVectors`.

        Parameters
        ----------
        sinr : SINr
            A SINr object with extracted vectors
        name : string
            Name of the model
        n_jobs : int, optional
            DESCRIPTION. The default is -1, all the available threads.
        n_neighbors : int, optional
            Number of neighbors to use for similarity. The default is 31.

        """
        self.sinr = sinr
        self.model = SINrVectors(name, n_jobs, n_neighbors)

    def with_embeddings_nr(self, threshold=0):
        """Adding Node Recall vectors to the `SINrVectors` object.

        :param threshold:  (Default value = 0)
        :type threshold: float

        """
        if threshold == 0:
            self.model.set_vectors(self.sinr.get_nr())
        else:
            coo = self.sinr.get_nr().tocoo()
            for idx, val in enumerate(coo.data):
                if val < threshold:
                    coo.data[idx] = 0
            coo.eliminate_zeros()
            self.model.set_vectors(coo.tocsr())
        return self

    def with_embeddings_nfm(self):
        """Adding NFM (Node Recall + Node Predominance) vectors to the `SINrVectors` object. """
        self.model.set_vectors(self.sinr.get_nfm())
        return self

    def with_np(self):
        """Storing Node predominance values in order to label dimensions for instance. """
        self.model.set_np(self.sinr.get_np())
        return self

    def with_vocabulary(self):
        """To deal with word vectors or graph when nodes have labels."""
        self.model.set_vocabulary(self.sinr.get_vocabulary())
        return self

    def with_communities(self):
        """To keep the interpretability of the model using the communities."""
        self.model.set_communities(self.sinr.get_communities())
        return self

    def with_graph(self):
        """To keep the underlying graph ; useful to get co-occ statistics, degree of nodes or to label communities with central nodes."""
        self.model.set_graph(self.sinr.get_cooc_graph())
        return self

    def with_all(self):
        """ """
        return self.with_embeddings_nr().with_vocabulary().with_communities().with_graph().with_np()

    def build(self):
        """To get the `SINrVectors` object"""
        return self.model
    
class OnlyGraphModelBuilder(ModelBuilder):
    """Object that should be used after training word or graph embeddings using the SINr object to get interpretable vectors.
    The OnlyGraphModelBuilder will make use of the `SINr` object to build a `SINrVectors` object that will allow to use the resulting vectors efficiently.
    No need to use parent methods starting by "with", those are included in the "build" function.
    Just provide the name of the model and build it.

    """
    def build(self):
        """Build `OnlyGraphModelBuilder` which contains solely the embeddings. """
        self.with_np()
        return self.model

class InterpretableWordsModelBuilder(ModelBuilder):
    """Object that should be used after training word or graph embeddings using the `SINr` object to get interpretable word vectors.
    The `InterpretableWordsModelBuilder` will make use of the `SINr` object to build a `SINrVectors` object that will allow to use the resulting vectors efficiently.
    No need to use parent methods starting by "with", those are included in the `build` function.
    Just provide the name of the model and build it.

    """

    def build(self):
        """Build `InterpretableWordsModelBuilder` which contains the vocabulary, the embeddings and the communities. """
        self.with_embeddings_nr().with_vocabulary().with_communities()
        return self.model


class ThresholdedModelBuilder(ModelBuilder):
    """Object that should be used after the training of word or graph embeddings using the SINr object to get interpretable word vectors.
    The `ThresholdedModelBuilder` will make use of the `SINr` object to build a `SINrVectors` object that will allow to use the resulting vectors efficiently.
    Values in the vectors that are lower than the threshold will be discarded. Vectors are then sparser and more interpretable.
    No need to use parent methods starting by "with", those are included in the `build` function.
    Just provide the name of the model and build it.


    """

    def build(self, threshold=0.01):
        """Build `ThresholdedModelBuilder` which contains the vocabulary, the embeddings with values thresholded above a minimum and the communities. 

        :param threshold:  (Default value = 0.01)

        """
        self.with_embeddings_nr(threshold=threshold).with_vocabulary().with_communities()
        return self.model


class NoInterpretabilityException(Exception):
    """Raised when the communities were not included in the model that was built. It is thus not interpretable anymore."""
    pass


class NoVocabularyException(Exception):
    """Raised when no vocabulary was included in the model that was built. One cannot play with words."""
    pass


class InterpretableDimension:
    """ Internal class : should be used to encapsulate data about a dimension instead of using a simple dict.
    """

    def __init__(self, idx, type):
        """_summary_

        :param idx: identifier of the dimension
        :type idx: int
        :param type: whether one uses stereotypes or descriptors
        :type type: str
        """
        self.idx = idx
        self.type = type
        self.value = None
        self.interpreters = []

    def add_interpreter(self, obj, value):
        """Adding an element that would help to interpret the meaning of the dimension

        :param obj: a descriptor or a stereotype of the dimension
        :type obj: str
        :param value: a value describing the relevance of the descriptor for this dimension
        :type value: float
        """
        self.interpreters.append((round(value, 2), obj))

    def get_idx(self):
        """Getter of the idx attribute

        :return: the id of the dimension
        :rtype: int
        """
        return self.idx

    def get_value(self):
        """Getter for the value parameter, which is a boolean to detect if numerical values are used in the interpreters or not

        :return: the value attribute
        :rtype: bool
        """
        return self.value

    def get_interpreters(self):
        """Getting the list of interpreters, object that allows to describe the dimension

        :return: the list of interpreters
        :rtype: list
        """
        return self.interpreters

    def get_interpreter(self, id):
        """Get a specific interpreter

        :param id: id of the interpreter
        :type id: int
        :return: the interpreter of id for this dimension
        :rtype: an interpreter as a tuple (obj: str, value: float) if there is a value
        """
        return self.interpreters[id]

    def sort(self, on_value=True):
        """Sorting the interpreters, according to values if values is True, according to the str described of the interpreters instead if False

        :param on_value: sorting on values or not, defaults to True
        :type on_value: bool, optional
        """

        if on_value:
            self.interpreters.sort(key=lambda x: x[0], reverse=True)
        else:
            self.interpreters.sort(key=lambda x: x[1], reverse=True)

    def topk(self, topk):
        """ Selecting only the topk interpreters

        :param topk: number of interpreters to keep
        :type topk: int
        """
        topk = min(topk, len(self.interpreters))
        self.interpreters = self.interpreters[:topk]

    def with_value(self):
        """Seeting the value to True

        :return: the self object
        :rtype: InterpretableDimension
        """
        self.value = True
        return self

    def get_dict(self):
        """The dict that can be processed with the interpreters

        :return: a dict of interpreters for the dimension
        :rtype: dict
        """
        result = {"dimension": self.idx, "value": self.value,
                  self.type: self.interpreters} if self.value is not None else {"dimension": self.idx,
                                                                                self.type: self.interpreters}
        return result

    def __repr__(self):
        return str(self.get_dict())


class NoIntruderPickableException(Exception):
        """Raised when no intruder could be found with the percentages provided"""
        pass

class SINrVectors(object):
    """After training word or graph embeddings using SINr object, use the ModelBuilder object to build `SINrVectors`.
    `SINrVectors` is the object to manipulate the model, explore the embedding space and its interpretability

    """
    labels: bool

    def __init__(self, name, n_jobs=-1, n_neighbors=20):
        """
        Initializing `SINr` vectors objets
        :param name: name of the model
        :type name: str
        :param n_jobs: number of jobs to use (k-nearest neighbors to obtain most similar words or nodes), defaluts to -1
        :type n_jobs: int
        :param n_neighbors: number of neighbors to consider when querying the most similar words or nodes
        :type n_neighbors: int
        """
        self.G = None
        self.communities_sets = None
        self.community_membership = None
        self.np = None
        self.neighbors = None
        self.vocab = None
        self.vectors = None
        self.name = name
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.labels = False
        
    @classmethod
    def load_from_w2v(cls, w2v_path, name, n_jobs=-1, n_neighbors=20):
        """
        Initializing a SINrVectors object using a file at the word2vec format
        :param w2v_path: path of the file at word2vec format which contains vectors
        :type w2v_path: str
        :param name: name of the model, useful to save it
        :type name: str
        """
        file = open(w2v_path)
        i = 0
        vocabulary= []
        vectors = []
        for line in file:
            line = line.strip()
            line = line.split(" ")
            word = line[0].strip()
            vector = [[i,col, float(x)] for col, x in enumerate(line[1:]) if x != 0]
            vectors.extend(vector)
            vocabulary.append(word)
            i+=1
        file.close()
        
        model = cls(name, n_jobs=n_jobs, n_neighbors=n_neighbors)
        model.set_vocabulary(vocabulary)
        rows, cols, vals = zip(*vectors)
        matrix = csr_matrix((vals, (rows, cols)))
        model.set_vectors(matrix)
        
        return model

    def get_communities_as_labels_sets(self):
        """Get partition of communities as a list of sets each containing the label associates to the node in the community.


        :returns: List of communities each represented by a set of labels associated to the node in each subset

        :rtype: list[set[str]]
        :raises NoInterpretabilityException: `SINrVectors` was not exported with interpretable dimensions

        """
        if self.communities_sets is None:
            raise NoInterpretabilityException
        labels_sets = []
        for com in self.communities_sets:
            labels = set()
            for u in com:
                labels.add(self.vocab[u])
            labels_sets.append(labels)
        return labels_sets
    
    def get_matching_communities(self, sinr_vector):
        """Get the matching between two partitions with common vocabularies
        
        :param sinr_vector: Small model (target)
        :type sinr_vector: SINrVectors
        
        :returns: Lists. The first indicating, at each of its index corresponding to the community's index of the self object (src), its matching number in the parameter sinr_vector's communities (tgt) if it exists. The second indicating, at each of its index corresponding to the community's index of the object in parameter, its matching number in the self object.
        :rtype: (list[int],list[int])
        """
        
        src_communities = self.get_communities_as_labels_sets()
        l = [-1] * len(src_communities)
        tgt_communities = sinr_vector.get_communities_as_labels_sets()
        for id_src, lab_set_src in enumerate(src_communities):
            for id_tgt, lab_set_tgt in enumerate(tgt_communities):
                if len(lab_set_src.intersection(lab_set_tgt)) > 0:
                    l[id_src] = id_tgt
        tgt_from_src = [-1] * len(tgt_communities)
        for idx, val in enumerate(l):
            tgt_from_src[val] = idx
        
        return l, tgt_from_src
    
    def get_vectors_using_self_space(self, sinr_vector):
        """Transpose the vectors of the sinr_vector object in parameter in the embedding space of the self object, using matching communities
        
        :param sinr_vector: Small model (target)
        :type sinr_vector: SINrVectors
        
        :returns: Copy of the self model (the big one) with vectors of the parameter (small one) transposed to its referential
        :rtype: SINrVectors
        """
        from scipy.sparse import coo_matrix
        
        matching_st, matching_ts = self.get_matching_communities(sinr_vector)
        
        vectors = sinr_vector.vectors.tocoo()
        row = vectors.row
        data = vectors.data
        col = [matching_ts[val] for val in vectors.col]
    
        matrix = coo_matrix((data, (row, col)), shape=(sinr_vector.vectors.shape[0], self.vectors.shape[1]))
        
        import copy
        self_copy = copy.deepcopy(self)
        self_copy.set_vectors(matrix.tocsr())
        self_copy.vocab = sinr_vector.vocab
        return self_copy

    def set_n_jobs(self, n_jobs):
        """Set the number of jobs.

        :param n_jobs: number of jobs

        """
        self.n_jobs = n_jobs

    def set_graph(self, G):
        """Set the graph property.

        :param G: A networkit graph
        :type G: networkit.Graph

        """
        self.G = G

    def get_nnz(self):
        """Get the count of non-zero values in the embedding matrix.


        :returns: number of non zero values

        """
        return self.vectors.getnnz()
    
    def get_nz_dims(self, obj) :
        """Get the indices of non-zero dimensions.

        :param obj: An int or string for which to get non-zero dimensions
        :returns: set of indices of non zero dimensions

        """
        index = self._get_index(obj)
        vector = self._get_vector(index, row=True)
        return set(list(nonzero(vector)[0]))
    
    def get_value_dim_per_word(self, obj, dim_index):
        """Get the value of a dimension for a word.

        :param obj: a word or its index
        :type obj: str or int
        :param dim_index: the index of the dimension to retrieve
        :type dim_index: int
        :returns: the value for a given vector on a given dimension

        """
        index = self._get_index(obj)
        vector = self._get_vector(index, row=True)
        dict_in_dim = InterpretableDimension(dim_index,"descriptors").with_value().get_dict()
        value = dict_in_dim["value"]
        return value
     
    def get_nnv(self):
        """Get the number of null-vetors in the embedding matrix.


        :returns: number of null vectors

        """
        sum = self.vectors.sum(axis=1)
        nulls = where(sum == 0)[0]
        return len(nulls)

    def pct_nnz(self):
        """Get the percentage of non-zero values in the embedding matrix.


        :returns: percentage of non-zero values in the embedding matrix

        """
        nnz = self.get_nnz()
        return (nnz * 100) / (self.vectors.shape[0] * self.vectors.shape[1])

    def set_vocabulary(self, voc):
        """Set the vocabulary for word-co-occurrence graphs.

        :param voc: set the vocabulary when dealing with words or nodes with labels. label parameter is set to True.
        By default, labels from the vocab will be used.

        """
        self.vocab = voc
        # self.wrd_to_id = wrd_to_id
        self.labels = True

    def set_vectors(self, embeddings):
        """Set the embedding vectors and initialize nearest neighbors.

        :param embeddings: initialize the vectors and build the nearest neighbors data structure using sklearn
        :type embeddings: scipy.sparse.csr_matrix

        """
        self.vectors = embeddings
        self.neighbors = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine', n_jobs=self.n_jobs).fit(
            self.vectors)

    def set_np(self, np):
        """Set the embedding matrix.

        :param np: a sparse matrix of the embeddings
        :type np: scipy.sparse.csr_matrix

        """
        self.np = np

    def set_communities(self, com):
        """Set the communities from the partition in communities.

        :param com: partition in communities
        :type com: networkit.Partition

        """
        self.community_membership = com.getVector()
        nb_coms = max(self.community_membership)
        self.communities_sets = []
        for i in range(nb_coms + 1):
            self.communities_sets.append(set())
        for idx, com in enumerate(self.community_membership):
            self.communities_sets[com].add(idx)

    def sparsify(self, k):

        """Sparsify the vectors keeping activated the top k dimensions
        
        :param k: int
        
        """
    
        data = list()
        rows = list()
        cols = list()
    
        m_shape = self.vectors.get_shape()
    
        if not self.vectors.has_sorted_indices:
            self.vectors.sort_indices()
    
        for i_row in tqdm(range(m_shape[0]), desc="vectors to sparsify"):  
    
            if(i_row == 0):
                # datas/indices of the first line
                data_row = self.vectors.data[ 0 : self.vectors.indptr[1] - self.vectors.indptr[0]]
                indices_row = self.vectors.indices[ 0 : self.vectors.indptr[1] - self.vectors.indptr[0]]
            else :
                # datas/indices of a line : data/indices[indptr[line] - indptr[0] : indptr[next_line] - indptr[0]]
                data_row = self.vectors.data[self.vectors.indptr[i_row] - self.vectors.indptr[0] : 
                                            self.vectors.indptr[i_row + 1] - self.vectors.indptr[0]]
                indices_row = self.vectors.indices[self.vectors.indptr[i_row] - self.vectors.indptr[0] : 
                                                    self.vectors.indptr[i_row + 1] - self.vectors.indptr[0]]
                
            # datas are sorted if the number of activated dimensions is more than k
            # we save the top k dimensions 
            # if there is less than k datas for the line, we keep all the datas
            if(k < len(data_row)):
                ind_sort = argsort(data_row)
                
                data = concatenate((data_row[ind_sort[len(ind_sort)-k :]], data))
                rows = concatenate((repeat(i_row,k), rows))
                cols = concatenate((indices_row[ind_sort[len(ind_sort)-k :]],cols))
    
            else:
                data = concatenate((data_row, data))
                rows = concatenate((repeat(i_row,len(data_row)), rows))
                cols = concatenate((indices_row,cols))
                
        new_vec = csr_matrix((data, (rows, cols)), shape=m_shape)
        
        self.set_vectors(new_vec)

    def binarize(self):

        """Binarize the vectors
        
        """
        
        self.set_vectors(skp.binarize(self.vectors))
        
    def dim_nnz_count(self, dim):
        """ Count the number of non zero values in a dimension.
        :param dim: index of the dimension
        :type dim: int
        
        :return: the number of non zero values in the dimension
        :rtype: int
        """
        
        d = self.vectors.getcol(dim)
        return d.nnz
    
    def obj_nnz_count(self, obj):
        """ Count the number of non zero values in a word vector.
        :param obj: word
        :type obj: string
        
        :return: the number of non zero values in the word vector
        :rtype: int
        """
        
        vec = self.vectors.getrow(self.vocab.index(obj))
        return vec.nnz
        
    def remove_communities_dim_nnz(self, threshold_min = None, threshold_max = None):
        """Remove dimensions (communities) which are the less activated and those which are the most activated.
        
        :param threshold_min: minimal number of non zero values to have for a dimension to be kept
        :type threshold_min: int
        :param threshold_max: maximal number of non zero values to have for a dimension to be kept
        :type threshold_max: int
        
        """
        
        if threshold_min != None or threshold_max != None:
            dims = self.get_number_of_dimensions()
            
            indexes = list()

            for dim in tqdm(range(dims)):
                if threshold_min != None:
                    if self.dim_nnz_count(dim) < threshold_min:
                        indexes.append(dim)
                if threshold_max != None:
                    if self.dim_nnz_count(dim) > threshold_max:
                        indexes.append(dim)

            # Remove dimensions from the matrix of embeddings
            self.set_vectors(csr_matrix(delete(self.vectors.toarray(), indexes, axis=1)))

            # Remove communities from communities sets
            self.communities_sets = delete(self.communities_sets, indexes, axis=0)

            # Update of the community membership for each word of the vocabulary
            self.community_membership = list()
            for w in range(shape(self.vocab)[0]):
                found = 0
                #print(type(self.communities_sets))
                for i, com in enumerate(self.communities_sets):
                    #print(com)
                    if w in com:
                        self.community_membership.append(i)
                        found = 1
                if not found:
                    self.community_membership.append(-1)
    
    def dim_nnz_thresholds(self, step = 100, diff_tol = 0.005):
        """Give the minimal and the maximal number of non zero values to have for a dimension to be kept and not lower the model's similarity. Taking into account the datasets MEN, WS353, SCWS and SimLex-999.
        
        :param step: step to search thresholds (default value : 100)
        :param: diff_tol: difference of similarity tolerated with the low threshold (default value : 0.005)
        
        :return: thresholds (low, high)
        :rtype: tuple of int
        
        """
        
        # Disable tqdm to clear output
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        
        # Copy of the sinrvectors to remove dimensions
        name = self.name
        name_tmp = 'vec_ref_' + str(round(time.time()*1000))
        self.name = name_tmp
        self.save(name_tmp + '.pk')
        self.name = name
        
        # Maximum of non zero values in dimensions
        nnz_count = list()

        for d in range(self.get_number_of_dimensions()):
            nnz_count.append(self.dim_nnz_count(d))

        max_nnz = max(nnz_count)
        min_nnz = min(nnz_count)
        
        print(f'Minimum of non zero values in dimensions : {min_nnz}')
        print(f'Maximum of non zero values in dimensions : {max_nnz}')

        # Mean similarity of vectors with all dimensions (MEN, WS353, SCWS, SimLex-999)
        simlex999 = ev.fetch_SimLex(which='999')
        men = ev.fetch_data_MEN()
        ws353 = ev.fetch_data_WS353()
        scws = ev.fetch_data_SCWS()

        sim_all_dim = list()
        sim_all_dim.append(ev.eval_similarity(self, simlex999, print_missing=False))
        sim_all_dim.append(ev.eval_similarity(self, men, print_missing=False))
        sim_all_dim.append(ev.eval_similarity(self, ws353, print_missing=False))
        sim_all_dim.append(ev.eval_similarity(self, scws, print_missing=False))

        sim_mean_all_dim = mean(sim_all_dim)
        
        print(f'Mean similarity of the model with all dimensions (MEN, WS353, SCWS, SimLex-999) : {sim_mean_all_dim}\n')

        # Low threshold for the number of nnz per dimension
        # Taking the maximal threshold (multiple of step) 
        # for which the similarity is greater than the similarity - 0.01 of the vectors with all dimensions

        sim = sim_mean_all_dim
        seuil = 0

        while(sim > sim_mean_all_dim - diff_tol and seuil + step <= max_nnz and sim != float('nan')):
            seuil += step

            vec_seuil = SINrVectors(name_tmp)
            vec_seuil.load(name_tmp + '.pk')
            vec_seuil.remove_communities_dim_nnz(threshold_min = seuil)
            
            # if a word is no longer represented by the dimensions, we stop the filtering
            word_nnz_count = list()
            for word in vec_seuil.vocab:
                word_nnz_count.append(vec_seuil.obj_nnz_count(word))
            if min(word_nnz_count) == 0:
                break

            sim_vec_seuil = list()

            sim_vec_seuil.append(ev.eval_similarity(vec_seuil, simlex999, print_missing=False))
            sim_vec_seuil.append(ev.eval_similarity(vec_seuil, men, print_missing=False))
            sim_vec_seuil.append(ev.eval_similarity(vec_seuil, ws353, print_missing=False))
            sim_vec_seuil.append(ev.eval_similarity(vec_seuil, scws, print_missing=False))
            
            sim = nanmean(sim_vec_seuil)
            
            print(str(seuil) + ' : ' + str(round(sim, 4)) + ' ', end='')

        print('\n')
        min_threshold = seuil - step
        
        print(f'Low threshold : {min_threshold}\n')

        # High threshold for the number of nnz per dimension
        # Taking the threshold (multiple of 10) for which the similarity is maximal

        vec_seuil = SINrVectors(name_tmp)
        vec_seuil.load(name_tmp + '.pk')

        sim = list()
        seuils = [x for x in range(ceil(max_nnz / step) * step, 0, -step)]

        for s in seuils:
            vec_seuil.remove_communities_dim_nnz(threshold_max = s)
            
            # if a word is no longer represented the dimensions, we stop the filtering
            word_nnz_count = list()
            for word in vec_seuil.vocab:
                word_nnz_count.append(vec_seuil.obj_nnz_count(word))
            if min(word_nnz_count) == 0:
                break

            sim_vec_seuil = list()

            sim_vec_seuil.append(ev.eval_similarity(vec_seuil, simlex999, print_missing=False))
            sim_vec_seuil.append(ev.eval_similarity(vec_seuil, men, print_missing=False))
            sim_vec_seuil.append(ev.eval_similarity(vec_seuil, ws353, print_missing=False))
            sim_vec_seuil.append(ev.eval_similarity(vec_seuil, scws, print_missing=False))
            
            sim.append(nanmean(sim_vec_seuil))
            print(str(s) + ' : ' + str(round(nanmean(sim_vec_seuil), 4)) + ' ', end='')

        print('\n')
        
        max_threshold = seuils[sim.index(nanmax(sim))]
        
        print(f'High threshold : {max_threshold}\nSimilarity with high threshold : {nanmax(sim)}\n')
        
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
        
        # Delete the copy file
        os.remove(name_tmp + '.pk')

        return min_threshold, max_threshold
    
    def get_community_membership(self, obj):
        """Get the community index of a node or label.

        :param obj: an integer of the node or of its label
        :type obj: int or str
        :returns: the community of a specific object

        """
        if self.communities_sets is None:
            raise NoInterpretabilityException
        index = self._get_index(obj)
        return self.community_membership[index]

    def get_community_sets(self, idx):
        """Get the indices of the nodes in for a specific community.

        :param obj: an integer index of a community
        :type obj: int or str
        :param idx: index of the community
        :type idx: int
        :returns: the set of ids of nodes belonging to this community

        """
        if self.communities_sets is None:
            raise NoInterpretabilityException
        return self.communities_sets[idx]

    def _get_index(self, obj):
        """Returns the index for a label or an index.

        :param obj: the object for which the index should be fetched
        :type obj: int or str
        :returns: the index of the object

        """
        if type(obj) is int or type(obj) is int64:
            return obj
        index = self.vocab.index(obj) if self.labels else obj
        return index   
        
    def most_similar(self, obj):
        """Get the most similar objects of the one passed as a parameter using the cosine of their vectors.

        :param obj: the object for which to fetch the nearest neighbors
        :type obj: int or str

        """
        index = self._get_index(obj)

        distances, neighbor_idx = self.neighbors.kneighbors(self.vectors[index, :], return_distance=True)
        # print(similarities, neighbor_idx)
        return {"object ": obj,
                "neighbors ": [(self.vocab[nbr] if self.labels else nbr, round(1 - dist, 2)) for dist, nbr in
                               list(zip(distances.flatten(), neighbor_idx.flatten()))[1::]]}

    def _get_vector(self, idx, row=True):
        """Returns a list from the csr matrix

        :param idx: id of the vector requested
        :type idx: int
        :param row: if the vector should be a row or a column of the csr matrix of embeddings (Default value = True)
        :type row: int
        :returns: the vector

        """
        vector = asarray(self.vectors.getrow(idx).todense()).flatten() if row else asarray(
            self.vectors.getcol(idx).todense()).flatten()
        return vector
    
    def cosine_sim(self, obj1, obj2):
        """Return cosine similarity between specified item of the model

        :param obj1: first object to get embedding
        :type obj1: int or str
        :param obj2: second object to get embedding
        :type obj2: int or str

        :return: cosine similarity between `obj1`and `obj2`
        :rtype: float
        """
        return 1-self.cosine_dist(obj1, obj2)

    def cosine_dist(self, obj1, obj2):
        """Return cosine distance between specified item of the model
        
        :param obj1: first object to get embedding
        :type obj1: int or str
        :param obj2: second object to get embedding
        :type obj2: int or str
        
        :return: cosine distance between `obj1` and `obj2` 
        :rtype: float
        """
        id1 = self._get_index(obj1)
        id2 = self._get_index(obj2)
        vec1 = self._get_vector(id1)
        vec2 = self._get_vector(id2)
        return spatial.distance.cosine(vec1, vec2)
        
    def _get_topk(self, idx, topk=5, row=True):
        """Returns indices of the `topk` values in the vector of id `idx`

        :param idx: idx of the vector in which the topk values are searched
        :type idx: int
        :param topk: number of values to get (Default value = 5)
        :type topk: int
        :param row: if the vector is a row or a column (Default value = True)
        :type row: int
        :returns: the indices of the topk values in the vector
        :rtype: list[int]

        """
        if topk <= 0:
            topk = 1
        vector = self._get_vector(idx, row)
        topk = -topk
        ind = argpartition(vector, topk)[topk:]
        ind = ind[argsort(vector[ind])[::-1]]
        return ind

    def _get_bottomk(self, idx, topk=5, row=True):
        """Returns indices of the `bottomk` values in the vector of id `idx`

        :param idx: idx of the vector in which the bottomk values are searched
        :type idx: int
        :param topk: number of values to get (Default value = 5)
        :type topk: int
        :param row: if the vector is a row or a column (Default value = True)
        :type row: int
        :returns: the indices of the topk values in the vector
        :rtype: list[int]

        """
        if topk <= 0:
            topk = 1
        #print("_get_bottomk, dim", idx)
        #print("_get_bottomk, row", row)
        vector = self._get_vector(idx, row)
        ind = argpartition(vector, topk)[:topk]
        ind = ind[argsort(vector[ind])]
        return ind
    
    def get_topk_dims(self, obj, topk=5):
        """Get `topk` dimensions for an object.

        :param obj: the object for which to get `topk` dimensions
        :type obj: int or str
        :param topk:  (Default value = 5)
        :type topk: int

        :returns: the `topk` dimensions for `obj`
        :rtype: list[int]

        """
        index = self._get_index(obj)
        return self._get_topk(index, topk, True)


    def get_value_obj_dim(self, obj, dim):
        """Get the value of `obj` in dimension `dim`.

        :param obj: an object for which to return the value
        :type obj: int or str
        :param dim: the index of the dimension for which to return the value
        :type dim: int

        :returns: The value of `obj` at dimension `dim`
        :rtype: float
        """
        index = self._get_index(obj)
        vector = self._get_vector(index)
        return vector[dim]

    # Using communities to describe dimensions
    def get_dimension_descriptors(self, obj, topk=-1):
        """Returns the objects that constitute the dimension of obj, i.e. the members of the community of obj

        :param obj: an object for which to return the descriptors
        :type obj: int or str
        :param topk: top values to retrieve for `obj` (Default value = -1)
        :returns: a set of object, the community of obj

        """
        if self.communities_sets is None:
            raise NoInterpretabilityException
        index = self._get_index(obj)
        return self.get_dimension_descriptors_idx(self.community_membership[index], topk)

    def get_dimension_descriptors_idx(self, index, topk=-1):
        """Returns the objects that constitute the dimension of obj, i.e. the members of the community of obj

        :param topk: 1 returns all the members of the community, a positive int returns juste the `topk` members with
        highest `nr` values on the community (Default value = -1)
        :type topk: int
        :param index: the index of the dimension
        :type index: int
        :returns: a set of object, the community of `obj`

        """
        if self.communities_sets is None:
            raise NoInterpretabilityException
        vector = self._get_vector(index, row=False)
        in_dim = InterpretableDimension(index, "descriptors")
        for member in self.get_community_sets(index):
            in_dim.add_interpreter(self.vocab[member], vector[member]) if self.labels else in_dim.add_interpreter(
                member, vector[member])
        in_dim.sort(on_value=True)
        if topk >= 1:
            in_dim.topk(topk)
        return in_dim
        # community_nr = [(round(vector[member], 2), self.vocab[member]) for member in self.get_community_sets(index)]
        # community_nr.sort(key=lambda x: x[0], reverse=True)
        # if topk < 1:
        #     return community_nr
        # else:
        #     topk = min(topk, len(community_nr))
        #     return community_nr[:topk]

    def get_obj_descriptors(self, obj, topk_dim=5, topk_val=-1):
        """Returns the descriptors of the dimensions of obj.

        :param topk_dim: int, topk dimensions to consider to describe obj (Default value = 5)
        :type topk_dim: int
        :param obj: an id or a word/label
        :type obj: int or str
        :param topk_val: 1 returns all the members of the community, a positive int returns juste the topk members with
        highest nr values on the community (Default value = -1)
        :type topk_val: int
        :returns: the dimensions (and the objects that constitute these dimensions) that matter to describe obj

        """
        if self.communities_sets is None:
            raise NoInterpretabilityException
        index = self._get_index(obj)
        highest_dims = self._get_topk(index, topk_dim, row=True)
        vector = self._get_vector(index, row=True)
        highest_dims = [self.get_dimension_descriptors_idx(idx, topk_val).with_value().get_dict() for idx in
                        highest_dims]
        return highest_dims

    # Using top words to describe dimensions
    def get_dimension_stereotypes(self, obj, topk=5):
        """Get the words with the highest values on dimension obj.

        :param obj: id of a word, or label of a word (then turned into the id of its community)
        :type obj: int or str
        :param topk: topk value to consider on the dimension (Default value = 5)
        :type topk: int
        :returns: the topk words that describe this dimension (highest values)

        """
        index = self._get_index(obj)
        if self.community_membership[index] != -1:
            return self.get_dimension_stereotypes_idx(self.get_community_membership(index), topk)
        else:
            raise DimensionFilteredException("'"+self.vocab[index] + "' (id "+str(index)+') is member of a community which got removed by filtering.')

    def get_dimension_stereotypes_idx(self, idx, topk=5):
        """Get the indices of the words with the highest values on dimension obj.

        :param obj: id of a dimension, or label of a word (then turned into the id of its community)
        :type obj: int or str
        :param topk: `topk` value to consider on the dimension (Default value = 5)
        :type topk: int
        :param idx: dimension to fetch `topk` on
        :type idx: int
        :returns: the `topk` words that describe this dimension (highest values)

        """
        highest_idxes = self._get_topk(idx, topk, row=False)
        vector = self._get_vector(idx, row=False)
        in_dim = InterpretableDimension(idx, "stereotypes")
        for idx in highest_idxes:
            in_dim.add_interpreter(self.vocab[idx], vector[idx]) if self.labels else in_dim.add_interpreter(idx,
                                                                                                            vector[idx])
        return in_dim

    def get_obj_stereotypes(self, obj, topk_dim=5, topk_val=3):
        """Get the top dimensions for a word.

        :param obj: the word to consider
        :type obj: int or str
        :param topk_dim: `topk` dimension to consider (Default value = 5)
        :type topk_dim: int
        :param topk_val: `topk` values to describe each dimension (Default value = 3)
        :type topk_val: int
        :returns: the most useful dimensions to describe a word and for each dimension,
        the topk words that describe this dimension (highest values)

        """
        # get index of the word considered
        index = self._get_index(obj)
        # get the topk dimensions for this word
        highest_dims = self._get_topk(index, topk_dim, row=True)
        vector = self._get_vector(index, row=True)
        highest_dims = [self.get_dimension_stereotypes_idx(idx, topk_val).with_value().get_dict() for idx in
                        highest_dims]
        return highest_dims

    def get_obj_stereotypes_and_descriptors(self, obj, topk_dim=5, topk_val=3):
        """Get the stereotypes and descriptors for obj.

        :param obj: object for which to fetch stereotypes and descriptors
        :type obj: int or str
        :param topk_dim: number of dimensions to consider  (Default value = 5)
        :type topk_dim: int
        :param topk_val: number of values per dimension (Default value = 3)
        :type topk_val: int
        :returns: both stereotypes and descriptors

        """
        sters = self.get_obj_stereotypes(obj, topk_dim, topk_val)
        descs = self.get_obj_descriptors(obj, topk_dim, topk_val)
        for ster, desc in zip(sters, descs):
            ster["descriptors"] = desc["descriptors"]
        return sters

    def get_number_of_dimensions(self):
        """Get the number of dimensions of model.


        :returns: Number of dimensions of the model.

        :rtype: int

        """
        return self.vectors.shape[1]
    
    def get_vocabulary_size(self):
        """
        :returns: Number of words that constitute the vocabulary
        :rtype: int
        """
        return self.vectors.shape[0]

    def _prcnt_vocabulary(self, prct:int):
        """
        :param prct: percentage of the vocabulary required
        :type prct: int
        :returns: number of words required to deal with prct percents of the vocabulary
        :rtype: int
        """
        return (int)(round(prct * self.get_vocabulary_size() / 100, 0))

    def get_union_topk(self, prct:int):
        """
        :param prct: percentage of the vocabulary among the top for each dimension
        :type prct: int
        
        :returns: list of the ids of words that are among the top prct of the dims, can be useful to pick intruders
        :rtype: int list
        """
        nb =  self._prcnt_vocabulary(prct)
        #print(nb)
        intruder_candidates = set()
        for i in tqdm(range(self.get_number_of_dimensions())):
            #print("nb", nb,", dim ", i,  ", topk for union", self._get_topk(i, topk = nb, row=False))
            intruder_candidates =  intruder_candidates.union(self._get_topk(i, topk = nb, row=False))
        return intruder_candidates

    def pick_intruder(self, dim, union=None, prctbot=50, prcttop=10):
        """Pick an intruder word for a dimension
        
        :param dim: the index of the dimension for which to return intruders
        :type dim: int
        :param union: ids of words that are among the top prct of at least one dimension (defaults to None)
        :type union: int list
        :param prctbot: bottom prctbot to pick (defaults to 50)
        :type prctbot: int
        :param prcttop: top prcttop to pick (defaults to 10)
        :type prcttop: int
        
        :returns: ids of an intruder word from the dimension
        :rtype: int
        
        """
        
        #we search intruders which are:
        #- words from the bottom bottomk of the dimension dim (bottoms)
        #- words which are in the top prcttop of a dimension of the sinrVector (union)
        
        bottomk = self._prcnt_vocabulary(prctbot)
        bottoms = self._get_bottomk(dim, topk=bottomk, row=False)
        
        if union is None:
            union = self.get_union_topk(prct=prcttop)
            
        intersection = union.intersection(bottoms)
        
        if (len(intersection) <= 0):
            raise NoIntruderPickableException
            
        alea = randint(0,len(intersection)-1)
        
        return list(intersection)[alea]
    
    def intra_sim(self, topks, dist=True):
        """ Get the average cosine distance (or cosine similarity) between top words
        
        :param topks: number of top words to pick
        :type topks: int
        :param dist: set to True (default) to use cosine distance and False to use cosine similarity
        :type dist: boolean
        
        :returns: average cosine distance (or cosine similarity) between top words
        :rtype: float
        """
        topks = [self._get_index(o) for o in topks]
        k = 0
        cosine = 0.0
        for i,j in combinations(topks,2):
            if dist:
                cosine += self.cosine_dist(i,j)
            else:
                cosine += self.cosine_sim(i,j)
            k += 1
        return cosine / k
    
    def inter_sim(self, intruder, topk, dist=True):
        """ Get the average cosine distance (or cosine similarity) between top words and the intruder word
        
        :param intruder: id of the intruder word
        :type intruder: int
        :param topk: number of top words to consider
        :type topk: int
        :param dist: set to True (default) to use cosine distance and False to use cosine similarity
        :type dist: boolean
        
        :returns: average cosine distance (or cosine similarity) between top words and the intruder word
        :rtype: float
        """
        cosine = 0.0
        for o in topk:
            if dist:
                cosine += self.cosine_dist(intruder,o)
            else:
                cosine += self.cosine_sim(intruder,o)
        return cosine / (float)(len(topk))


    def load(self, path=None):
        """Load a SINrVectors model.
        
        :param path: Path of the pickle file of the model.
        :type path: string
        
        """
        if path != None:
            f = open(path, 'rb')
        else:
            f = open(self.name + '.pk', 'rb')
        tmp_dict = pk.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, path=None):
        """Save a SINrVectors model.
        
        :param path: Path of the pickle file of the model.
        :type path: string
        
        """
        if path != None:
            f = open(path, 'wb')
        else:
            f = open(self.name + '.pk', 'wb')
        pk.dump(self.__dict__, f, 2)
        f.close()

    def get_my_vector(self, obj, row=True):
        """Get the column or the row obj.

        :param obj: Index of the row/column to return.
        :type obj: int
        :param row: Return a row if True else a column. Defaults to True.
        :type row: bool
        :returns: A row/column.
        :rtype: np.ndarray

        """
        index = self._get_index(obj)
        vector = asarray(self.vectors.getrow(index).todense()).flatten() if row else asarray(
            self.vectors.getcol(index).todense()).flatten()
        return vector
    
    def light_model_save(self):
        """Save a minimal version of the model that is readable as a `dict` for evaluation on `word-embeddings-benchmark`
        https://github.com/kudkudak/word-embeddings-benchmarks


        """
        data={}
        for item in self.vocab :
            data[item]=self.get_my_vector(item)
        f = open(self.name + "_light.pk", 'wb')
        pk.dump(data,f)
        f.close()

class DimensionFilteredException(Exception):
    """Exception raised when trying to access a dimension removed by filtering. """
    
    pass


