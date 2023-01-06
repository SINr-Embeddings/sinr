
from networkit import Graph, components, community, setNumberOfThreads
import pickle as pk

from .nfm import get_nfm_embeddings

from .logger import logger

from . import strategy_norm
from . import strategy_loader

from sklearn.neighbors import NearestNeighbors
from numpy import argpartition, argsort, asarray


class SINr(object):
    """
    Object that can be used to extract word or graph embeddings using the SINr approach.
    This object cannot then be used to inspect the resulting vectors. Instead, using the ModelBuilder class, a SINrVectors object should be created that will allow to use the resulting vectors.

    ...

    Attributes
    ----------
    Attributes should not be read
    """
    
        
    @classmethod
    def load_from_cooc_pkl(cls, cooc_matrix_path, norm=None, n_jobs=1):
        """
        Build a sinr object from a co-occurrence matrix stored as a pickle : useful to deal with textual data.
        Co-occurrence matrices should for instance be generated using sinr.text.cooccurrence

        Parameters
        ----------
        cooc_matrix_path : string
            Path to the cooccurrence matrix generated using sinr.text.cooccurrence : the file should be a pickle
        norm : strategy_norm, optional
            If the graph weights be normalized (for example using PMI). The default is None.
        n_jobs : int, optional
            Number of jobs that should be used The default is 1.

        Returns
        -------
        TYPE
            A SINr object, this method acts as a factory.

        """
        logger.info("Building Graph.")
        
        word_to_idx, matrix = strategy_loader.load_pkl_text(cooc_matrix_path)
        graph = SINr.getGraphFromMatrix(matrix)
        graph = strategy_norm.apply_norm(graph, norm)
        out_of_LgCC = SINr.getLgcc(graph)
        logger.info("Finished building graph.")
        return cls(graph, out_of_LgCC, word_to_idx)
    
    @classmethod
    def load_from_adjacency_matrix(cls, matrix_object, norm=None, n_jobs=1):
        """
        Build a sinr object from an adjacency matrix as a sparse one (csr)

        Parameters
        ----------
        matrix_object : csr_matrix
            Matrix describing the graph.
        norm : strategy_norm, optional
            If the graph weights be normalized (for example using PMI). The default is None.
        n_jobs : int, optional
            Number of jobs that should be used The default is 1.

        Returns
        -------
        TYPE
            A SINr object, this method acts as a factory.

        """
        logger.info("Building Graph.")
        word_to_idx, matrix = strategy_loader.load_adj_mat(matrix_object)
        graph = SINr.getGraphFromMatrix(matrix)
        graph = strategy_norm.apply_norm(graph, norm)
        out_of_LgCC = SINr.getLgcc(graph)
        logger.info("Finished building graph.")
        return cls(graph, out_of_LgCC, word_to_idx)
    
    @classmethod
    def load_from_graph(cls, graph,  norm=None, n_jobs=1):
        """
        Build a sinr object from a networkit graph object

        Parameters
        ----------
        graph : networkit
            Networkit graph object.
        norm : strategy_norm, optional
            If the graph weights be normalized (for example using PMI). The default is None.
        n_jobs : int, optional
            Number of jobs that should be used The default is 1.

        Returns
        -------
        TYPE
            A SINr object, this method acts as a factory.

        """
        word_to_idx = dict()
        idx = 0
        for u in graph.iterNodes():
            word_to_idx[u] = idx
            idx+=1
        graph = strategy_norm.apply_norm(graph, norm)
        out_of_LgCC = SINr.getLgcc(graph)
        logger.info("Finished building graph.")
        return cls(graph, out_of_LgCC, word_to_idx)

    def run(self, algo=None):
        """
        Runs the training of the embedding, i.e. community detection + vectors extraction

        Parameters
        ----------
        algo : networkit.algo.community, optional
            Community detection algorithm. The default, None allorws to run a Louvain algorithm

        Returns
        -------
        None.

        """
        #G = self.build_graph(norm=norm)
        if algo == None:
            algo = community.PLM(self.cooc_graph, refine=False, gamma=100, turbo=True, recurse=False)
        self.communities = self.detect_communities(self.cooc_graph, algo=algo)
        self.extract_embeddings(G=self.cooc_graph, communities=self.communities)

    
    def detect_communities(self, gamma=100, algo=None, inspect=True ):
        """
        Runs community detection on the graph

        Parameters
        ----------
        gamma : int, optional
            For Louvain algorithm which is the default algorithm (ignore this parameter if param algo is used), allows to control the size of the communities. The greater it is, the smaller the communities. The default is 100.
        algo : networkit.algo.community, optional
            Community detection algorithm. The default, None allorws to run a Louvain algorithm
        inspect : boolean, optional
            Whether or not one wants to get insight about the communities extracted. The default is True.

        Returns
        -------
        communities : networkit partition
            Community structure

        """
        logger.info("Detecting communities.")
        if algo == None:
            algo = community.PLM(self.cooc_graph, refine=False, gamma=gamma, turbo=True, recurse=False)
        communities = community.detectCommunities(self.cooc_graph, algo=algo, inspect=inspect)
        communities.compact(useTurbo=True) #Consecutive communities from 0 to number of communities - 1
        self.communities = communities
        logger.info("Finished detecting communities.")
        return communities


    def extract_embeddings(self, communities):
        """
        

        Parameters
        ----------
        communities : networkit partition
            Community structures

        Returns
        -------
        Extracts the vectors and store them in the SINr object

        """
        logger.info("Extracting embeddings.")

        logger.info("Applying NFM.")
        np, nr, nfm = get_nfm_embeddings(self.cooc_graph, communities.getVector(), self.n_jobs)
        self.np = np
        self.nr = nr
        self.nfm = nfm
        logger.info("NFM successfully applied.")
        logger.info("Finished extracting embeddings.")

    
    def __init__(self, graph, lgcc, wrd_to_idx, n_jobs=1):
        """
        Should not be used ! Some factory methods below starting with "load_" should be used instead.

        Parameters
        ----------
        graph : networkit graph
        lgcc : networkit graph
            the largest connected component of graph
        wrd_to_idx : dict
            A matching between a vocabulary and ids. Useful for text. Otherwise, the vocabulary and the ids are the same.
        n_jobs : int, optional
            Number of jobs that should be runned. The default is 1.

        Returns
        -------
        None.

        """
        self.n_jobs = n_jobs
        setNumberOfThreads(n_jobs)
        self.wrd_to_idx = wrd_to_idx
        self.idx_to_wrd = self._flip_keys_values(self.wrd_to_idx)
        self.cooc_graph = graph
        self.out_of_LgCC = lgcc
        
        self.wd_before, self.wd_after = None, None
       
    @staticmethod
    def getGraphFromMatrix(matrix):
        """

        Parameters
        ----------
        matrix : coo_matrix
            A sparse matrix describing a graph

        Returns
        -------
        graph : networkit graph
            the graph corresponding to the coo matrix

        """
        graph = Graph(weighted=True) 
        rows, cols = matrix.row, matrix.col
        weights = matrix.data
        for row, col, weight in zip(rows, cols, weights):
                graph.addEdge(u=row, v=col, w=weight, addMissing=True)
        return graph
        
    @staticmethod
    def getLgcc(graph):
        """

        Parameters
        ----------
        graph : networkit graph
            

        Returns
        -------
        out_of_LgCC : networkit graph
            the largest connected component of the graph provided as a parameter

        """
        out_of_LgCC = set(graph.iterNodes()) - set(components.ConnectedComponents.extractLargestConnectedComponent(graph).iterNodes()) # Extract out of largest connected component vocabulary
        return out_of_LgCC

    def get_out_of_LgCC_coms(self, communities):
        set_out_of_LgCC = set(self.out_of_LgCC)
        out_of_LgCC_coms = []
        for com in communities.getSubsetIds():
            if set(communities.getMembers()) & set_out_of_LgCC != {}:
                out_of_LgCC_coms.append(com)
        return out_of_LgCC_coms
    
    def get_cooc_graph(self):
        return self.cooc_graph
    
    def get_nr(self):
        if hasattr(self, 'nr'):
            return self.nr
        else:
            raise NoEmbeddingExtractedException
            
    def get_np(self):
        if hasattr(self, 'np'):
            return self.np
        else:
            raise NoEmbeddingExtractedException

    def get_nfm(self):
        if hasattr(self, 'nfm'):
            return self.nfm
        else:
            raise NoEmbeddingExtractedException    
    
    def get_vocabulary(self):
        return list(self.idx_to_wrd.values())
    
    def get_wrd_to_id(self):
        return self.wrd_to_idx
    
    def get_communities(self):
        if hasattr(self, 'communities'):
            return self.communities
        else:
            raise NoCommunityDetectedException

    def _flip_keys_values(self, dictionary):
        return dict((v, k) for k,v in dictionary.items())

class NoCommunityDetectedException(Exception):
    "Raised when the communities were not detected"
    pass

class NoEmbeddingExtractedException(Exception):
    "Raised when the embeddings were not extracted"
    pass


class ModelBuilder:
    """
    Object that should be used after the training of word or graph embeddings using the SINr object.
    The ModelBuilder will make use of the SINr object to build a SINrVectors object that will allow to use the resulting vectors efficiently.
    ...

    Attributes
    ----------
    Attributes should not be read
    """
    def __init__(self, sinr, name, n_jobs=1, n_neighbors=31):
        """
        Creating a ModelBuilder object to build a SINrVectors one

        Parameters
        ----------
        sinr : SINr
            A SINr object with extracted vectors
        name : string
            Name of the model
        n_jobs : TYPE, optional
            DESCRIPTION. The default is 1.
        n_neighbors : int, optional
            Number of neighbors to use for similarity. The default is 31.

        Returns
        -------
        None.

        """
        self.sinr = sinr
        self.model = SINrVectors(name, n_jobs, n_neighbors)
        
    def with_embeddings_nr(self):
        """
        Adding Node Recall vectors to the SINrVectors object

        """
        self.model.set_vectors(self.sinr.get_nr())
        return self
        
    def with_embeddings_nfm(self):
        """
        Adding NFM (Node Recall + Node Predominance) vectors to the SINrVectors object

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.model.set_vectors(self.sinr.get_nfm())
        return self
        
    def with_np(self):
        """
        Storing Node predominance values in order to label dimensions for instance

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.model.set_np(self.sinr.get_np())
        return self
        
    def with_vocabulary(self):
        """
        To deal with word vectors or graph when nodes have labels

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.model.set_vocabulary(self.sinr.get_vocabulary())
        return self
        
    def with_communities(self):
        """
        To keep the interpretability of the model using the communities

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.model.set_communities(self.sinr.get_communities())
        return self
        
    def build(self):
        """
        To get the SINrVectors object

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.model
        
class NoInterpretabilityException(Exception):
    "Raised when the communities were not included in the model that was built. It is thus not interpretable anymore."
    pass

class NoVocabularyException(Exception):
    "Raised when no vocabulary was included in the model that was built. One cant play with words."
    pass
    
class SINrVectors(object):

    def __init__(self, name, n_jobs, n_neighbors):
        self.name = name
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.labels = False
        
    def set_n_jobs(self, n_jobs):
        self.n_jobs = n_jobs
        
    def set_vocabulary(self, voc):
        self.vocab = voc
        #self.wrd_to_id = wrd_to_id
        self.labels = True
        
    def set_vectors(self, embeddings):
        self.vectors = embeddings
        self.neighbors = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine', n_jobs=self.n_jobs).fit(self.vectors)
        
    def set_np(self, np):
        self.np = np
        
    def set_communities(self, com):
        self.communities = com
        
    def _get_index(self, obj):
        index = self.vocab.index(obj) if self.labels else obj
        return index
    
    def most_similar(self, obj):
        index = self._get_index(obj)
            
        distances, neighbor_idx = self.neighbors.kneighbors(self.vectors[index,:], return_distance=True)
        #print(similarities, neighbor_idx)
        return {"object ": obj,
                   "neighbors ": [(self.vocab[nbr] if self.labels else nbr , 1-dist) for dist, nbr in list(zip(distances.flatten(), neighbor_idx.flatten()))[1::]]}
    
    
    
    
    
    def _get_vector(self, idx, row=True):
        vector = asarray(self.vectors.getrow(idx).todense()).flatten() if row else  asarray(self.vectors.getcol(idx).todense()).flatten()
        return vector 
    
    def _get_topk(self, idx, topk=5, row=True):
        vector = self._get_vector(idx, row)
        topk = -topk
        ind = argpartition(vector, topk)[topk:]
        ind = ind[argsort(vector[ind])[::-1]]
        return ind
        
    #Using communities to describe dimensions
    def get_dimension_descriptors(self, obj):
        "returns the other objects that constitute the dimension (i.e. the community) of obj"
        index = self._get_index(obj)
        return self.get_dimension_descriptors_idx(self.communities.subsetOf(index))
    
    def get_dimension_descriptors_idx(self, idx):
        return [self.vocab[member]  if self.labels else member for member in self.communities.getMembers(idx)]

    def get_obj_descriptors(self, obj, topk=5):
        "returns the dimensions (and the objects that constitute these dimensions) that matter to describe obj"
        index = self._get_index(obj)
        highest_dims = self._get_topk(index, topk, row=True)
        vector = self._get_vector(index, row=True)
        highest_dims = [{"dimension" : idx, "value" : vector[idx], "descriptors" : self.get_dimension_descriptors_idx(idx)} for idx in highest_dims]
        return highest_dims
    
    
    
    #Using words to describe dimensions
    def get_dimension_stereotypes(self, obj, topk=5):
        index = self._get_index(obj)
        return self.get_dimension_stereotypes_idx(self.communities.subsetOf(index), topk)
    
    def get_dimension_stereotypes_idx(self, idx, topk=5):
        highest_idxes = self._get_topk(idx, topk, row=False)
        highest_idxes = [self.vocab[idx] if self.labels else idx for idx in highest_idxes]
        return highest_idxes
    
    #Using wor
    def get_obj_stereotypes(self, obj, topk=5):
        #get index of the word considered
        index = self._get_index(obj)
        #get the topk dimensions for this word
        highest_dims = self._get_topk(index, topk, row=True)
        vector = self._get_vector(index, row=True)
        highest_dims = [{"dimension" : idx, "value" : vector[idx], "stereotypes" : self.get_dimension_stereotypes_idx(idx)} for idx in highest_dims]
        return highest_dims
             
    def load(self):
        f = open(self.name, 'rb')
        tmp_dict = pk.load(f)
        f.close()          
        self.__dict__.update(tmp_dict) 

    def save(self):
        f = open(self.name, 'wb')
        pk.dump(self.__dict__, f, 2)
        f.close()
    
#    def save(self, output_path):
#        with open(output_path, 'wb+') as file:
#            pk.dump((self.vocab, self.vectors), file)

#    def load(model_path):
#        with open(model_path, 'rb') as file:
#            vocab, vectors = pk.load(file)
#        return Model(vocab, vectors)
