import os

from networkit import Graph, components, community, setNumberOfThreads, graphio 
import pickle as pk

from .nfm import get_nfm_embeddings

from .logger import logger

from . import strategy_norm
from . import strategy_loader

from sklearn.neighbors import NearestNeighbors


class SINr(object):

    def __init__(self, graph, lgcc, wrd_to_idx, n_jobs=1):
        """
        :param matrix_path: Path to coocurrence matrix.
        :type matrix_path: `str`
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
        graph = Graph(weighted=True) 
        rows, cols = matrix.row, matrix.col
        weights = matrix.data
        for row, col, weight in zip(rows, cols, weights):
                graph.addEdge(u=row, v=col, w=weight, addMissing=True)
        return graph
        
    @staticmethod
    def getLgcc(graph):
        out_of_LgCC = set(graph.iterNodes()) - set(components.ConnectedComponents.extractLargestConnectedComponent(graph).iterNodes()) # Extract out of largest connected component vocabulary
        return out_of_LgCC
        
    @classmethod
    def load_from_cooc_pkl(cls, cooc_matrix_path, norm=None, n_jobs=1):
        """
        Build a sinr object from a co-occurrence matrix : useful to deal with textual data.
        
        :param cooc_matrix_path: path to the pickle obtaiuned using text.cooccurrence
        
        :type cooc_matrix_path: `pickle`

        :param norm: Normalization to apply to edges, used in IDA paper, but deprecated, default at None works fine

        :type norm: `strategy_norm`

        :return: A weighted graph.
        :rtype: `networkit.Graph`
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
        Build a sinr object from a matrix object

        :param norm: Normalization to apply to edges, used in IDA paper, but deprecated, default at None works fine

        :type norm: `strategy_norm`

        :return: A weighted graph.
        :rtype: `networkit.Graph`
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
        Build a sinr object from a graph

        :param norm: Normalization to apply to edges, used in IDA paper, but deprecated, default at None works fine

        :type norm: `strategy_norm`

        :return: A weighted graph.
        :rtype: `networkit.Graph`
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

    def detect_communities(self, gamma=1, algo=None, inspect=True ):
        """
        Detect the communities in a graph.

        :param G: A graph.
        :param algo: The community detection algorithm.

        :type G: Networkit.Graph
        :type algo: Networkit.community.PLP (PLM, LPDegreeOrdered)

        :return: The partition representing the community structure.

        :rtype: Networkit.community.Partition
        """
        logger.info("Detecting communities.")
        if algo == None:
            algo = community.PLM(self.cooc_graph, refine=False, gamma=gamma, turbo=True, recurse=False)
        communities = community.detectCommunities(self.cooc_graph, algo=algo, inspect=inspect)
        communities.compact(useTurbo=True) #Consecutive communities from 0 to number of communities - 1
        self.communities = communities
        logger.info("Finished detecting communities.")
        return communities

    def get_out_of_LgCC_coms(self, communities):
        set_out_of_LgCC = set(self.out_of_LgCC)
        out_of_LgCC_coms = []
        for com in communities.getSubsetIds():
            if set(communities.getMembers()) & set_out_of_LgCC != {}:
                out_of_LgCC_coms.append(com)
        return out_of_LgCC_coms


    def extract_embeddings(self, communities):
        """
        Extract the word embeddings from the graph and its community structure. The word embeddings are computed by
        counting the number of neighbors in each community.

        :param G: A graph.
        :param communities: A community structure.

        :type G: Networkit.graph
        :type communities: Networkit.community.Partitions

        :return: The word embeddings.
        :rtype: `graph_embeddings.Model()`
        """
        logger.info("Extracting embeddings.")

        logger.info("Applying NFM.")
        np, nr, nfm = get_nfm_embeddings(self.cooc_graph, communities.getVector(), self.n_jobs)
        self.np = np
        self.nr = nr
        self.nfm = nfm
        logger.info("NFM successfully applied.")
        logger.info("Finished extracting embeddings.")

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
    
    def get_communities(self):
        if hasattr(self, 'communities'):
            return self.communities
        else:
            raise NoCommunityDetectedException
            
    def run(self, output_path=None, mat=None, dictionary=None, norm=None, algo=None):
        #G = self.build_graph(norm=norm)
        if algo == None:
            algo = community.PLM(self.cooc_graph, refine=False, gamma=1, turbo=True, recurse=False)
        self.communities = self.detect_communities(self.cooc_graph, algo=algo)
        model = self.extract_embeddings(G=self.cooc_graph, communities=self.communities)
        
        #if output_path:
        #    self.save(output_path, G=self.cooc_graph, communities=self.communities, model=model)
        return model

    def save(self, output_path, G=None, communities=None, model=None):
        if G != None:
            graphio.writeGraph(G, os.path.join(output_path, 'graph.METIS'), graphio.Format.METIS)
        if communities != None:
            communities = {c: communities.getMembers(c) for c in communities.getSubsetIds()}
            with open(os.path.join(output_path ,'communities.dict.pickle'), 'wb') as out:
                pk.dump(communities, out)
        if model != None:
            #embeddings.save(os.path.join(output_path, "embeddings.kv"))
            #embeddings.save_word2vec_format(os.path.join(output_path, "embeddings.w2v_format.keyedvectors"))
            model.save(os.path.join(output_path, "model.pk"))
        with open(os.path.join(output_path, 'wrd_to_idx.pickle'), 'wb') as w2i:
            pk.dump(self.wrd_to_idx, w2i)
        with open(os.path.join(output_path, 'idx_to_wrd.pickle'), 'wb') as i2w:
            pk.dump(self.idx_to_wrd, i2w)


    def _flip_keys_values(self, dictionary):
        return dict((v, k) for k,v in dictionary.items())

class NoCommunityDetectedException(Exception):
    "Raised when the communities were not detected"
    pass

class NoEmbeddingExtractedException(Exception):
    "Raised when the embeddings were not extracted"
    pass


class ModelBuilder:
    def __init__(self, sinr, name, n_jobs=1, n_neighbors=31):
        self.sinr = sinr
        self.model = SINrVectors(name, n_jobs, n_neighbors)
        
    def with_embeddings_nr(self):
        self.model.set_vectors(self.sinr.get_nr())
        return self
        
    def with_embeddings_nfm(self):
        self.model.set_vectors(self.sinr.get_nfm())
        return self
        
    def with_np(self):
        self.model.set_np(self.sinr.get_np())
        return self
        
    def with_vocabulary(self):
        self.model.set_vocabulary(self.sinr.get_vocabulary())
        return self
        
    def with_communities(self):
        self.model.set_communities(self.sinr.get_)
        
    def build(self):
        return self.model
        
        
    
class SINrVectors(object):

    def __init__(self, name, n_jobs, n_neighbors):
        self.name = name
        self.n_jobs=n_jobs
        self.n_neighbors=n_neighbors
        
    def set_n_jobs(self, n_jobs):
        self.n_jobs = n_jobs
        
    def set_vocabulary(self, voc):
        self.vocab=voc
        
    def set_vectors(self, embeddings):
        self.vectors=embeddings
        self.neighbors = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine', n_jobs=self.n_jobs).fit(self.vectors)
        
    def set_np(self, np):
        self.np = np
        
    def set_communities(self, com):
        self.communities = com
        
    def most_similar(self, word):
        similarities, neighbor_idx = self.neighbors.kneighbors(self.vectors[self.vocab.index(word),:], return_distance=True)
        #print(similarities, neighbor_idx)
        return {"word":word,
               "neighbors": [(self.vocab[nbr], 1-sim) for sim, nbr in list(zip(similarities.flatten(), neighbor_idx.flatten()))[1::]]}
        
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
