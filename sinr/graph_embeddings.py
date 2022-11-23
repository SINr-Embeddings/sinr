import os
import logging
from scipy.sparse import csr_matrix
from networkit import Graph, components, community, setNumberOfThreads, graphio 
from pathlib import PurePath
import pickle as pk
import numpy as np
import parallel_sort
#from gensim.models import KeyedVectors
#from .sparse_nfm import get_nfm_embeddings
from .nfm import get_nfm_embeddings
import copy

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

if not len(logger.handlers):
        logger.addHandler(ch)

logger.setLevel(logging.INFO)

class EmbeddingExtraction(object):

    def nfm_extraction(G, vector, nb_com, nb_nodes,  n_jobs):
        #print("Vector size : ", len(communities.getVector()))
        return get_nfm_embeddings(G, vector, nb_com,nb_nodes)[2] #Concatenation of Node F-measure and Node Predominance


class WeightNorm(object):

    def apply_norm(G, norm):
        """
        Apply a normalisation to the graph (IPMI, PMI...) before detecting communities

        :param norm: Type of normalisation.
        :param norm_args: Arguments to pass to the normalisation alfgorithm.
        :return: The graph and the weighted degrees before and after normalisation.
        """
        #try:
        G = norm(G)#, wd_before, wd_after = norm(G)
        #except:
         #   raise TypeError('Unknown norm type')
        return G#, wd_before, wd_after

    def iterative_pmi(G):
        """
        Iteratively applies PMI to the graph to allow for community detection.

        @return:  The graph and the weighted degrees before and after normalisation.
        """
        logger.info("Starting IPMI normalisation.")
        # Using Decorate-Sort-Undecorate as apparently it's more efficient for large lists
        dtype = [('edge', tuple), ('weighted_degree', int)]
        edges_wd = np.array([(e, G.weightedDegree(e[0]) + G.weightedDegree(e[1])) for e in G.iterEdges() ], dtype=dtype)
        #edges_wd[::-1].sort(order='weighted_degree') # Sort by weighted degree of edge
        #edges = sorted(list(G.iterEdges()),
                       #key=lambda x: G.weightedDegree(x[0]) + G.weightedDegree(x[1]), reverse=True)
        edges = edges_wd['edge']
        edges = edges[parallel_sort.parallel_argsort(edges_wd['weighted_degree'])[::-1]]
        #wd_before = {e: G.weightedDegree(e) for e in G.iterNodes()}
        ipmi = lambda u,v: (G.weight(u, v) / (G.weightedDegree(u) * G.weightedDegree(v)))
        nb_edges = edges.shape[0]
        cnt = 0
        for u, v in edges:
            G.setWeight(u, v,
                        w= ipmi(u,v))
            if cnt%1000000 == 0:
                logger.info("Edge %i/%i normalized."%(cnt, nb_edges))
            cnt+=1
        #wd_after = {e: G.weightedDegree(e) for e in G.iterNodes()}
        logger.info("Finished IPMI normalisation.")
        return G#, wd_before, wd_after


class SINr(object):

    def __init__(self, cooc_matrix_path, n_jobs=1):
        """
        :param matrix_path: Path to coocurrence matrix.
        :type matrix_path: `str`
        """
        self.n_jobs = n_jobs
        setNumberOfThreads(n_jobs)
        self.wrd_to_idx, self.matrix = self.load(cooc_matrix_path)
        self.idx_to_wrd = SINr._flip_keys_values(self.wrd_to_idx)
        self.wd_before, self.wd_after = None, None
        self.cooc_graph = None
        self.out_of_LgCC = None

    def build_graph(self, norm=WeightNorm.iterative_pmi, n_jobs=1):
        """
        Build a graph and filter its nodes.

        :param norm: Normalization to apply to edges.
        :param rm_hg_deg: Number of highest degree nodes to remove.
        :param k_core: Minimum size of k-core in graph.

        :type norm: `WeightNorm`
        :type rm_hg_deg: `int`
        :type k-core: `int`

        :return: A weighted graph.
        :rtype: `networkit.Graph`
        """
        logger.info("Building Graph.")
        graph = Graph(weighted=True)
        rows, cols = self.matrix.row, self.matrix.col
        weights = self.matrix.data

        for row, col, weight in zip(rows, cols, weights):
                graph.addEdge(u=row, v=col, w=weight, addMissing=True)
        self.cooc_graph = Graph(graph, weighted=True)
        if norm != None:
            graph = WeightNorm.apply_norm(graph, norm)
            #, self.wd_before, self.wd_after = WeightNorm.apply_norm(graph, norm)
        #print(list(graph.iterEdgesWeights())[:50])
        self.out_of_LgCC = set(graph.iterNodes()) - set(components.ConnectedComponents.extractLargestConnectedComponent(graph).iterNodes()) # Extract out of largest connected component vocabulary
        logger.info("Finished building graph.")
        return graph

    def load(self, mat_path):
        """
        Load a cooccurrence matrix.

        :param cooc_mat_path: Path to coocurrence matrix.

        :type cooc_mat_path : str

        :return: The loaded cooccurrence matrix and the word index.

        :rtype: `tuple(dict(), scipy.sparse.coo_matrix)`
        """
        logger.info("Loading cooccurrence matrix and dictionary.")
        with open(mat_path, "rb") as file:
            word_to_idx, mat = pk.load(file)
        logger.info("Finished loading data.")
        return (word_to_idx, mat)

    def detect_communities(self, G, algo=community.PLP):
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
        communities = community.detectCommunities(G, algo=algo(G))
        communities.compact(useTurbo=True) #Consecutive communities from 0 to number of communities - 1
        logger.info("Finished detecting communities.")
        return communities

    def get_out_of_LgCC_coms(self, communities):
        set_out_of_LgCC = set(self.out_of_LgCC)
        out_of_LgCC_coms = []
        for community in communities.getSubsetIds():
            if set(communities.getMembers()) & set_out_of_LgCC != {}:
                out_of_LgCC_coms.append(community)
        return out_of_LgCC_coms


    def extract_embeddings(self, G, communities, extraction=EmbeddingExtraction.nfm_extraction, n_jobs=1):
        """
        Extract the word embeddings from the graph and its community structure. The word embeddings are computed by
        counting the number of neighbors in each community.

        :param G: A graph.
        :param communities: A community structure.
        :param extraction: The extraction method for the word embeddings.

        :type G: Networkit.graph
        :type communities: Networkit.community.Partitions
        :type extraction: EmbeddingExtraction

        :return: The word embeddings.
        :rtype: `graph_embeddings.Model()`
        """
        logger.info("Extracting embeddings.")
#        nodes  = set(G.iterNodes())
        logger.info("Applying NFM.")
        embeddings = extraction(G=self.cooc_graph, vector=communities.getVector(), nb_com=communities.numberOfSubsets(), nb_nodes=self.cooc_graph.numberOfNodes(), n_jobs=n_jobs)
        #print(embeddings)
        logger.info("NFM successfully applied.")

#        if self.out_of_LgCC: #Even though we extracted the embeddings in communities outside of the largest connected component, we do not wish to extract them in our model
#            out_of_LgCC_coms = np.array([com_ids_map[i] for i in self.get_out_of_LgCC_coms(communities)])
#
#            if embeddings.shape[1] == 2 * nb_com: # Concatenation of node predominance and node recall
#                out_of_LgCC_coms = np.append(out_of_LgCC_coms, out_of_LgCC_coms+nb_com, 0) # Delete component for node recall and node predominance
#            embeddings = np.delete(embeddings, self.out_of_LgCC, axis=0) # Delete words out of LgCC from embeddings
#            embeddings = np.delete(embeddings, out_of_LgCC_coms, axis=1) # Delete components corresponding to communities out of LgCC

#            nodes      =sorted( list(nodes - self.out_of_LgCC))

#        words = [self.idx_to_wrd[i] for i in nodes] 
        words = list(self.idx_to_wrd.values())
        #model = KeyedVectors(embeddings.shape[1])
        #model.add(words, embeddings)
        #del model.vectors_norm
        logger.info("Finished extracting embeddings.")
        return Model(words, embeddings)

    def sinr(cooc_mat_path, output_path=None, mat=None, dictionary=None, n_jobs=1, norm=WeightNorm.iterative_pmi, algo=community.PLP, extraction=EmbeddingExtraction.nfm_extraction):
        sinr = SINr(cooc_mat_path, n_jobs=n_jobs)
        G = sinr.build_graph(norm=norm, n_jobs=n_jobs)
        communities = sinr.detect_communities(G, algo=algo)
        model = sinr.extract_embeddings(G=sinr.cooc_graph, communities=communities, extraction=extraction, n_jobs=n_jobs)
        if output_path:
            sinr.save(output_path, G=G, communities=communities, model=model)
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


    def _flip_keys_values(dictionary):
        return dict((v, k) for k,v in dictionary.items())


class Model(object):

    def __init__(self, voc, embeddings):
        self.vocab=voc
        self.vectors=embeddings
    
    def save(self, output_path):
        with open(output_path, 'wb+') as file:
            pk.dump((self.vocab, self.vectors), file)

    def load(model_path):
        with open(model_path, 'rb') as file:
            vocab, vectors = pk.load(file)
        return Model(vocab, vectors)
