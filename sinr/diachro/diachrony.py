from sinr.graph_embeddings import SINr, SINrVectors, ModelBuilder, OnlyGraphModelBuilder
import networkit as nk
from networkit import Partition
import leidenalg as la
import igraph as ig
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def to_igraph(G):
    graph_igraph = ig.Graph()
    graph_igraph.add_vertices(len(list(G.iterNodes())))
    graph_igraph.add_edges(G.iterEdges(), attributes={"weight":[weight for _, _, weight in G.iterEdgesWeights()]})
    return graph_igraph

def to_igraphs(sinrmodels:list[SINr]):
    graphs_igraph = list()        
    for model in sinrmodels:
        graph_igraph = to_igraph(model.get_cooc_graph())
        graphs_igraph.append(graph_igraph)
    return graphs_igraph


class DiachronicModels:
    models: list[SINrVectors] = list()

    def __init__(self, models):
        self.models = models

    @classmethod
    def multilayer_factory(cls, sinrmodels:list[SINr], name: str, gamma: int =1):
        """_summary_

        Args:
            sinrmodels (list[SINr]): SINrVectors initialized with a graph for each time slice
            name (str): name of the dataset

        Returns:
            _type_: an instance of Diachronic models, acts as a factory
        """
        graphs_igraph = to_igraphs(sinrmodels)
        for graph in graphs_igraph:
            ig.summary(graph)
        membership, improvement = la.find_partition_multiplex(graphs_igraph, la.ModularityVertexPartition)
        models = list()
        for idx, model in enumerate(sinrmodels):
            model.extract_embeddings(membership)
            sinrvectors = OnlyGraphModelBuilder(model, name+"_"+str(idx))
            models.append(sinrvectors)
        return cls(models)

    @classmethod
    def multislices_factory(cls, sinrmodels:list[SINr], name: str, gamma: int =1):
        """_summary_

        Args:
            sinrmodels (list[SINr]): SINrVectors initialized with a graph for each time slice
            name (str): name of the dataset

        Returns:
            _type_: an instance of Diachronic models, acts as a factory
        """
        graphs_igraph = to_igraphs(sinrmodels)
        for g in graphs_igraph:
            g.vs["id"] = [node.index for node in g.vs]

        memberships, improvement = la.find_partition_temporal(graphs_igraph, la.ModularityVertexPartition)
        models = list()
        for idx, zip_info in enumerate(zip(sinrmodels, memberships)):
            model = zip_info[0]
            membership = zip_info[1]
            model.extract_embeddings(membership)
            sinrvectors = OnlyGraphModelBuilder(model, name+"_"+str(idx))
            models.append(sinrvectors)
        return cls(models)

    @classmethod
    def static_factory(cls, sinrmodels : list[SINr], name : str, gamma: int =1 ):
        """ One Sinr model per slice, independent from each other

        Args:
            sinrmodels (list[SINr]): _description_
            name (str): _description_
            gamma (int, optional): _description_. Defaults to 1:int.

        Returns:
            DiachronicModels: An instance of DiachronicModels with a model for each time slice.
        """
        

        static_models = []
        for idx, model in enumerate(sinrmodels):
            model.detect_communities(gamma=gamma) # One partition per slice
            model.extract_embeddings() # One extraction according to the partition of the slice
            static_models.append(OnlyGraphModelBuilder(model, f"{name}_{idx}"))
        return cls(static_models)

    @classmethod
    def coalesced_factory(cls, sinrmodels : list[SINr], name : str, gamma : int = 1):
        """_summary_

        Args:
            sinrmodels (list[SINr]): List of SINr models each representing one time slice.
            name (str): Name of the model/dataset processed.
            gamma (int, optional): Resolution parameter for community detection. Defaults to 1:int.

        Returns:
            DiachronicModels: An instance of DiachronicModels with a model for each time slice.
        """
        coalesced_models = []
        isWeighted = all([model.get_cooc_graph().isWeighted() for model in sinrmodels])
        common_nodes = set([i for model in sinrmodels for i in model.get_cooc_graph().iterNodes()]) # Get the set of common edges accross all time slices.
        if isWeighted:
            common_edges = set([(u,v,w) for model in sinrmodels for u, v, w in model.iterEdgesWeights() if u in common_edges and v in common_edges]) # Retrieve common edges accross all time steps.
            graph_common = nk.Graph(weighted=True, directed=False)
            for u, v, w in common_edges:
                graph_common.addEdge(u, v, w=w, addMissing=True, checkMultiEdge=True) # Reconstruct a graph restricted to the common edges
        else:
            common_edges = set([(u,v) for model in sinrmodels for u, v in model.iterEdges() if u in common_edges and v in common_edges]) # Retrieve common edges accross all time steps.
            graph_common = nk.Graph(weighted=False, directed=False)
            for u, v in common_edges:
                graph_common.addEdge(u, v, addMissing=True, checkMultiEdge=True) # Reconstruct a graph restricted to the common edges
        common_sinr = SINr.load_from_graph(graph_common)
        common_sinr.detect_communities()
        partition_common = common_sinr.get_communities()
        for idx, model in enumerate(sinrmodels):
            model_graph = model.get_cooc_graph()
            node_subset = list(set(model.iterNodes()) & common_nodes) # Get the intersection of nodes between graph of slice and coalesced graph
            model_common = SINr.load_from_graph(nk.graphtools.GraphTools.subgraphFromNodes(model_graph, node_subset))
            model_common.extract_embeddings(communities=partition_common)
            coalesced_models.append(OnlyGraphModelBuilder(model, f"{name}_{idx}"))
        return cls(coalesced_models)

    # To amend for text
    # @classmethod
    # def coalesced_global_factory(cls, sinrmodels : list[SINr], name : str, gamma : int = 1):
    #     coalesced_models = []
    #     all_nodes = set([node for model in sinrmodels for node in model.get_cooc_graph().iterEdges()]) # Get set of edges accross all temporal slices
    #     isWeighted = all([model.get_cooc_graph().isWeighted() for model in sinrmodels])
    #     if isWeighted:
    #         all_edges = set([(u,v,w) for model in sinrmodels for u, v, w in model.iterEdgesWeights()])
    #         graph_global = nk.Graph(weighted=True, directed=False)
    #         for u, v, w in all_edges:
    #             graph_global.addEdge(u, v, w=w, addMissing=True, checkMultiEdge=True) # Reconstruct a graph restricted to the common edges
    #     else:
    #        all_edges = set([(u,v) for model in sinrmodels for u, v in model.iterEdges()])
    #        for u, v in all_edges:
    #             graph_global.addEdge(u, v, addMissing=True, checkMultiEdge=True) # Reconstruct a graph restricted to the common edges
    #     global_model = SINr.load_from_graph(graph_global)
    #     global_model.detect_communities()
    #     partition = global_model.get_communities()
    #     for idx, model in enumerate(sinrmodels):
    #         model_graph = model.get_cooc_graph()
    #         missing_nodes = all_nodes - set(model_graph.iterNodes())
    #         partition_model = nk.Partition(partition.get)

            


            
    

    def get_model(self, slice:int) -> SINrVectors:
        """Getting Sinr model for a specific slice

        Args:
            slice (int): slice number

        Returns:
            SINrVectors: A SinrVectors object for a specific slice
        """
        return self.models[slice]

    def get_vector(self, slice:int, node:int):
        """Getting vector of a node for a specific slice

        Args:
            slice (int): slice number
            node (int): node id

        Returns:
            _type_: a csr matrix
        """
        model = self.models[slice]
        return model.get_my_vector(node)
    
    def get_similarity_matrix(self, slice:int):
        """ Getting similarity matrix of vectors of a specific slice

        Args:
            slice (int): slice number

        Returns:
            _type_: matrix of float
        """
        model = self.models[slice]
        return cosine_similarity(model.vectors)
    
    def get_k_highest(self, slice:int, k:int):
        """Getting the k (row, column) pairs with the highest similarities

        Args:
            slice (int): slice number
            k (int): k

        Returns:
            _type_: dict[(row, column)] = similarity
        """
        similarity = self.get_similarity_matrix(slice=slice)
        voc = 10
        limit = - (voc + 2 * k)
        rows = np.argpartition(similarity.flatten(), limit)[limit:] // voc
        columns = np.argpartition(similarity.flatten(), limit)[limit:] % voc
        k_highest = dict()
        for r,c in zip(rows, columns):
            if r != c:
                if (c,r) not in k_highest:
                    k_highest[(r,c)] = similarity[r,c]
        return k_highest

    def reconstruct_graph(self, slice:int, k:int):
        """Ordered edges : the most probable according to our embeddings
        """
        dico = self.get_k_highest(slice, k)
        ordered_k_edges = [k for k, v in sorted(x.items(), key=lambda item: item[1])]
        return ordered_k_edges

