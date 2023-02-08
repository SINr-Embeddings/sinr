from sinr.graph_embeddings import SINr, SINrVectors, ModelBuilder, OnlyGraphModelBuilder
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
    def mutilayer_factory(cls, sinrmodels:list[SINr], name: str, gamma: int =1):
        """ Second strategy : modularity gain is computing on each layer and averaged to decide whether or not to move

        Args:
            sinrmodels (list[SINr]): SINrVectors initialized with a graph for each time slice
            name (str): name of the dataset
            gamma (int, optional): resolution parameter. Defaults to 1.

        Returns:
            _type_: an instance of Diachronic models, acts as a factory
        """
        graphs_igraph = to_igraphs(sinrmodels)
        membership, improvement = la.find_partition_multiplex(graphs_igraph, la.ModularityVertexPartition, gamma=gamma)
        models = list()
        for idx, model in enumerate(sinrmodels):
            model.extract_embeddings(membership)
            sinrvectors = OnlyGraphModelBuilder(model, name+"_"+str(idx))
            models.append(sinrvectors)
        return cls(models)

    @classmethod
    def mutislices_factory(cls, sinrmodels:list[SINr], name: str, gamma: int =1):
        """ Fourth strategy : multi slices, optimization is made on each slice, taking into account the other ones

        Args:
            sinrmodels (list[SINr]): SINrVectors initialized with a graph for each time slice
            name (str): name of the dataset
            gamma (int, optional): resolution parameter. Defaults to 1.

        Returns:
            _type_: an instance of Diachronic models, acts as a factory
        """
        graphs_igraph = to_igraphs(sinrmodels)
        for g in graphs_igraph:
            g["id"] = g.index

        memberships, improvement = la.find_partition(graphs_igraph, la.ModularityVertexPartition, gamma=gamma)
        models = list()
        for idx, model, membership in enumerate(zip(sinrmodels, memberships)):
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
            DiachronicModels: _description_
        """
        

        static_models = []
        for idx, model in enumerate(sinrmodels):
            model.detect_communities(gamma=gamma) # One partition per slice
            model.extract_embeddings() # One extraction according to the partition of the slice
            static_models.append(OnlyGraphModelBuilder(model, f"{name}_{idx}"))
        return cls(static_models)

    @classmethod
    def coalesced_factory(cls, sinrmodels : list[SINr], name : str, gamma: int =1 ):
        """_summary_

        Args:
            sinrmodels (list[SINr]): _description_
            name (str): _description_
            gamma (_type_, optional): _description_. Defaults to 1:int.

        Returns:
            _type_: _description_
        """
        common_edges = set([i for model in sinrmodels for i in model.get_cooc_graph.iterNodes()]) # Get the set of common edges accross all time slices.

        return cls()

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

