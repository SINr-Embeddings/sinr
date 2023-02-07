from sinr.graph_embeddings import SINr, SINrVectors, ModelBuilder, OnlyGraphModelBuilder
import leidenalg as la
import igraph as ig


def to_igraph(G):
    graph_igraph = ig.Graph()
    graph_igraph.add_vertices(len(list(G.iterNodes())))
    graph_igraph.add_edges(G.iterEdges(), attributes={"weight":[weight for _, _, weight in G.iterEdgesWeights()]})
    return graph_igraph

class DiachronicModels:
    models: list[SINrVectors] = list()

    def __init__(self, models):
        self.models = models

    @classmethod
    def mutilayer_factory(cls, sinrmodels:list[SINr], name: str, gamma=1):
        """_summary_

        Args:
            sinrmodels (list[SINr]): SINrVectors initialized with a graph for each time slice
            name (str): name of the dataset
            gamma (int, optional): resolution parameter. Defaults to 1.

        Returns:
            _type_: an instance of Diachronic models, acts as a factory
        """
        graphs_igraph = list()        
        for model in sinrmodels:
            graph_igraph = to_igraph(model.get_cooc_graph())
            graphs_igraph.append(graph_igraph)
        membership, improvement = la.find_partition(graphs_igraph, la.ModularityVertexPartition, gamma=gamma)
        models = list()
        for idx, model in enumerate(sinrmodels):
            model.extract_embeddings(membership)
            sinrvectors = OnlyGraphModelBuilder(model, name+"_"+str(idx))
            models.append(sinrvectors)
        return cls(models)

    @classmethod
    def static_factory(cls, graphs):
        return cls()
