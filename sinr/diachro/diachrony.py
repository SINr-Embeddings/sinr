from sinr.graph_embeddings import SINr, SINrVectors, ModelBuilder, OnlyGraphModelBuilder
import networkit as nk
from networkit import Partition

class DiachronicModels:
    models = list()

    def __init__(self, models):
        self.models = models

    @classmethod
    def mutilayer_factory(cls, graphs):
        return cls()

    @classmethod
    def static_factory(cls, sinrmodels : list[SINr],  name : str, gamma : int= 1):
        """_summary_

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
    def coalesced_common_factory(cls, sinrmodels : list[SINr], name : str, gamma : int = 1):
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

            


            
    



        
            

