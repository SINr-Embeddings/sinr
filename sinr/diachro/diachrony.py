from sinr.graph_embeddings import SINr, SINrVectors, ModelBuilder, OnlyGraphModelBuilder

class DiachronicModels:
    models = list()

    def __init__(self, models):
        self.models = models

    @classmethod
    def mutilayer_factory(cls, graphs):
        return cls()

    @classmethod
    def static_factory(cls, sinrmodels : list[SINr], gamma=1 : int, name : str):
        """_summary_

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
    def coalesced_factory(cls, sinrmodels : list[SINr], gamma=1 : int, name : str):
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

    @classmethod


    
       '''
        

        Parameters
        ----------
        graphs : list(SINr())
            

        Returns
        -------
        

        '''