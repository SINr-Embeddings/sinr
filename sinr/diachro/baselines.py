from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



class BaselineVectors:

    def __init__(self, models):
        """_summary_

        Args:
            models (_type_): A matrix with rows as vectors representing nodes
        """
        self.models = models
    

    def get_model(self, slice:int):
        """Getting model (matrix) for a specific slice

        Args:
            slice (int): slice number

        Returns:
            A matrix object for a specific slice
        """
        return self.models[slice]

    def get_vector(self, slice:int, node:int):
        """Getting vector of a node for a specific slice

        Args:
            slice (int): slice number
            node (int): node id

        Returns:
            _type_: a vector
        """
        model = self.get_model(slice)
        return model[node]
    
    def get_similarity_matrix(self, slice:int):
        """ Getting similarity matrix of vectors of a specific slice

        Args:
            slice (int): slice number

        Returns:
            _type_: matrix of float
        """
        model = self.models[slice]
        matrix = np.triu(cosine_similarity(model.vectors))
        np.fill_diagonal(matrix, 0)
        return matrix
    
    def get_k_highest(self, slice:int, k:int):
        """Getting the k (row, column) pairs with the highest similarities

        Args:
            slice (int): slice number
            k (int): k

        Returns:
            _type_: dict[(row, column)] = similarity
        """
        similarity = self.get_similarity_matrix(slice=slice)
        voc = self.models[slice].vectors.shape[0]
        print(voc)
        limit = - k
        print(limit)
        rows = np.argpartition(similarity.flatten(), limit)[limit:] // voc
        columns = np.argpartition(similarity.flatten(), limit)[limit:] % voc
        print(rows.shape)
        print(columns.shape)
        k_highest = dict()
        for r,c in zip(rows, columns):
            k_highest[(r,c)] = similarity[r,c]
        return k_highest

    def reconstruct_graph(self, slice:int, k:int):
        """Ordered edges : the most probable according to our embeddings
        """
        dico = self.get_k_highest(slice, k)
        ordered_k_edges = [k for k, v in sorted(dico.items(), key=lambda item: item[1])]
        return ordered_k_edges

