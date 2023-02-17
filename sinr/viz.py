from sinr.graph_embeddings import SINrVectors
import matplotlib.pyplot as plt
import numpy as np


class SINrViz:
    """Visualization package for SINr embdeddings. The goal is to visualize and interpret
    the diemnesions of the embeddings produced."""
    def __init__(self, sinr_vectors: SINrVectors):
        self.sinr_vectors = sinr_vectors

    def compare_stereotypes(self, args, topk_dim=5):
        """Make a heatmap comparing top dimensions for elements in args (words).

        :param args: A list of indices (words).
        :type args: list[int]
        :param topk_dim: Number of top dimensions to fetch, defaults to 5
        :type topk_dim: int, optional

        """
        if len(args) == 0:
            print("No objects passed as parameters")
            return
        else:
            args = args
            dims = set()
            data = []
            for idx in args:
                int_dims = self.sinr_vectors.get_topk_dims(idx, topk=topk_dim)
                dims = dims.union(set(int_dims))
            print(dims)
            cpt = 0
            stereotypes = []
            for dim in dims:
                stereotypes.append(self.sinr_vectors.get_dimension_stereotypes_idx(dim, topk=1).get_interpreter(0))
            for idx in args:
                data.append([])
                for dim in dims:
                    data[cpt].append(self.sinr_vectors.get_value_obj_dim(idx, dim))
                cpt += 1

            fig, ax = plt.subplots()
            im = ax.imshow(data)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Strength of relatedness", rotation=-90, va="bottom")
            ax.set_xticks(np.arange(len(stereotypes)), labels=stereotypes)
            ax.set_yticks(np.arange(len(args)), labels=args)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            plt.show()
