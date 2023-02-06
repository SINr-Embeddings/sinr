from .logger import logger
import numpy as np
import parallel_sort


def apply_norm(G, norm):
    """
    Apply a normalisation to the graph (IPMI, PMI...) before detecting communities

    :param norm: Type of normalisation.
    :param norm_args: Arguments to pass to the normalisation alfgorithm.
    :return: The graph and the weighted degrees before and after normalisation.
    """
    # try:
    if norm != None:
        G = norm(G)  # , wd_before, wd_after = norm(G)
    # except:
    #   raise TypeError('Unknown norm type')
    return G  # , wd_before, wd_after


def iterative_pmi(G):
    """
    Iteratively applies PMI to the graph to allow for community detection.

    @return:  The graph and the weighted degrees before and after normalisation.
    """
    logger.info("Starting IPMI normalisation.")
    # Using Decorate-Sort-Undecorate as apparently it's more efficient for large lists
    dtype = [('edge', tuple), ('weighted_degree', int)]
    edges_wd = np.array([(e, G.weightedDegree(e[0]) + G.weightedDegree(e[1])) for e in G.iterEdges()], dtype=dtype)
    # edges_wd[::-1].sort(order='weighted_degree') # Sort by weighted degree of edge
    # edges = sorted(list(G.iterEdges()),
    # key=lambda x: G.weightedDegree(x[0]) + G.weightedDegree(x[1]), reverse=True)
    edges = edges_wd['edge']
    edges = edges[parallel_sort.parallel_argsort(edges_wd['weighted_degree'])[::-1]]
    # wd_before = {e: G.weightedDegree(e) for e in G.iterNodes()}
    ipmi = lambda u, v: (G.weight(u, v) / (G.weightedDegree(u) * G.weightedDegree(v)))
    nb_edges = edges.shape[0]
    cnt = 0
    for u, v in edges:
        G.setWeight(u, v,
                    w=ipmi(u, v))
        if cnt % 1000000 == 0:
            logger.info("Edge %i/%i normalized." % (cnt, nb_edges))
        cnt += 1
    # wd_after = {e: G.weightedDegree(e) for e in G.iterNodes()}
    logger.info("Finished IPMI normalisation.")
    return G  # , wd_before, wd_after
