import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import normalize
import networkit as nk

from .logger import logger


def get_nfm_embeddings(G, vector, compute_np=False, merge=False):
    logger.info("Starting NFM")
    adjacency = nk.algebraic.adjacencyMatrix(G, matrixType='sparse')  # Extract the adjacency matrix of the graph
    membership_matrix = get_membership(vector)  # Word-Community membership matrix

    node_recall = compute_NR(adjacency, membership_matrix)

    node_pred = None
    nfm = None

    if compute_np or merge:
        node_pred = compute_NP(adjacency, membership_matrix)
        if merge:
            nfm = hstack([node_pred, node_recall])

    return node_pred, node_recall, nfm


def compute_NR(adjacency, membership_matrix):
    norm_adjacency = distributed_degree(adjacency)  # Make rows of matrix sum at 1
    return norm_adjacency.dot(membership_matrix)


def distributed_degree(adjacency):
    return normalize(adjacency, "l1")


def compute_NP(adjacency, membership_matrix):  # , community_weights):
    # weighted_membership = membership_matrix.multiply(np.reciprocal(community_weights.astype('float'))) # 1/community_weight for each column of the membership matrix
    community_weights = get_community_weights(adjacency, membership_matrix)  # Weighted degree of each community
    weighted_membership = membership_matrix.multiply(
        np.reciprocal(community_weights).astype('float'))  # 1/community_weight for each column of the membership matrix
    return adjacency.dot(weighted_membership)


def get_community_weights(adjacency, membership_matrix):
    return adjacency.dot(membership_matrix).sum(axis=0)


def get_membership(vector):
    nb_nodes = len(vector)
    nb_communities = len(set(vector))
    rows = range(nb_nodes)
    columns = vector
    data = np.ones(nb_nodes)
    return coo_matrix((data, (rows, columns)), shape=(nb_nodes, nb_communities)).tocsr()
