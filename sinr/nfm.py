import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import normalize
import networkit as nk

from .logger import logger


def get_nfm_embeddings(G, vector, compute_np=False, merge=False):
    """Compute the Node F-Measure metrics to build the embedding matrix using the graph and community structure detected.

    :param G: Graph on which to compute the embeddings
    :type G: networkit.Graph
    :param vector: The node-community membership vector
    :type vector: list[int]
    :param compute_np: Compute the node predominance metric, defaults to False
    :type compute_np: bool, optional
    :param merge: Merge the NR and NP measure in a common matrix, defaults to False
    :type merge: bool, optional
    :returns: The node predominance, node recall and merged matrix (nfm) if applicable.
    :rtype: tuple[Scipy.sparse.csr_matrix, Scipy.sparse.csr_matrix, Scipy.sparse.csr_matrix]

    """
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
    """Compute the node-recall based on the adjacency matrix and the community-membership matrix of the graph.

    :param adjacency: Adjacency matrix of the graph.
    :type adjacency: Scipy.sparse.csr_matrix
    :param membership_matrix: Community membership matrix.
    :type membership_matrix: Scipy.sparse.csr_matrix
    :returns: NR measures for each node and each community
    :rtype: Scipy.sparse.csr_matrix

    """
    norm_adjacency = distributed_degree(adjacency)  # Make rows of matrix sum at 1
    return norm_adjacency.dot(membership_matrix)


def distributed_degree(adjacency):
    """Make values in the adjacency matrix be between 0 and 1 depending on how the degree of the node is distributed over each community.

    :param adjacency: Adjacency matrix of the graph.
    :type adjacency: Scipy.sparse.csr_matrix
    :returns: l1 normalized adjacency matrix.
    :rtype: Scipy.sparse.csr_matrix

    """
    return normalize(adjacency, "l1")


def compute_NP(adjacency, membership_matrix):
    """Compute the node-predominance based on the adjacency matrix and the community-membership matrix of the graph.

    :param adjacency: Adjacency matrix of the graph.
    :type adjacency: Scipy.sparse.csr_matrix
    :param membership_matrix: Community membership matrix.
    :type membership_matrix: Scipy.sparse.csr_matrix
    :returns: NP measures for each node and each community
    :rtype: Scipy.sparse.csr_matrix

    """
    community_weights = get_community_weights(adjacency, membership_matrix)  # Weighted degree of each community
    weighted_membership = membership_matrix.multiply(
        np.reciprocal(community_weights).astype('float'))  # 1/community_weight for each column of the membership matrix
    return adjacency.dot(weighted_membership)


def get_community_weights(adjacency, membership_matrix):
    """Get the total weight of each community in terms of degree.

    :param adjacency: Adjacency matrix of the graph.
    :type adjacency: Scipy.sparse.csr_matrix
    :param membership_matrix: Community membership matrix.
    :type membership_matrix: Scipy.sparse.csr_matrix
    :returns: Degree-based weight of each community.
    :rtype: Scipy.sparse.csr_matrix

    """
    return adjacency.dot(membership_matrix).sum(axis=0)


def get_membership(vector):
    """Return the membership matrix based on the community membership vector.

    :param vector: The vector of community index for each node
    :type vector: list[int]
    :returns: The community membership matrix of shape (#nodes x #communities).
    :rtype: Scipy.sparse.csr_matrix

    """
    nb_nodes = len(vector)
    nb_communities = len(set(vector))
    rows = range(nb_nodes)
    columns = vector
    data = np.ones(nb_nodes)
    return coo_matrix((data, (rows, columns)), shape=(nb_nodes, nb_communities)).tocsr()
