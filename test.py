import networkx as nx
import scipy, pickle
import sinr.graph_embeddings as ge

import directedlouvain as dl
import networkit

import itertools

# reading the graph from the file computed by directed_louvain.by
G = nx.read_weighted_edgelist("graph_text.txt", nodetype=int, create_using=nx.DiGraph)

# creating the adjacency matrix
matrix = scipy.sparse._coo.coo_matrix(nx.to_scipy_sparse_array(G))

# loading the dict node<->word computed by directed_louvain.by
with open("dict.pk","rb") as f:
    dico = pickle.load(f)

# creating the SINr object from matrix and dico
sinr = ge.SINr.load_from_adjacency_matrix(matrix, dico)
#dico = {value: key for key,value in dico.items()}

# computing communities using Directed Louvain 
dl_obj = dl.Community("graph_text.txt", weighted=True, gamma=55)
dl_obj.run(verbose=False)
communities = dl_obj.last_level()

# creating the Partition from networkit
partition = networkit.Partition(len(G.nodes))

for node, community in communities.items():
    partition.addToSubset(community,node)

# computing embeddings 
sinr.extract_embeddings(partition)
sinr_vectors = ge.ModelBuilder(sinr, "harry", n_jobs=8, n_neighbors=5).with_embeddings_nr().with_vocabulary().with_communities().build()

print(sinr_vectors.most_similar("harry"))
