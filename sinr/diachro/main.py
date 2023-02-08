import diachrony
from sinr.graph_embeddings import SINr

import networkit as nk
import pickle

path_pickles = "datasets/pickles/"
datasets = [ 'college' ]#, 'stackoverflow', 'eucore', 'stackoverflow', 'enron' ]
extension = ".pkl"

def test_diachrony(factory, sinr_objects,dataset):
    diac = factory(sinr_objects,name=dataset)
    return diac

if __name__ == "__main__":

    sinr_objects = list()

    for dataset in datasets: 
        with open(path_pickles+dataset+extension, 'rb') as f:
            graphs = pickle.load(f)
            for i,graph in enumerate(graphs):
                sinr_objects.append(SINr.load_from_graph(nk.nxadapter.nx2nk(graph)))
                print(len(graphs),i,len(graph.nodes),len(graph.edges))
            #diac = test_diachrony(diachrony.DiachronicModels.static_factory,sinr_objects,dataset)
            #test_diachrony(diachrony.DiachronicModels.multilayer_factory,sinr_objects,dataset)
            diac = test_diachrony(diachrony.DiachronicModels.multislices_factory,sinr_objects,dataset)
            slice = 7
            k_edges = diac.reconstruct_graph(slice ,1000)
            print(len(k_edges))
            test_graph = graphs[slice]
            cpt_ok = 0
            cpt=0
            for edge in k_edges:
                if test_graph.degree[edge[0]+1] > 2 or test_graph.degree[edge[1]+1] > 2:
                    cpt += 1
                    if test_graph.has_edge(edge[0]+1, edge[1]+1):
                        cpt_ok += 1
            print(cpt_ok / cpt, cpt)
