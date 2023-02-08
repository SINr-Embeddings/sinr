import diachrony
from sinr.graph_embeddings import SINr

import networkit as nk
import pickle

path_pickles = "datasets/pickles/"
datasets = [ 'college' ]#, 'stackoverflow', 'eucore', 'stackoverflow', 'enron' ]
extension = ".pkl"

def test_diachrony(factory, sinr_objects,dataset):
    diac = factory(sinr_objects,name=dataset)

if __name__ == "__main__":

    sinr_objects = list()

    for dataset in datasets: 
        with open(path_pickles+dataset+extension, 'rb') as f:
            graphs = pickle.load(f)
            for i,graph in enumerate(graphs):
                sinr_objects.append(SINr.load_from_graph(nk.nxadapter.nx2nk(graph)))
                print(len(graphs),i,len(graph.nodes),len(graph.edges))
            #test_diachrony(diachrony.DiachronicModels.static_factory,sinr_objects[:-1],dataset)
            #test_diachrony(diachrony.DiachronicModels.multilayer_factory,sinr_objects[:-1],dataset)
            test_diachrony(diachrony.DiachronicModels.multislices_factory,sinr_objects[:-1],dataset)
