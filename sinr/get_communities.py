from sinr.text.cooccurrence import Cooccurrence
from sinr.text.pmi import pmi_filter
from sinr.text.preprocess import extract_text
import sinr.graph_embeddings as ge
from os.path import isfile

import sys

if __name__=="__main__":

    if (len(sys.argv)!=2):
        print(f"Usage: {sys.argv[0]} path_to_corpus")
        exit(1)

    # Generating corpus name and path to matrix
    corpus = sys.argv[1]
    path_to_corpus = "/".join(corpus.split("/")[:-1])+"/"
    corpus_name = (corpus.split("/")[-1]).split(".")[0]
    path_to_matrix = path_to_corpus+corpus_name+"_matrix.pk"
    
    c = Cooccurrence()
    # Creating the cooccurrence matrix if not existing
    if not isfile(path_to_matrix):
        sentences = extract_text(corpus)
        c.fit(sentences, window=100)
        c.matrix = pmi_filter(c.matrix)
        c.save(path_to_matrix)

    sinr = ge.SINr.load_from_cooc_pkl(path_to_matrix)
    communities = sinr.detect_communities(gamma=300)
