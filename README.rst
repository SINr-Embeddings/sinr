=====
SINr
=====
|languages| |downloads| |license| |version| |cpython| |wheel| |python| |docs| |activity| |contributors| |quality| |build|

*SINr* is an open-source tool to efficiently compute graph and word
embeddings. Its aim is to provide sparse interpretable vectors from a
graph structure. The dimensions of the vector produced are related to
the community structure detected in the graph. By leveraging the
relative connection of vertices to communities, *SINr* builds an
interpretable space. *SINr* is focused on providing tools to build and
interpret the embeddings produced.

*SINr* is a Python module relying on
`Networkit <https://networkit.github.io>`__ for the graph structure and
community detection. *SINr* also provides efficient implementations to
extract word co-occurrence graphs from large text corpora. One of the
strength of *SINr* is its ability to work with text and produce
interpretable word embeddings that are competitive with similar
approaches. For more details on the performances of *SINr* on downstream
evaluation tasks, please refer to the `Publications <#publications>`__
section.

Requirements
============

-  As SINr relies on libraries implemented using C/C++, a modern C++
   compiler is required.
-  OpenMP (required for `Networkit <https://networkit.github.io>`__ and
   compiling *SINr*\ ’s Cython)
-  Python 3.9
-  Pip
-  Cython
-  Conda (recommended)

Install
=======

SINr can be installed through ``pip``.

pip
---

.. code:: bash

   conda activate sinr # activate conda environment
   pip install sinr

Usage example
=============

To get started using *SINr* to build graph and word embeddings, have a
look at the `notebook <./notebooks>`__ directory.

Here is a minimum working example of *SINr*

.. code:: python

       import nltk # For textual resources

       import sinr.text.preprocess as ppcs
       from sinr.text.cooccurrence import Cooccurrence
       from sinr.text.pmi import pmi_filter
       import sinr.graph_embeddings as ge
       import sinr.text.evaluate as ev

       # Get a textual corpus
       # For example, texts from the Project Gutenberg electronic text archive,
       # hosted at http://www.gutenberg.org/
       nltk.download('gutenberg')
       gutenberg = nltk.corpus.gutenberg # contains 25,000 free electronic books
       file = open("my_corpus.txt", "w")
       file.write(gutenberg.raw())
       file.close()

       # Preprocess corpus
       vrt_maker = ppcs.VRTMaker(ppcs.Corpus(ppcs.Corpus.REGISTER_WEB,
                                             ppcs.Corpus.LANGUAGE_EN,
                                             "my_corpus.txt"),
                                             ".", n_jobs=8)
       vrt_maker.do_txt_to_vrt()
       sentences = ppcs.extract_text("my_corpus.vrt", min_freq=20)

       # Construct cooccurrence matrix
       c = Cooccurrence()
       c.fit(sentences, window=5)
       c.matrix = pmi_filter(c.matrix)
       c.save("my_cooc_matrix.pk")

       # Train SINr embedding
       model = ge.SINr.load_from_cooc_pkl("my_cooc_matrix.pk")
       commu = model.detect_communities(gamma=10)
       model.extract_embeddings(commu)

       # Construct SINrVectors to manipulate the model
       sinr_vec = ge.InterpretableWordsModelBuilder(model,
                                                    'my_sinr_vectors',
                                                    n_jobs=8,
                                                    n_neighbors=25).build()
       sinr_vec.save()

       # Sparsify vectors for better interpretability and performances
       sinr_vec.sparsify(100)

       # Evaluate the model with the similarity task
       print('\nResults of the similarity evaluation :')
       print(ev.similarity_MEN_WS353_SCWS(sinr_vec))

       # Explore word vectors and dimensions of the model
       print("\nDimensions activated by the word 'apple' :")
       print(sinr_vec.get_obj_stereotypes('apple', topk_dim=5, topk_val=3))

       print("\nWords similar to 'apple' :")
       print(sinr_vec.most_similar('apple'))

       # Load an existing SinrVectors object
       sinr_vec = ge.SINrVectors('my_sinr_vectors')
       sinr_vec.load()
