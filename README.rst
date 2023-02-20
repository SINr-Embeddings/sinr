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
   compiling *SINr*\ ’s Cython
-  Python 3.9
-  Pip
-  Cython
-  Conda (recommended)

Install
=======

SINr can be installed through ``pip`` or from source using ``poetry``
directives.

pip
---

.. code:: bash

   conda activate sinr # activate conda environment
   pip install sinr

from source
-----------

.. code:: bash

   conda activate sinr # activate conda environment
   git clone git@github.com:SINr-Embeddings/sinr.git
   cd sinr
   pip install poetry # poetry solves dependencies and installs SINr
   poetry install # installs SINr based on the pyproject.toml file

Usage example
=============

To get started using *SINr* to build graph and word embeddings, have a
look at the `notebook <./notebooks>`__ directory.

Here is a minimum working example of *SINr*

.. code:: python

       import urllib
       import io
       import gzip
       import networkit as nk
       import sinr.graph_embeddings as ge


       url = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
       graph_file = "wikipedia-votes.txt"
       # Read a graph from SNAP
       sock = urllib.request.urlopen(url)  # open URL
       s = io.BytesIO(sock.read())  # read into BytesIO "file"
       sock.close()
       with gzip.open(s, "rt") as f_in:
           with open(graph_file, "wt") as f_out:
               f_out.writelines(f_in.readlines())
       # Initialize a networkit.Graph object from SNAP graph
       G = nk.readGraph(graph_file, nk.Format.SNAP)

       # Build a SINr model and extract embeddings
       model = ge.SINr.load_from_graph(G)
       model.run(algo=nk.community.PLM(G))
       embeddings = model.get_nr()
       print(embeddings)

Documentation
=============

The documentation for *SINr* is `available
online <https://sinr-embeddings.github.io/sinr/index.html>`__.

Contributing
============

Pull requests are welcome. For major changes, please open an issue first
to disccus the changes to be made.

License
=======

Released under `CeCILL 2.1 <https://cecill.info/>`__, see `LICENSE <./LICENSE>`__ for more details.

Publications
============

*SINr* is currently maintained at the *University of Le Mans*. If you
find *SINr* useful for your own research, please cite the appropriate
papers from the list below. Publications can also be found on
`publications page in the
documentation <https://sinr-embeddings.github.io/sinr/_build/html/publications.html>`__.

**Initial SINr paper, 2021**

-  Thibault Prouteau, Victor Connes, Nicolas Dugué, Anthony Perez,
   Jean-Charles Lamirel, et al.. SINr: Fast Computing of Sparse
   Interpretable Node Representations is not a Sin!. Advances in
   Intelligent Data Analysis XIX, 19th International Symposium on
   Intelligent Data Analysis, IDA 2021, Apr 2021, Porto, Portugal.
   pp.325-337,
   ⟨\ `10.1007/978-3-030-74251-5_26 <https://dx.doi.org/10.1007/978-3-030-74251-5_26>`__\ ⟩.
   `⟨hal-03197434⟩ <https://hal.science/hal-03197434>`__

**Interpretability of SINr embedding**

-  Thibault Prouteau, Nicolas Dugué, Nathalie Camelin, Sylvain Meignier.
   Are Embedding Spaces Interpretable? Results of an Intrusion Detection
   Evaluation on a Large French Corpus. LREC 2022, Jun 2022, Marseille,
   France. `⟨hal-03770444⟩ <https://hal.science/hal-03770444>`__
   
   
.. |languages| image:: https://img.shields.io/github/languages/count/SINr-Embeddings/sinr
.. |downloads| image:: https://img.shields.io/pypi/dm/sinr
.. |license| image:: https://img.shields.io/pypi/l/sinr?color=green
.. |version| image:: https://img.shields.io/pypi/v/sinr
.. |cpython| image:: https://img.shields.io/pypi/implementation/sinr
.. |wheel| image:: https://img.shields.io/pypi/wheel/sinr
.. |python| image:: https://img.shields.io/pypi/pyversions/sinr
.. |docs| image:: https://img.shields.io/website?url=https%3A%2F%2Fsinr-embeddings.github.io%2Fsinr%2F_build%2Fhtml%2Findex.html
.. |activity| image:: https://img.shields.io/github/commit-activity/y/SINr-Embeddings/sinr
.. |contributors| image:: https://img.shields.io/github/contributors/SINr-Embeddings/sinr
.. |quality| image:: https://scrutinizer-ci.com/g/SINr-Embeddings/sinr/badges/quality-score.png?b=main
.. |build| image:: https://scrutinizer-ci.com/g/SINr-Embeddings/sinr/badges/build.png?b=main
