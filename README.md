SINr
==============================

Build word embeddings based on community detection in graphs.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Background
----------



Features
----------

SINr is composed of two main modules :

* Cooccurrence : a cython based module to efficiently compute a cooccurrence matrix from a given corpus
* SINr : a module to compute cooccurence network based, sparse word embeddings


Installation 
------------

1. Launch a job on a slurm node -> `srun -p gpu --gres "gpu:1" --time 1-0 --mem 5G --pty bash`
2. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
3. Clone repository -> `git clone --branch nfm_sparse https://git-lium.univ-lemans.fr/tprouteau/sinr.git && cd sinr`
4. Build conda environment -> `conda env create -f environment.yml`
5. Activate environment -> `conda activate sinr_release`
6. Install**SINr** in development mode and SpaCy Transformer model for english -> `cd src && python setup.py cythonize && pip install -e . && python -m spacy download en_core_web_trf`
7. Use SINr!


Launch a Jupyter Notebook in jupyterlab
---------------------------------------

1. Activate your conda environment -> `conda activate sinr_release`
2. (upon first launch) install environment kernel in IPython -> `ipython kernel install --name sinr_release --user`
3. Launch a notebook on a node -> `srun -p gpu --gres "gpu:1" --mem 80G -c15 -w "gpu15" jlaunch jupyter-lab` #Use the -w option to choose the node one should not use a K20/K40 GPU as is it not supported by cupy anymore.
4. ctrl+click on the link displayed on the terminal and select the adequate kernel (_sinr\_release_)



Usage
---------

For additional examples see [notebooks](/notebooks)

### Cooccurence

```python
from sinr.cooccurrence import Cooccurrence
from sinr.pmi import pmi_filter

# Load your corpus as list of lists of tokens
sentences = [["sinr", "is", "fun"], ["sinr", "is", "a", "python", "package"]]
# Build cooccurrence matrix
c = Cooccurrence()
c.fit(sentences, window=2)

#Normalise cooccurrence matrix using PPMI
c.matrix = pmi_filter(c.matrix)
c.save("/path_to_output/matrix.pk")

```

### SINr

The extraction of the embedding is currently greedy in terms of memory. When working with large corpora, do not hesitate to ask for rather large amounts of RAM (>100G)... This is currently being fixed.


```python
from sinr.graph_embeddings import SINr

model = SINr.sinr("/path_to_output/matrix.pickle", output_path="path_to_output", n_jobs=4)  
#If an output_path is supplied, the model will be saved -- Embeddings are returned
#as a Model object comprised of a dictionnary for the vocabulary and a scipy.sparce.csr_matrix for the vectors
```

Contributing
------------

Pull requests are welcome. For major changes, please open an issue first to disccuss the changes to be made.

## Compile/Install from source

In order to compile and install SINr from source follow the procedure described below

```bash
git clone --branch nfm_sparse https://git-lium.univ-lemans.fr/tprouteau/sinr.git
cd sinr
conda env create -f environment.yml
conda activate sinr_release
python setup.py cythonize
pip install -e .
```


## Evaluate Word Embeddings
In order to evaluate the word embeddings on the similarity task you may use the library **Word Embedding Benchmarks** developped by Stanislaw Jastrzebski : https://github.com/kudkudak/word-embeddings-benchmarks

⚠️ The embeddings returned by the model are of type `Scipy.sparse.csr_matrix` you will need to pass them as a dense matrix using the function

```python
matrix = my_sparse_csr_matrix.todense()
```
Refer to the [documentation and examples](https://github.com/kudkudak/word-embeddings-benchmarks/tree/master/examples) to know which format to use in input of the benchmarking library.

# License


<p><small>Project based on the <a target="_blank" href="https:/


/drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
