SINr
==============================

Build word and graph embeddings based on community detection in graphs.




Installation 
------------

1. Install**SINr** in development mode and SpaCy Transformer model for english -> `cd src && python setup.py cythonize && pip install -e . && python -m spacy download en_core_web_trf`
2. Use SINr!


Usage
---------

For examples, see [notebooks](/notebooks)


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


# License


<p><small>Project based on the <a target="_blank" href="https:/


/drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
