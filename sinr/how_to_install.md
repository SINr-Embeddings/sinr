## Development/Editable install for SINr


1. (optional) Install conda : `https://docs.anaconda.com/anaconda/install/`
2. Create a conda environnement: `conda create --name sinr-v2 python==3.9 pip`
3. Install poetry with `pip`: `pip install poetry
4. Activate the environment : `conda activate sinr-v2`
5. Install sinr with `poetry`: `poetry install`
6. Install sinr in development mode with `pip` : `pip install -e .`
7. (optional) Create kernel to be used in`jupyter-notebook`/`jupyter-lab` :`ìpython kernel install --name YOURKERNELNAME --user`
8. (optional) Use `autoreload` in `notebook`/`ipython` :
    ```python
        %load_ext autoreload
        %autoreload 2
    ```

## TL;DR


   ```python
   # With conda install
   conda create --name sinr-v2 python==3.9 pip # create conda environment
   pip install poetry # installpoetry through pip
   conda activate sinr-v2 # activate the environment
   poetry install # install dependencies and package
   pip install -e . # make package installed in development/editable mode
   ```
