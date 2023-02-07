## Development/Editable install for SINr


1. (optional) Install conda : `https://docs.anaconda.com/anaconda/install/`
2. Create a conda environnement: `conda create --name sinr-v2 python==3.9 poetry`
3. Activate the environment : `conda activate sinr-v2`
4. Install sinr with `poetry`: `poetry install`
5. Install sinr in development mode with `pip` : `pip install -e .`
6. (optional) Create kernel to be used in`jupyter-notebook`/`jupyter-lab` :`ìpython kernel install --name YOURKERNELNAME --user`
7. (optional) Use `autoreload` in `notebook`/`ipython` :
    ```python
        %load_ext autoreload
        %autoreload 2
    ```

## TL;DR


   ```python
   # With conda install
   conda create --name sinr-v2 python==3.9 poetry # create conda environment
   conda activate sinr-v2 # activate the environment
   poetry install # install dependencies and package
   pip install -e . # make package installed in development/editable mode
   ```