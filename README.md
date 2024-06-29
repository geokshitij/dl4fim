# dl4fim
Use deep learning to predict the floodplain areas to help in hydraulic designs.

## How to set up a Python environment

```bash
# download the miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh

# install miniconda
./Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/usr/local/opt/miniconda3
~/usr/local/opt/miniconda3/bin/conda init
. ~/.bashrc

# config to use the conda-forge channel by default
conda config --add channels conda-forge
conda config --set channel_priority strict

# create a new environment
conda create -n tf
conda activate tf

# install conda packages
conda install tensorflow-gpu jupyterlab geopandas geotile fiona imgaug
```
