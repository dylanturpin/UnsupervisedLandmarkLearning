#!/bin/bash
conda config --set ssl_verify False
conda config --set channel_priority strict
conda install -c conda-forge scikit-image matplotlib scipy pillow jupyter visdom scikit-learn tqdm gensim h5py pip
pip install --upgrade pip
pip install torch==1.4.0 tensorboardX opencv_contrib_python_headless
mkdir /pkgs
cd /pkgs
git clone https://github.com/dylanturpin/hydra
cd /pkgs/hydra
pip install -e .
