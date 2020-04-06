#!/bin/bash
conda config --set ssl_verify False
conda config --set channel_priority strict
conda install -c conda-forge scikit-image matplotlib scipy pillow jupyter visdom scikit-learn tqdm gensim h5py pip
pip install --upgrade pip
pip install tensorboardX opencv_contrib_python_headless
