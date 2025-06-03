#!/bin/bash
set -e

echo "Installing PyTorch 2.7.0 + CUDA 11.8..."
python -m pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118

echo "Installing torch-scatter with PyG wheels..."
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu118.html

echo "Installing imagecorruptions from GitHub..."
python -m pip install git+https://github.com/bethgelab/imagecorruptions.git

echo "Installing spatial-correlation-sampler from GitHub..."
python -m pip install git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git

echo "Installing local ptlflow package..."
cd ptlflow
python -m pip install .

echo "Installing alt_cuda_corr extensiocondn..."
cd ptlflow/utils/external/alt_cuda_corr/
python setup.py install
cd ../../../

echo "All done!"
