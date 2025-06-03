# Robustness Benchmark: Adversarial Attacks on Optical Flow Estimation

## Installation
Create Conda environment:
```
conda create --name benchmark python=3.10
conda activate benchmark
```

Install PyTorch version 1.13.1 following the official instructions at https://pytorch.org/.

If you are using Conda on Linux, you can use:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install PTLFlow, COSPGD and ImageCorruptions:
```
pip install ptlflow
pip install cospgd
pip install imagecorruptions
```

### Optional Dependencies
Two optional dependencies can be installed to increase the performance of some models.

For those, you need to have the CUDA toolkit installed (from https://developer.nvidia.com/cuda-toolkit-archive). For PyTorch version 1.13.1, this would be CUDA toolkit version 11.6. Pick the version corresponding to your PyTorch installation.

Install alt_cuda_corr:
```
cd ptlflow/ptlflow/utils/external/alt_cuda_corr/
python setup.py install
```

Install spatial-correlation-sampler:
```
pip install git+https://github.com/ClementPinard/Pytorch-Correlation-extension.git
```

If you used Conda and for your installation of PyTorch and have problems with you installation of CUDA, try:
```
conda config --add channels conda-forge
conda install conda-forge::cudatoolkit-dev=11.6.0

```

If you get ImportError: cannot import name 'packaging' from 'pkg_resources', try:
```
python -m pip install setuptools==69.5.1
```

### Model specific dependencies
matchflow: install Quadtree Attention (under ptlflow/ptlflow/models/matchflow/QuadtreeAttention setup.py install)
separableflow: intall GANet (under ptlflow/ptlflow/models/separableflow compile.sh)
scv4: install pytorch-scatter (conda install pytorch-scatter -c pyg)
neuflow: needs cupy (pip install cupy)
splatflow: needs pytorch 2.x

Note: To install, a gpu with CUDA is needed. 

### Optional Dependencies for Horeka:
After installing pytorch with Cuda 12.1, I installed Cudatoolkit with conda:
conda install nvidia/label/cuda-12.1.0::cuda-toolkit

then get with srun a gpu and you should be able to install the optionals.

If this is not working install cudatoolkit without sudo from nvidia:
https://developer.nvidia.com/cuda-12-1-0-download-archive

After this stackoverflow:
https://stackoverflow.com/questions/39379792/install-cuda-without-root

Then get the gpus with srun and install the optionals.