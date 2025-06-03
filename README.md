# Robustness Benchmark: Adversarial Attacks on Optical Flow Estimation

This repository provides tools and pre-trained models for benchmarking the robustness of optical flow estimation.

---

## Installation

This package requires **Python 3.10.x** (tested with 3.10.17). Please ensure you have a compatible Python version installed.

> ⚠️ **CUDA Toolkit Version Warning:**  
> This package is installed assuming your system uses **CUDA 11.8** (i.e., you have the CUDA 11.8 toolkit installed system-wide).  
> If your system CUDA version differs from the one used to build the PyTorch installation, you may encounter runtime issues.  
> In that case, follow the official PyTorch installation instructions to install the appropriate version matching your system:  
> https://pytorch.org/get-started/locally/

### Step 1: Run the installation script

```bash
bash install.sh
```

### Step 2: Install this package in editable mode

After running the install script, install the core `flowbench` package:

```bash
pip install -e .
```

---

### Datasets

#### KITTI2015

1. Download the KITTI 2015 dataset from the [KITTI Scene Flow Benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
2. After unzipping, ensure the contents include `training/` and `testing/` directories.
3. Move the dataset to the following path: `datasets/kitti2015`.

#### MPI Sintel

1. Download the MPI Sintel dataset from the [MPI Sintel Flow Dataset](http://sintel.is.tue.mpg.de/downloads).
2. After unzipping, ensure the contents include `training/` and `test/` directories.
3. Move the dataset to the following path: `datasets/Sintel`.
4. Download the MPI Sintel Depth training data from the [MPI Sintel Depth Training Data](http://sintel.is.tue.mpg.de/depth).
5. Unzip the archive and ensure it contains `training/camdata_left`, `training/depth`, and `training/depth_viz`. Move these directories under `datasets/Sintel/training`.

#### 3D Common Corruptions Images

1. Download the precomputed 3D Common Corruption Images for KITTI2015 and MPI Sintel using the script below. After download, the directory structure should look like:

```
datasets/3D_Common_Corruption_Images/kitti2015
datasets/3D_Common_Corruption_Images/Sintel
```

```bash
bash download.sh
```

---

## How to Use

### Model Zoo

```python
from flowbench.evals import load_model

model = load_model(
    model_name='RAFT',
    dataset='KITTI2015',
)
```

#### Supported Models

To browse the full list of supported models:

- View [`SUPPORTED_MODELS.md`](./SUPPORTED_MODELS.md)

### Evaluation

#### Adversarial Attacks

```python
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT', 
    dataset='KITTI2015', 
    retrieve_existing=True, 
    threat_model='PGD', 
    iterations=20, epsilon=8/255, alpha=0.01,
    lp_norm='Linf', optim_wrt='ground_truth',
    targeted=True, target='zero',
)
```

- `retrieve_existing` is a boolean flag. If set to `True` and a matching evaluation exists in the benchmark, the cached result will be returned. Otherwise, the evaluation will be run.
- `threat_model`: the type of the adversarial attack
- `iterations`: number of attack iterations
- `epsilon`: permissible perturbation budget (ε)
- `alpha`: step size of the attack (ϑ)
- `lp_norm`: the norm used to bound perturbation. Supported values: `'Linf'` or `'L2'`
- `targeted`: boolean flag indicating whether the attack is targeted
- `target`: target flow for a targeted attack (only applicable if `targeted=True`). Supported values: `'zero'` or `'negative'`
- `optim_wrt`: flow used as a reference for optimization. Supported values: `'ground_truth'` or `'initial_flow'`

#### Adversarial Weather

```python
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT', 
    dataset='KITTI2015', 
    retrieve_existing=True, 
    threat_model='Adversarial_Weather', 
    weather='snow', num_particles=10000, 
    targeted=True, target='zero',
)
```

- `retrieve_existing` works as described above.
- `threat_model`: `'Adversarial_Weather'`
- `weather`: weather condition in adversarial weather attack. Supported values: `'snow'`, `'fog'`, `'rain'` or `'sparks'`
- `num_particles`: number of particles per frame to be used
- `targeted`: boolean flag indicating whether the attack is targeted
- `target`: target flow for a targeted attack (only applicable if `targeted=True`). Supported values: `'zero'` or `'negative'`

#### 2D Common Corruptions

```python
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT',
    dataset='KITTI2015',
    retrieve_existing=True, 
    threat_model='2DCommonCorruption', 
    severity=3, 
)
```

- `retrieve_existing` works as described above.
- `threat_model`: must be `'2DCommonCorruption'`; returns the evaluations across 15 corruption types
- `severity`: an integer from 1 to 5 indicating the corruption severity

#### 3D Common Corruptions

```python
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT',
    dataset='KITTI2015',
    retrieve_existing=True, 
    threat_model='3DCommonCorruption', 
    severity=3, 
)
```

- `retrieve_existing` works as described above.
- `threat_model`: must be `'3DCommonCorruption'`; returns the evaluations across 8 corruption types
- `severity`: an integer from 1 to 5 indicating the corruption severity
