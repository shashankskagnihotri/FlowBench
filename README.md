# FlowBench: A Robustness Benchmark for Optical Flow Estimation

This repository provides comprehensive tools and pre-trained models for benchmarking the robustness of optical flow estimation algorithms.

---

## Installation

This package requires **Python 3.10.x** (tested with 3.10.17). Please ensure you have a compatible Python version installed.

> ⚠️ **CUDA Toolkit Version Note:**
> This package assumes your system uses **CUDA 11.8** (installed system-wide).
> If your system CUDA version differs from the one used to build the PyTorch installation, you may encounter runtime issues.
> In such cases, follow the official PyTorch installation instructions to install a compatible version:
> https://pytorch.org/get-started/locally/

### Step 1: Run the installation script

```bash
bash install.sh
```

### Step 2: Install this package in development mode

After running the installation script, install the core `flowbench` package:

```bash
pip install -e .
```

---

## Datasets

### KITTI2015

1. Download the KITTI 2015 dataset from the [KITTI Scene Flow Benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
2. After extraction, verify that the contents include `training/` and `testing/` directories.
3. Place the dataset at `datasets/kitti2015` or adjust the path in `ptlflow/datasets.yml`.

### MPI Sintel

1. Download the MPI Sintel dataset from the [MPI Sintel Flow Dataset](http://sintel.is.tue.mpg.de/downloads).
2. After extraction, verify that the contents include `training/` and `test/` directories.
3. Place the dataset at `datasets/Sintel`.
4. Download the MPI Sintel Depth training data from the [MPI Sintel Depth Training Data](http://sintel.is.tue.mpg.de/depth).
5. Extract the archive and verify it contains `training/camdata_left`, `training/depth`, and `training/depth_viz`. Place these directories under `datasets/Sintel/training` or adjust the path in `ptlflow/datasets.yml`.

Alternatively, use our convenience script:

```bash
bash download_mpi_sintel.sh
```

### 3D Common Corruptions Images

Download the precomputed 3D Common Corruption Images for KITTI2015 and MPI Sintel using the script below:

```bash
bash download_3dcc_data.sh
```

After download, the directory structure should look like:

```
datasets/3D_Common_Corruption_Images/kitti2015
datasets/3D_Common_Corruption_Images/Sintel
```

**Note:** Even if you only want to evaluate on the 3D Common Corruption Images, you must still download the original datasets to the expected folders to avoid errors.

### Adversarial Weather

There are two options:

1) Download precomputed weather particle files via script (or manually from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3677):

```bash
bash download_weather_files.sh
```

2) Generate your own variants with custom parameters. Follow the instructions from the [DistractingDownpour repository](https://github.com/cv-stuttgart/DistractingDownpour).

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

- See [`SUPPORTED_MODELS.md`](./SUPPORTED_MODELS.md)

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

- `retrieve_existing`: When `True` and a matching evaluation exists in the benchmark, returns the cached result. Otherwise, runs a new evaluation.
- `threat_model`: The type of adversarial attack
- `iterations`: Number of attack iterations
- `epsilon`: Permissible perturbation budget (ε)
- `alpha`: Step size of the attack (ϑ)
- `lp_norm`: Norm used to bound perturbation. Supported values: `'Linf'` or `'L2'`
- `targeted`: Boolean flag indicating whether the attack is targeted
- `target`: Target flow for a targeted attack (only applicable if `targeted=True`). Supported values: `'zero'` or `'negative'`
- `optim_wrt`: Flow used as a reference for optimization. Supported values: `'ground_truth'` or `'initial_flow'`

#### Adversarial Weather

```python
# demo.py
from flowbench.evals import evaluate

model, results = evaluate(
    model_name='RAFT',
    dataset='Sintel-Final',
    retrieve_existing=False,
    threat_model='Adversarial_Weather',
    weather='snow',
    num_particles=10000,
    targeted=True,
    target='zero',
    weather_data="datasets/adv_weather_data/weather_particles_red",
)
```

See the docstring of `evaluate()` for additional configuration options.

To use evaluate via command line:

```bash
python demo.py --weather_data path_to_particle_data
```

If you used `download_weather_files.sh`, three variants of particle data for the Sintel dataset should be available under `datasets/adv_weather_data/`. For example:

```bash
python demo.py --weather_data datasets/adv_weather_data/weather_snow_3000
```

- `retrieve_existing`: Works as described above
- `threat_model`: Must be `'Adversarial_Weather'`
- `weather`: Weather condition for the adversarial weather attack. Supported values: `'snow'`, `'fog'`, `'rain'`, or `'sparks'`
- `num_particles`: Number of particles per frame
- `targeted`: Boolean flag indicating whether the attack is targeted
- `target`: Target flow for a targeted attack (only applicable if `targeted=True`). Supported values: `'zero'` or `'negative'`

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

- `retrieve_existing`: Works as described above
- `threat_model`: Must be `'2DCommonCorruption'`; returns evaluations across 15 corruption types
- `severity`: An integer from 1 to 5 indicating the corruption severity

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

- `retrieve_existing`: Works as described above
- `threat_model`: Must be `'3DCommonCorruption'`; returns evaluations across 8 corruption types
- `severity`: An integer from 1 to 5 indicating the corruption severity