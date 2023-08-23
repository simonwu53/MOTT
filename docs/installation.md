# Installation

## Using Conda (Miniconda/Anaconda)d

### 0. Prepare Conda Env

Install `miniconda` from this [link](https://docs.conda.io/en/latest/miniconda.html), set conda-forge as first channel
and set flexible channel_priority.

```bash
# update conda
conda update -n base conda

# add and make conda-forge the first channel
conda config --add channels conda-forge
```

install libmamba (a faster conda solver):

```bash
conda install -n base conda-libmamba-solver
```

### 1. Create Env

create a new virtualenv for this project:

```bash
# create a virtualenv for this project
conda create --name trackformer --experimental-solver=libmamba python=3.9 ipython jsonpatch yaml lap tqdm sacred submitit visdom pycocotools matplotlib opencv motmetrics scikit-image seaborn einops 

# activate virtualenv
conda activate trackformer
```

### 2. install pytorch

Before this step, it's wise to check the local CUDA installation first. While `conda` uses a separated
compiled `cudatoolkit` for each environment, we still need a local CUDA installation to install the customized PyTorch
module written in pure CUDA & cpp code in. Hence, the CUDA version in `conda`
and local must be matched. As an example below, if you install `cudatoolkit=11.3` in `conda`, you should have the same
11.3 version shown in `nvcc -v` in a terminal.

```bash
# install pytorch 1.12.1 and cudatoolkit 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch --experimental-solver=libmamba --strict-channel-priority
```

### 3. install MSD-Attn Module

Before installing the customized PyTorch module, we need the following prerequisites:

* local CUDA version 11.3 (the same as above)
* gcc version 6.0-10.0

Install the MSD-Attn module:

```bash
cd src/trackformer/models/ops/
python setup.py build install
```

### 4. install additional libraries

#### 4.1 PyAV

For video loading in testing:

```bash
conda install -c conda-forge av --experimental-solver=libmamba
```