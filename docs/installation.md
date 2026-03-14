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


### 1. Create Env

create a new virtualenv for this project:

```bash
# create a virtualenv for this project
conda create --name mott python=3.12 ipython jsonpatch yaml lap tqdm sacred submitit visdom pycocotools matplotlib opencv motmetrics scikit-image seaborn einops

# activate virtualenv
conda activate mott
```

### 2. install PyTorch

It is required to install CUDA toolkit locally in your OS to compile deformable attention module required by the decoder in the next step. This local CUDA version should match the version used to compile PyTorch library (you can check it in official PyTorch website).

For example, the PyTorch version 2.10.0 is compiled with CUDA 12.8. In order to compile the deformable attention module, you need to install CUDA 12.8 locally in your OS (check with `nvcc -V` in a terminal). 

```bash
# install pytorch (conda is no longer supported)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. install MSD-Attn Module

Note that we leverage Deformable Decoder in MOTT, and the original Deformable Decoder is no longer being updated and become completely obsolete, making it very difficult to install on newer systems. Thus, here we use a maintained version of Deformable Decoder from [Normam-Ou](https://github.com/Norman-Ou/Deformable-DETR-Torch2.x-cuda12), which is compatible with PyTorch 2.x and CUDA 12.x. 

Before installing the customized PyTorch module, we need the following prerequisites:

* local CUDA version match the one used to compile PyTorch library (e.g. CUDA 12.8)
* gcc version 10.0+

Install the MSD-Attn module:

```bash
cd src/trackformer/models/ops/
python setup.py build install
```

### 4. install additional libraries

#### 4.1 PyAV

For video loading in testing:

```bash
conda install conda-forge::av
```
