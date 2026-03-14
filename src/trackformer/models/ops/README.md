# Deformable-DETR-Torch2.x-cuda12

🔧 A minimal, **PyTorch 2.x** and **CUDA 12** compatible reimplementation of the **Multi-Scale Deformable Attention** CUDA ops from the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) project.

This repo serves as a **drop-in replacement** for the original [`models/ops`](https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops) directory in Deformable DETR.

It is also compatible with **any downstream variant** or fork that uses Deformable Attention and requires custom CUDA compilation—such as **DINO**, **OpenSeeD**, or other detection/segmentation models built on top of Deformable DETR.

By resolving API deprecations and compilation issues in PyTorch ≥2.1 and CUDA ≥12.0, this repo allows seamless integration into modern codebases and hardware environments (e.g., NVIDIA H100).

This module implements both the forward and backward CUDA kernels compatible with:
- `AT_DISPATCH_FLOATING_TYPES_AND_HALF`
- `value.scalar_type()` (instead of deprecated `value.type()`)

# 🔄 Update Info

This repo was last updated on **2025.10.10**, and verified to work under the following environment:

- **PyTorch**: 2.8.0  
- **TorchVision**: 0.23.0  
- **CUDA Toolkit**: 12.5  
- **GPU**: NVIDIA H100  

# 🚀 Usage

Clone this repo and install the extension with:

```bash
git clone git@github.com:Norman-Ou/Deformable-DETR-Torch2.x-cuda12.git
cd Deformable-DETR-Torch2.x-cuda12
```

```python
python setup.py build install
```

Or, for development mode:

```bash
pip install -v -e .
```

# 🧪 Quick Test

After installation, you can test if the module works:

```python
from MultiScaleDeformableAttention import MSDeformAttnFunction
print("Deformable attention module imported successfully!")
```

# 🧩 Why This Repo?

The original ms_deform_attn module from Deformable DETR is not compatible with recent versions of PyTorch and CUDA due to deprecated API usage. This repo provides a clean fix without modifying the model logic, suitable for modern environments and HPC clusters with newer GPUs.
