# Optimizing the NVIDIA DGX Spark (Grace CPU & Blackwell GPU)

## Overview of NVIDIA DGX Spark

The NVIDIA DGX Spark is built around the NVIDIA GB10 Grace–Blackwell Superchip, pairing a high-performance Grace CPU with a next-generation Blackwell GPU on the same package. 

### Grace CPU (GB10)
- 20 ARM cores  
  - 10× Cortex-X925 (performance cores)  
  - 10× Cortex-A725 (efficiency cores)  

### Blackwell GPU (GB10)
- 6,144 CUDA cores  
- 192 5th Generation Tensor Cores
- Shares unified memory with the CPU (no PCIe bottlenecks)

### ASCII Architecture Diagram

```
                 NVIDIA GB10 Grace-Blackwell Superchip
   +-------------------------------------------------------------+
   |  Grace CPU: 20 cores (10× Cortex-X925 + 10× Cortex-A725)    |
   |                                                             |
   |    NVLink C2C Interconnect (CPU <--> GPU, coherent memory)  |
   |                                                             |
   |  Blackwell GPU: 6144 CUDA Cores                             |
   |                 192 Tensor Cores                            |
   |                                                             |
   |  128 GB Unified LPDDR5x Memory (CPU+GPU shared access)      |
   +-------------------------------------------------------------+
```

---

## Why the Software Stack Must Be Updated

When new hardware architectures such as **Grace–Blackwell** arrive, most libraries do **not yet contain optimized version** for the new Tensor Core instructions, memory hierarchy, or compute capability (SM **12.x** for Blackwell).

**Out-of-the-box PyTorch or CUDA libraries may:**
- fall back to older kernels (e.g., Hopper SM 8.9)
- fail to recognize new tensor core formats (FP8/FP4)
- miss optimized paths for sparsity or unified memory
- mis-handle Arm64 optimizations on Grace CPU

To unlock full performance, we must update/rebuild:
- CUDA libraries
- Triton compiler
- PyTorch itself  
… all compiled **specifically for SM 12.0 / 12.1** and ARM64.

---

## Goal: Optimize PyTorch for DGX Spark

We want PyTorch to:
- fully detect SM 12.x capabilities
- use the latest cuBLAS/cuDNN kernels
- enable FP4/FP8 Tensor Core kernels
- use cuFile (GDS) for direct SSD→GPU loading
- enable cuSPARSELt for 2:4 sparsity acceleration
- use Triton to JIT kernels optimized for Blackwell
- scale correctly with Grace ARM CPU performance

---

> **Skip the build?** If you want to use prebuilt wheels from releases instead of building from source, jump to [Install Directly From Prebuilt Wheels](#install-directly-from-prebuilt-wheels-skip-build).

## Libraries to Update

### CPU-side CUDA Libraries
| Library | Description |
|--------|-------------|
| **cuBLAS** | GPU-accelerated BLAS (matrix multiplication). Critical for Tensor Core GEMMs. |
| **cuFile** | Enables GPUDirect Storage (SSD → GPU DMA bypassing CPU). |
| **cuDNN** | Deep learning primitives (convolutions, RNNs, activation kernels). |
| **cuSPARSELt** | Structured sparse matrix multiplication (2:4 sparsity for Tensor Cores). |

### GPU-side Compiler Libraries
| Library | Description |
|--------|-------------|
| **LLVM** | Compiler backend required by Triton; generates PTX for SM 12.x. |
| **Triton** | Kernel compiler used by PyTorch for fused ops (FlashAttention, LayerNorm). |
| **PyTorch** | Must be rebuilt with all the above components and Blackwell flags enabled. |

---

## Prerequisites

### Recommended: Grace–Blackwell PyTorch autosetup helper

This repo includes `grace_blackwell_pytorch_autosetup.sh`, which discovers CUDA/cuDNN/NCCL/cuSPARSELt/cuFile on a DGX Spark (Grace CPU + Blackwell GPU) and exports the right `USE_*` / `TORCH_CUDA_ARCH_LIST` / `CUDA_*` environment variables for a PyTorch 2.9.1 build.

From a fresh Ubuntu/DGX OS environment on the DGX Spark:

```bash
git clone https://github.com/GuigsEvt/dgx_spark_config.git
cd dgx_spark_config

# Source the helper to install system deps and prepare env
source grace_blackwell_pytorch_autosetup.sh
```

This will:
- install core build dependencies and NVIDIA libraries via `apt-get`
- install cuSPARSELt local repo and packages for Ubuntu 24.04 arm64
- auto-detect CUDA, cuDNN, NCCL, cuFile, cuSPARSELt
- set `USE_CUDA`, `USE_CUDNN`, `USE_CUSPARSELT`, `USE_CUFILE`, `USE_SYSTEM_NCCL`, `USE_DISTRIBUTED`, `USE_FBGEMM_GENAI`, `USE_FLASH_ATTENTION`, `USE_MEM_EFF_ATTENTION`, etc.

You can still override any exported variable before building PyTorch if needed.

### Manual venv setup

```bash
mkdir mllib
cd mllib

python3 -m venv .venv --prompt mllib
source .venv/bin/activate
```

### Essential system packages for from-scratch releases

If you build all wheels from source (PyTorch + audio + vision + flash-attention), make sure this minimal set of system packages is installed first:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  curl \
  wget \
  pkg-config \
  python3 python3-dev python3-pip python3-setuptools python3-wheel python3-venv \
  libopenblas-dev \
  libcublas-dev-13-0 \
  libomp-dev \
  libopenmpi-dev mpi-default-bin \
  libuv1-dev \
  libssl-dev \
  zlib1g \
  cudnn9-cuda-13-0
```

---

## LLVM


We install the system LLVM package first because some build scripts expect
`clang`, `llvm-config`, or basic LLVM libraries to exist on the system.

```bash
sudo apt install -y llvm-20-dev
```

---

## Triton

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
```

### Python env for Triton:

```bash
sudo apt install -y python3.12-dev
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

After built the dependencies we need to build a custom LLVM to build the triton wheel from. 

```bash
cd $HOME/llvm-project  # your clone of LLVM.
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX"
ninja

export LLVM_BUILD_DIR=$(pwd)
```

### Build Triton Wheel

```bash
export LLVM_BUILD_DIR=$HOME/llvm-install
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR

pip wheel . -w dist --no-deps --verbose
```

### Install Triton

Switch back to main venv:

```bash
deactivate
source ~/mllib/.venv/bin/activate
pip install triton/dist/*.whlhi
```

---

# Building PyTorch for DGX Spark

## Create `build_env.sh`

```bash
export USE_CUDNN=1
export USE_CUBLAS=1
export USE_CUSPARSELT=1
export USE_CUFILE=1
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export USE_DISTRIBUTED=1
export USE_TENSORPIPE=1
export USE_FBGEMM=1
export USE_FBGEMM_GENAI=1
export USE_FLASH_ATTENTION=1
export USE_MEM_EFF_ATTENTION=1

export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export CUDAARCHS="121"

export TORCH_NVCC_FLAGS="-gencode=arch=compute_121,code=sm_121 --allow-unsupported-compiler"  # Force sm_121 sans fallback
export NVCC_APPEND_FLAGS="-D_FORCE_INLINES -allow-unsupported-compiler"  # Ignore les warnings sur arches futures
export NVCC_FLAGS_EXTRA=-"gencode;arch=compute_121,code=sm_121"
export CUDA_NVCC_EXECUTABLE="/usr/local/cuda/bin/nvcc"

export CMAKE_CUDA_ARCHITECTURES="120;121"

export PYTORCH_BUILD_VERSION="2.9.1"
export PYTORCH_BUILD_NUMBER="1"

export PATH=/usr/local/cuda-13.0/bin:$PATH;
export CUDA_HOME=/usr/local/cuda-13.0;
export CPATH=$CUDA_HOME/include:$CPATH;
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13;
export CMAKE_INCLUDE_PATH=/usr/local/cuda-13/include;
export CMAKE_LIBRARY_PATH=/home/${user}/jupyterlab/.venv/lib/python3.12/site-packages/nvidia/cu13/include;
```

Load it:

```bash
source build_env.sh
```

---

## Clone PyTorch

```bash
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v2.9.1
git submodule update --init --recursive
```

---

## Modify files for compute_121 & sm_121

```bash
git patch apply pytorch/pytorch.patch
cd third_party/flash-attention
git patch apply pytorch/flash-attention-local.patch
cd ../../
```

## Install build requirements

```bash
pip install -r requirements.txt -r requirements-build.txt
```

---

## Build Wheel

```bash
pip wheel . -w dist --no-deps --verbose
```

---

## Install your custom PyTorch

```bash
pip install dist/torch-*.whl
```

Verify:

```python
import torch
print(torch.__version__)
print(torch.cuda.get_device_properties(0))
```

You should see:
- CUDA capability **12.0 / 12.1**
- Device name: **Blackwell**
- Unified memory size: **128 GB**

---

# Building PyTorch ramifications

## Pytorch audio

```bash
sudo apt install ffmpeg

pip install numpy

git clone https://github.com/pytorch/audio.git
git checkout v2.9.1

export USE_CUDA=1;
export USE_FFMPEG=1;

python setup.py bdist_wheel # Create wheel file
```

## Pytorch video

```bash
git clone https://github.com/pytorch/vision.git
git checkout v0.24.1

python setup.py bdist_wheel # Create wheel file
```

## Flash attention

Flash attention is built from third party of Pytorch, we just need to build from there

```bash
cd pytorch/third_party/flash-attention

python setup.py bdist_wheel # Create wheel file
```

## Onnx

Onnx is built from third party of Pytorch, we just need to build from there

```bash

cd pytorch/third_party/onnx

export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
export CMAKE_ARGS="-DONNX_USE_LITE_PROTO=ON"

python setup.py bdist_wheel # Create wheel file
```

Do the same for any library from third party that you would need for your project.

## Reusable Wheelhouse (Symlink Strategy)

> Quick path: If you already have a tarball or directory of prebuilt release wheels (internal CI artifacts or official distribution), you can skip all build steps and jump directly to the wheelhouse usage below. Just extract/copy them into your wheelhouse directory.

After building all wheels (audio, vision, torch, triton, onnx, flash-attention, etc.) OR collecting prebuilt release wheels, you can create a central "wheelhouse" directory (symlinks or copies). This lets you: 
- Rapidly bootstrap new virtual environments (`pip install --no-index --find-links ...`) without rebuilding.
- Copy the collection to another DGX Spark node for identical setup.

### Create Wheelhouse
Choose a directory (example: `~/jupyterlab/mlwheel`).

```bash
export MLWHEEL=~/jupyterlab/mlwheel
mkdir -p "$MLWHEEL"

# Assuming your build directories (adjust paths as needed)
ln -s ~/jupyterlab/mllib/audio/dist/*.whl            "$MLWHEEL"/
ln -s ~/jupyterlab/mllib/vision/dist/*.whl           "$MLWHEEL"/
ln -s ~/jupyterlab/mllib/torch/dist/*.whl            "$MLWHEEL"/
ln -s ~/jupyterlab/mllib/triton/dist/*.whl           "$MLWHEEL"/
ln -s ~/jupyterlab/mllib/pytorch/third_party/onnx/dist/*.whl "$MLWHEEL"/
ln -s ~/jupyterlab/mllib/pytorch/third_party/flash-attention/dist/*.whl "$MLWHEEL"/

ls -1 "$MLWHEEL"  # verify links
```

If you need root-owned shared location (e.g. `/opt/mlwheel`):
```bash
sudo mkdir -p /opt/mlwheel
sudo chown $USER /opt/mlwheel
export MLWHEEL=/opt/mlwheel
# repeat ln -s commands above targeting /opt/mlwheel
```

### Use Wheelhouse in a New Environment
```bash
python3 -m venv .venv --prompt reuse
source .venv/bin/activate
pip install --no-index --find-links "$MLWHEEL" \
  torch torchvision torchaudio triton onnx flash-attention
```

`--no-index` ensures pip does not query PyPI; `--find-links` points pip to your local wheel set.

### Updating Wheels
When you rebuild a component (e.g. new PyTorch commit):
```bash
rm "$MLWHEEL"/torch-*.whl
ln -s ~/jupyterlab/mllib/torch/dist/torch-NEWVERSION.whl "$MLWHEEL"/
```
You can keep multiple versions if desired—omit the `rm` and pip will select the latest matching version unless you pin explicitly.

### Copying to Another DGX Spark Node
On source node:
```bash
tar czf mlwheel.tgz -C "$MLWHEEL" .
scp mlwheel.tgz other-node:/tmp/
```
On destination node:
```bash
mkdir -p ~/jupyterlab/mlwheel
tar xzf /tmp/mlwheel.tgz -C ~/jupyterlab/mlwheel
export MLWHEEL=~/jupyterlab/mlwheel
```
Now repeat the environment creation step using the extracted wheelhouse.

### Install Directly From Prebuilt Wheels (Skip Build)

You can use prebuilt wheels from the latest GitHub release. **Even when using wheels, you must first install the same system packages as for from-scratch builds:**

```bash
sudo apt-get update

sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  curl \
  wget \
  pkg-config \
  python3 python3-dev python3-pip python3-setuptools python3-wheel python3-venv \
  libopenblas-dev \
  libcublas-dev-13-0 \
  libomp-dev \
  libopenmpi-dev mpi-default-bin \
  libuv1-dev \
  libssl-dev \
  zlib1g \
  cudnn9-cuda-13-0
```

Then download the wheels archive from the [Releases page](https://github.com/GuigsEvt/dgx_spark_config/releases):

```bash
wget https://github.com/GuigsEvt/dgx_spark_config/releases/download/v1.0/wheels.tar.gz
mkdir -p wheels
tar xzf wheels.tar.gz -C wheels/
```

When installing wheels, **always install PyTorch and Triton first**, then the rest. This prevents other libraries from pulling a different, unoptimized PyTorch from PyPI:

```bash
cd wheels

# First: core stack (optimized Triton + PyTorch)
pip install ./triton-*.whl
pip install ./torch-*.whl

# Then: remaining wheels (audio, vision, onnx, flash-attention, etc.)
pip install ./*.whl
```

Validation:
```bash
python - <<'PY'
import torch
print('Torch version:', torch.__version__, 'CUDA:', torch.version.cuda)
print(torch.cuda.get_device_properties(0))
PY
```

# Conclusion

You now have a **Python environment with all libraries optimized for DGX Spark**. 

If you need to create a new environment, simply reinstall all packages using the built wheels. 

This optimized environment includes:

- Latest CUDA libraries  
- Triton compiled for SM 12.x  
- PyTorch rebuilt for Grace + Blackwell  
- Support for FP4/FP8 Tensor Cores  
- Support for GPUDirect Storage  
- Support for 2:4 sparsity acceleration  

---

# Benchmarking (FP16 GEMM Burn-In)

The FP16 GEMM burn‑in script `bench_gemm.py` lets you:
- Confirm the Blackwell GPU + SM 12.x path is active
- Load Tensor Cores with sustained FP16 GEMMs for ~60s
- Compare effective TFLOPs between baseline and optimized stacks

## Install PyTorch

Option A — Basic wheel via `bench/requirements.txt` (recommended for a quick baseline):

```bash
python3 -m venv .venv --prompt bench
source .venv/bin/activate
pip install -r bench/requirements.txt
```

Note: `bench/requirements.txt` pins `torch==2.9.0+cu130` and uses `--extra-index-url https://download.pytorch.org/whl/cu130` to ensure the GPU wheel is installed (CPU-only wheels are the default otherwise).

Option B — Source your previously built optimized environment (the one used to build the custom PyTorch wheel):

```bash
source ~/mllib/.venv/bin/activate
```

If you have not built it yet, follow the build steps above first.

Verify GPU build:
```python
import torch; print(torch.__version__, torch.version.cuda)
```

## Run the Benchmark (Both Environments)

Baseline (Option A environment):
```bash
python bench_gemm.py
```

Optimized (Option B environment):
```bash
python bench_gemm.py
```

The script will:
1. Print PyTorch / CUDA / cuDNN versions
2. Report GPU name, SM count, memory, compute capability
3. Warm up 20 FP16 GEMMs
4. Run FP16 GEMMs (4096×4096) ~60s with rolling TFLOPs
5. Emit summary (avg ms/iter + effective TFLOPs)

### Sample Output (Excerpt)
```
=== DGX Spark FP16 GEMM Burn-in ===
[PyTorch info]
  torch.version        : 2.9.0+cu130
  torch.cuda.version   : 13.0
  cudnn.version        : 9xx
[CUDA device]
  Name        : NVIDIA Blackwell ...
  SM count    : 128
  Compute cap : 12.1
[Burn-in]
  t=  1.0s | iters=   32 | avg= 31.25 ms | ~ 45.10 TFLOPs
  t=  2.0s | iters=   64 | avg= 31.30 ms | ~ 45.05 TFLOPs
  ...
[Summary]
  Effective    : 45.02 TFLOPs (FP16)
```

### Tuning
- Matrix size: change `M = N = K = 4096` in `bench_gemm.py` for lighter/heavier load (function defaults are larger; `main` overrides them).
- Duration: modify `target_seconds` (default 60).
- Data type: experiment with `torch.bfloat16` or FP8 types if kernels exist.

### Capturing System Metrics Concurrently
While the script runs you can capture utilization / power / clocks:

```bash
nvidia-smi dmon -s pucmt -d 1
```

Or, if DCGM is set up:
```bash
dcgmi stats -e 100 -d 1
```

### Interpreting Results
- Baseline vs optimized: expect up to ~50% uplift with rebuilt stack (cuBLAS + cuSPARSELt + Triton + PyTorch SM 12.x flags).
- Large variance (>5%) between seconds may indicate clock throttling or background load.
- Low TFLOPs (< expected) → confirm GPU wheel or custom build; check `TORCH_CUDA_ARCH_LIST` includes `12.0;12.1`.

### Benchmark Result Images

Baseline environment with matrix size of 8192 (public wheel):

![Baseline FP16 GEMM 8192](images/benchmark_basic_setup_dgx_gemm_8192.png)

Optimized environment with matrix size of 8192 (custom-built stack):

![Optimized FP16 GEMM 8192](images/benchmark_upgraded_setup_dgx_gemm_8192.png)

Observed improvement: ~50% higher sustained FP16 TFLOPs after rebuilding with Blackwell‑specific optimizations (better kernel selection, architecture flags, and sparse/flash attention paths). Exact uplift varies with thermal conditions and active system load.

### Summary of Uplift
- Public GPU wheel: baseline GEMM throughput (legacy kernels for SM 12.x)
- Custom build: enables tuned cuBLAS kernels + Triton JIT paths for SM 12.x
- Net effect: ~1.50× effective FP16 GEMM TFLOPs on this workload size.

---

## Co-Authors

- **Luc Bocahut** - luc@xrpl-commons.org
- **Martino Bettucci** - https://github.com/martinobettucci
