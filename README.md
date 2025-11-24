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

```bash
mkdir mllib
cd mllib

python3 -m venv .venv --prompt mllib
source .venv/bin/activate
```

Essential tools:

```bash
sudo apt update
sudo apt install -y build-essential git cmake ninja-build python3-dev python3-pip clang libomp-dev libopenmpi-dev openmpi-bin openmpi-common gfortran
```

---

# Installation Instructions

## cuBLAS

```bash
sudo apt install -y libcublas-dev
```

---

## cuFile (GDS)

```bash
sudo apt install -y nvidia-gds
```

---

## cuDNN

```bash
sudo apt install -y libcudnn-dev
```

---

## cuSPARSELt

**Why:** `cuSPARSELt` provides optimized sparse GEMM (2:4 sparsity) kernels required to unlock Blackwell’s sparse Tensor Core throughput.

On DGX Spark, the default Ubuntu/DGX OS APT repositories **do not yet provide the latest cuSPARSELt version required for CUDA 13 and Blackwell (SM 12.x)**.  
Because of that, we must install cuSPARSELt manually using NVIDIA’s local repository package.  
This is a temporary situation — once NVIDIA updates the APT repos for DGX OS, this workaround will no longer be needed.


Create `install_cusparselt.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

CUSPARSELT_VERSION="0.8.0"
DISTRO="ubuntu2404"
DEB="cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}_${CUSPARSELT_VERSION}-1_arm64.deb"

echo "[cuSPARSELt] Downloading local repo package..."
wget -q --show-progress --progress=bar:force:noscroll --no-check-certificate \
  -O "${DEB}" \
  "https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/${DEB}"

echo "[cuSPARSELt] Installing local repo..."
sudo dpkg -i "${DEB}"

echo "[cuSPARSELt] Installing GPG key..."
sudo cp /var/cusparselt-local-repo-${DISTRO}-${CUSPARSELT_VERSION}/cusparselt-local-*-keyring.gpg /usr/share/keyrings/

echo "[cuSPARSELt] Updating APT cache for cuSPARSELt repo..."
sudo apt-get update

# Detect CUDA major if nvcc is available, otherwise default to 13
if command -v nvcc >/dev/null 2>&1; then
  CUDA_MAJOR=$(nvcc --version | awk -F'release ' '/release/{print $2}' | cut -d. -f1)
else
  CUDA_MAJOR="13"
fi
echo "[cuSPARSELt] Using CUDA major ${CUDA_MAJOR}"

# Prefer CUDA-major-suffixed packages if they exist
PKG_RT="libcusparselt0-cuda-${CUDA_MAJOR}"
PKG_DEV="libcusparselt0-dev-cuda-${CUDA_MAJOR}"

if ! apt-cache show "${PKG_RT}" >/dev/null 2>&1; then
  echo "[cuSPARSELt] Fallback to generic package names"
  PKG_RT="libcusparselt0"
  PKG_DEV="libcusparselt0-dev"
fi

echo "[cuSPARSELt] Installing: ${PKG_RT} ${PKG_DEV}"
sudo apt-get install -y "${PKG_RT}" "${PKG_DEV}"

echo "[cuSPARSELt] Installed packages:"
dpkg -l | grep -Ei 'cusparselt|libcusparselt' || true

echo "[cuSPARSELt] Refreshing linker cache..."
sudo ldconfig

echo "[cuSPARSELt] Done."
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
pip install triton/dist/*.whl
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
git checkout v0.9.2

mv pyproject.toml pyproject.toml.bak # Cause weird behavior

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
