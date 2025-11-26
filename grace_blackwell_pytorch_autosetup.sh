#!/usr/bin/env bash
# Grace + Blackwell – PyTorch 2.9.1 auto-setup (CUDA, cuDNN, cuBLAS, cuSPARSELt, cuFile, NCCL, MPI)
# Usage:  source ./grace_blackwell_pytorch_autosetup.sh

set -euo pipefail

echo "[INFO] PyTorch 2.9.1 auto-setup for CUDA + Grace/Blackwell (AArch64)"

ARCH="$(uname -m || echo unknown)"
if [[ "${ARCH}" != "aarch64" ]]; then
  echo "[WARN] Detected arch: ${ARCH} (script is tuned for aarch64 / Grace, but still usable)"
fi

############################################
# 1) System packages (build + BLAS + MPI…)
############################################

echo "[INFO] Installing base build & math/network packages via apt-get..."
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

############################################
# 2) cuFile (GPUDirect Storage) via nvidia-gds
############################################

echo "[INFO] Installing cuFile (GPUDirect Storage) via nvidia-gds..."
sudo apt-get install -y nvidia-gds || \
  echo "[WARN] nvidia-gds not found in current repos. Check NVIDIA repo configuration if you need cuFile."

############################################
# 3) NCCL (assumes NVIDIA repos configured)
############################################

echo "[INFO] Installing NCCL system packages if available..."
sudo apt-get install -y libnccl2 libnccl-dev || \
  echo "[WARN] libnccl2/libnccl-dev not found in current repos. Check NVIDIA repo configuration."

############################################
# 4) cuSPARSELt local repo + packages
############################################

echo "[INFO] Installing cuSPARSELt (0.8.1) local repo for Ubuntu 24.04 (arm64)..."

CUSPARSELT_DEB="cusparselt-local-repo-ubuntu2404-0.8.1_0.8.1-1_arm64.deb"
CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/${CUSPARSELT_DEB}"

if ! dpkg -s cusparselt-local-repo-ubuntu2404-0.8.1 >/dev/null 2>&1; then
  if [[ ! -f "${CUSPARSELT_DEB}" ]]; then
    echo "[INFO] Downloading ${CUSPARSELT_DEB} ..."
    wget -q "${CUSPARSELT_URL}"
  fi

  echo "[INFO] Installing ${CUSPARSELT_DEB} ..."
  sudo dpkg -i "${CUSPARSELT_DEB}" || true

  if [[ -d /var/cusparselt-local-repo-ubuntu2404-0.8.1 ]]; then
    sudo cp /var/cusparselt-local-repo-ubuntu2404-0.8.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ || true
  fi

  sudo apt-get update
else
  echo "[INFO] cuSPARSELt local repo already installed."
fi

sudo apt-get -y install cusparselt-cuda-12 cusparselt-cuda-13 || \
  echo "[WARN] cuSPARSELt CUDA packages not installed (check CUDA version / repo)."

############################################
# 5) Helper functions for discovery
############################################

has_ldconfig() {
  command -v ldconfig >/dev/null 2>&1
}

find_lib_dir() {
  local libname="$1"
  local path=""

  if has_ldconfig; then
    path="$(ldconfig -p 2>/dev/null | awk -v n="$libname" '$1 == n {print $NF; exit}')"
  fi

  if [[ -z "${path}" ]]; then
    path="$(find /usr /usr/local /opt -maxdepth 7 -type f -name "${libname}*" 2>/dev/null | head -n1 || true)"
  fi

  if [[ -n "${path}" ]]; then
    dirname "${path}"
  fi
}

find_header_dir() {
  local header="$1"
  local path=""

  for d in /usr/include /usr/local/include /usr/local/cuda/include; do
    if [[ -f "${d}/${header}" ]]; then
      echo "${d}"
      return
    fi
  done

  path="$(find /usr /usr/local /opt -maxdepth 7 -type f -name "${header}" 2>/dev/null | head -n1 || true)"
  if [[ -n "${path}" ]]; then
    dirname "${path}"
  fi
}

############################################
# 6) Discover CUDA / cuDNN / NCCL / cuSPARSELt / cuFile
############################################

echo "[INFO] Discovering CUDA / cuDNN / NCCL / cuSPARSELt / cuFile..."

# CUDA / nvcc
CUDA_NVCC_EXECUTABLE="$(command -v nvcc 2>/dev/null || true)"
if [[ -z "${CUDA_NVCC_EXECUTABLE}" ]]; then
  for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    if [[ -x "${p}" ]]; then
      CUDA_NVCC_EXECUTABLE="${p}"
      break
    fi
  done
fi

CUDA_HOME=""
if [[ -n "${CUDA_NVCC_EXECUTABLE}" ]]; then
  CUDA_HOME="$(cd "$(dirname "${CUDA_NVCC_EXECUTABLE}")/.." && pwd)"
  echo "[INFO] nvcc detected at: ${CUDA_NVCC_EXECUTABLE}"
  echo "[INFO] CUDA_HOME = ${CUDA_HOME}"
else
  echo "[ERROR] nvcc not found. Install a CUDA Toolkit (12.x+ recommended for Blackwell)."
fi

# NCCL
NCCL_LIB_DIR="$(find_lib_dir libnccl.so || true)"
NCCL_INCLUDE_DIR="$(find_header_dir nccl.h || true)"
NCCL_ROOT=""
if [[ -n "${NCCL_LIB_DIR}" ]]; then
  NCCL_ROOT="$(cd "${NCCL_LIB_DIR}/.." && pwd)"
fi

# cuDNN
CUDNN_LIB_DIR="$(find_lib_dir libcudnn.so || true)"
CUDNN_INCLUDE_DIR="$(find_header_dir cudnn.h || true)"
CUDNN_LIBRARY=""
if [[ -n "${CUDNN_LIB_DIR}" ]]; then
  CUDNN_LIBRARY="$(find "${CUDNN_LIB_DIR}" -maxdepth 1 -name 'libcudnn.so*' 2>/dev/null | sort | head -n1 || true)"
fi

# cuSPARSELt
CUSPARSELT_LIB_DIR="$(find_lib_dir libcusparseLt.so || true)"
CUSPARSELT_INCLUDE_DIR="$(find_header_dir cusparseLt.h || true)"

# cuFile
CUFILE_LIB_DIR="$(find_lib_dir libcufile.so || true)"
CUFILE_INCLUDE_DIR="$(find_header_dir cufile.h || true)"

# MPI
MPI_FOUND=0
if command -v mpicc >/dev/null 2>&1 && [[ -n "$(find_header_dir mpi.h || true)" ]]; then
  MPI_FOUND=1
fi

# libuv (TensorPipe / Gloo)
LIBUV_LIB_DIR="$(find_lib_dir libuv.so || true)"
LIBUV_INCLUDE_DIR="$(find_header_dir uv.h || true)"
LIBUV_FOUND=0
if [[ -n "${LIBUV_LIB_DIR}" && -n "${LIBUV_INCLUDE_DIR}" ]]; then
  LIBUV_FOUND=1
fi

# GPU compute capability via nvidia-smi
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
GPU_CC_MAJOR=""
GPU_CC_MINOR=""

if [[ -n "${GPU_CC}" ]]; then
  GPU_CC_MAJOR="${GPU_CC%%.*}"
  GPU_CC_MINOR="${GPU_CC##*.}"
  echo "[INFO] GPU compute capability detected: ${GPU_CC}"
else
  echo "[WARN] Could not query compute capability via nvidia-smi."
fi

############################################
# 7) Auto-decide USE_* and PyTorch env
############################################

# PyTorch versioning
export PYTORCH_BUILD_VERSION="2.9.1"
export PYTORCH_BUILD_NUMBER=""

# CUDA
if [[ -n "${CUDA_HOME}" ]]; then
  export USE_CUDA=1
else
  export USE_CUDA=0
fi

# cuDNN
if [[ -n "${CUDNN_LIB_DIR}" && -n "${CUDNN_INCLUDE_DIR}" ]]; then
  export USE_CUDNN=1
else
  export USE_CUDNN=0
fi

# cuSPARSELt
if [[ -n "${CUSPARSELT_LIB_DIR}" && -n "${CUSPARSELT_INCLUDE_DIR}" ]]; then
  export USE_CUSPARSELT=1
else
  export USE_CUSPARSELT=0
fi

# cuFile / GPUDirect Storage
if [[ -n "${CUFILE_LIB_DIR}" && -n "${CUFILE_INCLUDE_DIR}" ]]; then
  export USE_CUFILE=1
else
  export USE_CUFILE=0
fi

# NCCL système
if [[ -n "${NCCL_LIB_DIR}" && -n "${NCCL_INCLUDE_DIR}" ]]; then
  export USE_SYSTEM_NCCL=1
else
  export USE_SYSTEM_NCCL=0
fi

# MPI / Gloo / TensorPipe / Distributed
if [[ "${MPI_FOUND}" -eq 1 ]]; then
  export USE_MPI=1
else
  export USE_MPI=0
fi

if [[ "${LIBUV_FOUND}" -eq 1 ]]; then
  export USE_GLOO=1
  export USE_TENSORPIPE=1
else
  export USE_GLOO=1       # Gloo can still build
  export USE_TENSORPIPE=0 # disable if libuv missing
fi

if [[ "${USE_MPI}" -eq 1 || "${USE_SYSTEM_NCCL}" -eq 1 || "${USE_GLOO}" -eq 1 ]]; then
  export USE_DISTRIBUTED=1
else
  export USE_DISTRIBUTED=0
fi

# BLAS (CPU)
export BLAS="OpenBLAS"
export USE_SYSTEM_LIBS=1

# NNPACK
if [[ "${ARCH}" == "aarch64" ]]; then
  export USE_NNPACK=1
else
  export USE_NNPACK=0
fi

# FBGEMM CPU (x86 AVX-based) off on Arm
export USE_FBGEMM=0

# FBGEMM GenAI / Flash / MemEff attention (Ampere+)
if [[ -n "${GPU_CC_MAJOR}" && "${GPU_CC_MAJOR}" -ge 8 ]]; then
  export USE_FBGEMM_GENAI=1
  export USE_FLASH_ATTENTION=1
  export USE_MEM_EFF_ATTENTION=1
else
  export USE_FBGEMM_GENAI=0
  export USE_FLASH_ATTENTION=0
  export USE_MEM_EFF_ATTENTION=0
fi

# Misc options
export CMAKE_FRESH=1
export USE_ITT=0
export USE_MKLDNN=0
export BUILD_TEST=0
export USE_KINETO=0

# TORCH_CUDA_ARCH_LIST
if [[ -n "${GPU_CC_MAJOR}" ]]; then
  if [[ "${GPU_CC_MAJOR}" -eq 12 ]]; then
    # Grace + Blackwell expected case
    export TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"
  else
    export TORCH_CUDA_ARCH_LIST="${GPU_CC_MAJOR}.${GPU_CC_MINOR}+PTX"
  fi
else
  # Fallback: assume Blackwell
  export TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"
fi

# CMake / build parallelism
export CMAKE_GENERATOR="Ninja"
export MAX_JOBS="$(nproc)"

############################################
# 8) Export discovered paths
############################################

if [[ -n "${CUDA_NVCC_EXECUTABLE}" ]]; then
  export CUDA_NVCC_EXECUTABLE
fi

if [[ -n "${CUDA_HOME}" ]]; then
  export CUDA_HOME
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:/usr/local/lib:/usr/lib:${LIBRARY_PATH:-}"
fi

if [[ -n "${NCCL_INCLUDE_DIR}" ]]; then
  export NCCL_INCLUDE_DIR
fi
if [[ -n "${NCCL_LIB_DIR}" ]]; then
  export NCCL_LIB_DIR
fi
if [[ -n "${NCCL_ROOT}" ]]; then
  export NCCL_ROOT
fi

if [[ -n "${CUDNN_INCLUDE_DIR}" ]]; then
  export CUDNN_INCLUDE_DIR
fi
if [[ -n "${CUDNN_LIB_DIR}" ]]; then
  export CUDNN_LIB_DIR
fi
if [[ -n "${CUDNN_LIBRARY}" ]]; then
  export CUDNN_LIBRARY
fi

if [[ -n "${CUSPARSELT_LIB_DIR}" ]]; then
  export CUSPARSELT_LIB_DIR
fi
if [[ -n "${CUSPARSELT_INCLUDE_DIR}" ]]; then
  export CUSPARSELT_INCLUDE_DIR
fi

if [[ -n "${CUFILE_LIB_DIR}" ]]; then
  export CUFILE_LIB_DIR
fi
if [[ -n "${CUFILE_INCLUDE_DIR}" ]]; then
  export CUFILE_INCLUDE_DIR
fi

############################################
# 9) Final summary
############################################

echo
echo "========== PYTORCH BUILD CONFIG SUMMARY =========="
env | grep -E '^(PYTORCH_|USE_|TORCH_CUDA_ARCH_LIST=|CUDA_|CUDNN_|NCCL_|CUSPARSELT_|CUFILE_|BLAS=|CMAKE_|MAX_JOBS=)' | sort
echo "=================================================="
echo
echo "[INFO] Example build pipeline:"
echo "  git clone --recursive https://github.com/pytorch/pytorch.git"
echo "  cd pytorch"
echo "  git checkout v\${PYTORCH_BUILD_VERSION}"
echo "  pip3 install -r requirements.txt"
echo "  python3 setup.py bdist_wheel   # or develop/install"