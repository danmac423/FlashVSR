#!/usr/bin/env bash
set -euo pipefail

CC=gcc-11 CXX=g++-11 NVCC_CCBIN=gcc-11 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="8.0;8.6" BLOCK_SPARSE_ATTN_CUDA_ARCHS="8.0;8.6" uv sync -v