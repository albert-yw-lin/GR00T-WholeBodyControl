# Conda-Based Development Setup

This guide covers the full setup for building `gear_sonic_deploy` using a conda environment on x86_64 desktop systems.

## Prerequisites

- NVIDIA GPU driver installed system-wide (`nvidia-smi` should work)
- [Miniforge](https://github.com/conda-forge/miniforge) or Miniconda installed
- [TensorRT](https://developer.nvidia.com/tensorrt) downloaded and extracted (not available via conda)
- ROS2 Humble installed system-wide at `/opt/ros/humble` (for ROS2InputHandler support)

## Step 1: Create the Conda Environment

```bash
cd gear_sonic_deploy
conda env create -f environment.yml
conda activate gear-sonic
```

This installs all build tools (cmake, clang, just), C++ libraries (eigen, msgpack, zmq, etc.), ONNX Runtime, and the CUDA 12.4 toolkit.

## Step 2: Set Up Conda Activation Variables

Create an activation script so that environment variables are automatically set every time you `conda activate gear-sonic`:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/gear_sonic.sh" << 'EOF'
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
export onnxruntime_ROOT="$CONDA_PREFIX"
export TensorRT_ROOT="$HOME/TensorRT"  # adjust to your TensorRT location
EOF
```

> Change `$HOME/TensorRT` to wherever you extracted TensorRT.

After creating the script, re-activate the environment to pick up the variables:

```bash
conda deactivate && conda activate gear-sonic
```

## Step 3: Source the Runtime Environment

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
```

This configures:
- ONNX Runtime discovery
- ROS2 Humble (sources `/opt/ros/humble/setup.bash`)
- FastRTPS profile
- TensorRT and CUDA library paths (`LD_LIBRARY_PATH`)
- Git LFS

## Step 4: Build

```bash
just build
```

## Summary of What Lives Where

| Component | Source | Notes |
|-----------|--------|-------|
| Build tools (cmake, clang, just) | conda (`environment.yml`) | |
| C++ libraries (eigen, zmq, msgpack, etc.) | conda (`environment.yml`) | |
| Boost headers | conda (`libboost-headers`) | Required by `msgpack-cxx` |
| ONNX Runtime | conda (`onnxruntime-cpp`) | |
| CUDA toolkit (nvcc, headers, runtime) | conda (`cuda-toolkit`) | Driver must be system-wide |
| TensorRT | Manual install | Set `TensorRT_ROOT` in activation script |
| ROS2 Humble | System (`/opt/ros/humble`) | `setup_env.sh` sources it |
| Unitree SDK | Vendored in `thirdparty/` | |

## Troubleshooting

**Missing `boost/predef/other/endian.h`**
Install boost headers: `conda install -c conda-forge libboost-headers`

**`(g1_deploy)` in prompt**
This is cosmetic, added by `setup_env.sh` to indicate the deploy environment is sourced. It is not a virtual environment.

**TensorRT not found**
Ensure `TensorRT_ROOT` points to your TensorRT directory (must contain `include/` and `lib/`). Verify with: `echo $TensorRT_ROOT`

**CUDA version mismatch**
The conda environment pins `cuda-toolkit =12.4`. Your system driver must support CUDA 12.4+. Check with: `nvidia-smi`
