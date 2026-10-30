# Build environment for gsplat's JIT CUDA extensions.
# The system nvcc is 11.5, too old to parse this machine's gcc-11 libstdc++
# headers (the "parameter packs not expanded" error). CUDA 12.3 is present and
# compiles cleanly, so it is pinned here for every run.
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:${PATH:-}
# ${VAR:-} throughout: these are sourced by scripts running under `set -u`,
# where a bare $LD_LIBRARY_PATH on a machine that has never set it aborts.
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.0"
export COLMAP_BIN=${COLMAP_BIN:-colmap}
export TWINWORLD_WORK=work
