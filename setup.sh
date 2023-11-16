#!/bin/bash

prefix=$PWD/HPAC/build_compiler
threads=$2
current_dir=$(pwd)

# set cc to default 'gcc' if 'CC' is not in the environment
if [ -z "$CC" ]; then
  CC=gcc
fi

if [ -z "$CXX" ]; then
  CXX=g++
fi

echo "Using CC=$CC and CXX=$CXX for HPAC build."

NOCOLOR='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
LIGHTGRAY='\033[0;37m'
DARKGRAY='\033[1;30m'
LIGHTRED='\033[1;31m'
LIGHTGREEN='\033[1;32m'
YELLOW='\033[1;33m'
LIGHTBLUE='\033[1;34m'
LIGHTPURPLE='\033[1;35m'
LIGHTCYAN='\033[1;36m'
WHITE='\033[1;37m'

clang_bin=$prefix/bin/clang

if [ ! -d "artifact_files" ]; then
  wget https://zenodo.org/record/8200217/files/hpac_offload_artifact.zip
  unzip hpac_offload_artifact.zip 
  mv hpac_offload_artifact artifact_files
  rm hpac_offload_artifact.zip
  mv artifact_files/HPAC .
  mv artifact_files/input_data .
fi

if [ ! -f $clang_bin ]; then
  if [ "$1" == "--git" ]; then
      echo "Downloading HPAC from git.."
      rm -rf HPAC
      git clone --single-branch --branch hpac_offload https://github.com/LLNL/HPAC
  fi
  pushd HPAC
  mkdir -p build_compiler
  pushd build_compiler
  cmake -G Ninja \
    -DCMAKE_INSTALL_PREFIX=$prefix \
    -DLLVM_CCACHE_BUILD='Off'\
    -DCMAKE_EXPORT_COMPILE_COMMANDS='On'\
    -DCMAKE_BUILD_TYPE='RelWithDebInfo' \
    -DLLVM_FORCE_ENABLE_STATS='On' \
    -DLLVM_ENABLE_PROJECTS='clang' \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DLLVM_ENABLE_RUNTIMES='openmp' \
    -DLLVM_OPTIMIZED_TABLEGEN='On' \
    -DCLANG_BUILD_EXAMPLES='On' \
    -DBUILD_SHARED_LIBS='On' \
    -DLLVM_ENABLE_ASSERTIONS='On' \
    ../llvm

    ninja -j $threads
    ninja -j $threads install 
    popd
    echo "#!/bin/bash" > hpac_env.sh
    echo "export PATH=$prefix/bin/:\$PATH" >> hpac_env.sh
    echo "export LD_LIBRARY_PATH=$prefix/lib/:\$LD_LIBRARY_PATH" >> hpac_env.sh
    echo "export CC=clang" >> hpac_env.sh
    echo "export CPP=clang++" >> hpac_env.sh
	popd
fi

pushd HPAC
gpuarchsm=$(python3 approx/approx_utilities/detect_arch.py $prefix)
gpuarch=$(echo $gpuarchsm | cut -d ';' -f 1)
gpusm=$(echo $gpuarchsm | cut -d ';' -f 2)

echo "export HPAC_GPU_ARCH=$gpuarch" >> hpac_env.sh
echo "export HPAC_GPU_SM=$gpusm" >> hpac_env.sh

if [ ! $? -eq 0 ]; then
  echo "ERROR: No GPU architecture targets found, exiting..."
  exit 1
else
  echo "Building for GPU architecture $gpuarch, compute capability $gpusm"
fi
source hpac_env.sh

popd

if [ -d "benchmarks/input_data" ]; then
  echo "Directory benchmarks/input_data already exists. Skipping download and unzip."
else
  # Download the archive using wget
  # Create the 'benchmarks' directory if it doesn't exist
  mkdir -p benchmarks

  # Move the 'input_data' directory to the 'benchmarks' directory
  mv input_data benchmarks/
fi

