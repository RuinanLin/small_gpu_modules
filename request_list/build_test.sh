#!/bin/bash
module load nvhpc
SDK_HOME="$(echo $(dirname $(which nvc++)) | sed "s/\/compilers\/bin.*//g")"
NVSHMEM_HOME="$SDK_HOME/comm_libs/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"
export NVSHMEM_BOOTSTRAP_PMI=PMI-2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/rlin2/metis-5.1.0/build/Linux-x86_64/my_build/lib

nvcc -rdc=true -ccbin nvc++ -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen/gcc-8.5.0/nvhpc-22.11-eionxjt/Linux_x86_64/22.11/comm_libs/nvshmem/include test_request_list.cu -c -o test_request_list.o
nvc++ test_request_list.o -o test_request_list.out -cuda -gpu=cc80 -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen/gcc-8.5.0/nvhpc-22.11-eionxjt/Linux_x86_64/22.11/comm_libs/nvshmem/lib -L/u/rlin2/metis-5.1.0/build/Linux-x86_64/my_build/lib -lnvshmem_host -lnvshmem_device -lnvidia-ml -lcuda -lcudart -lmetis

srun --account=bcsh-delta-gpu --partition=gpuA100x4-interactive -G 4 -n 4 -N 1 --mem=240G ./test_request_list.out
