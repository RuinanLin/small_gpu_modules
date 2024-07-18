#include "common.h"

class SenderLocalPullBuffer{
protected:
    int slot_size;
    int n_sender_warps;

    vidType *buffer;    // [ n_sender_warps, slot_size ]

public:
    SenderLocalPullBuffer(int _slot_size, int _n_sender_warps)
    : slot_size(_slot_size), n_sender_warps(_n_sender_warps) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&buffer, n_sender_warps * slot_size * sizeof(vidType)));
    }

    __device__ vidType *pull(int sender_warp_id, vidType *addr, vidType v_degree, int v_partition_num) {
        vidType *pull_buffer_addr = &buffer[sender_warp_id * slot_size];
        nvshmemx_int32_get_warp(pull_buffer_addr, addr, v_degree, v_partition_num);
        return pull_buffer_addr;
    }
};