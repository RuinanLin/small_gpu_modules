#include "common.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cutil_subset.h"

class SendRecvBuffer {
protected:
    int ndevices;
    int slot_size;
    int n_slots_per_warp;
    int n_recver_warps;

    // occupied: the server warp has given this address to a sender warp
    // valid: the sender has filled in a message in this slot
    int *occupied;      // [ n_slots_per_warp, n_recver_warps ]
    vidType *buffer;    // [ n_slots_per_warp, n_recver_warps, (1+slot_size) ]
    int *end_counter;   // a counter that counts down. when reaching 0, all the servers have ended.

    int *next_slot_id_addr;

    __device__ int is_occupied(int slot_id_per_warp, int recver_warp_id) { return occupied[slot_id_per_warp * n_recver_warps + recver_warp_id]; }
    __device__ vidType isvalid(int slot_id_per_warp, int recver_warp_id) { return buffer[slot_id_per_warp * n_recver_warps * (1+slot_size) + recver_warp_id * (1+slot_size)]; }
    __device__ vidType *get_msg(int slot_id_per_warp, int recver_warp_id) { return &buffer[slot_id_per_warp * n_recver_warps * (1+slot_size) + recver_warp_id * (1+slot_size)]; }

public:
    // init
    SendRecvBuffer(int _ndevices, int _slot_size, int _n_slots_per_warp, int _n_recver_warps);

    // facing servers
    __device__ vidType *find_empty();
    __device__ void server_quit();

    // facing recvers
    __device__ vidType *check_msg(int recver_warp_id, int *slot_id_per_warp);
    __device__ void turn_invalid(int slot_id_per_warp, int recver_warp_id);
    __device__ int finished();
};

SendRecvBuffer::SendRecvBuffer(int _ndevices, int _slot_size, int _n_slots_per_warp, int _n_recver_warps)
: ndevices(_ndevices), slot_size(_slot_size), n_slots_per_warp(_n_slots_per_warp), n_recver_warps(_n_recver_warps) {
    occupied = (int *)nvshmem_malloc(n_slots_per_warp * n_recver_warps * sizeof(int));
    int *occupied_h = (int *)malloc(n_slots_per_warp * n_recver_warps * sizeof(int));
    for (int i = 0; i < n_slots_per_warp * n_recver_warps; i++) {
        occupied_h[i] = 0;
    }
    CUDA_SAFE_CALL(cudaMemcpy(occupied, occupied_h, n_slots_per_warp * n_recver_warps * sizeof(int), cudaMemcpyHostToDevice));

    // The first vidType is the valid field, so we must make it 0
    buffer = (vidType *)nvshmem_malloc(n_slots_per_warp * n_recver_warps * (1+slot_size) * sizeof(vidType));
    int *buffer_h = (int *)malloc(n_slots_per_warp * n_recver_warps * (1+slot_size) * sizeof(vidType));
    for (int i = 0; i < n_slots_per_warp * n_recver_warps; i++) {
        buffer_h[i * (1+slot_size)] = 0;
    }
    CUDA_SAFE_CALL(cudaMemcpy(buffer, buffer_h, n_slots_per_warp * n_recver_warps * (1+slot_size) * sizeof(vidType), cudaMemcpyHostToDevice));

    end_counter = (int *)nvshmem_malloc(sizeof(int));
    int end_counter_h = 1;  // TODO: change it to a multi-warp case
    CUDA_SAFE_CALL(cudaMemcpy(end_counter, &end_counter_h, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void **)&next_slot_id_addr, sizeof(int)));
    int next_slot_id_h = 0;
    CUDA_SAFE_CALL(cudaMemcpy(next_slot_id_addr, &next_slot_id_h, sizeof(int), cudaMemcpyHostToDevice));
}

// caller: server warp
__device__ vidType* SendRecvBuffer::find_empty() {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    while (1) {
        nvshmem_fence();    // TODO: can we remove it?
        __threadfence();    // TODO: can we remove it?
        int next_slot_id = *next_slot_id_addr;
        if (occupied[next_slot_id] == 0) {
            if (thread_lane == 0) {
                occupied[next_slot_id] = 1;     // TODO: multi-warp version
                nvshmem_fence();    // TODO: can we remove it?
                *next_slot_id_addr = next_slot_id + 1;  // TODO: multi-warp version
                __threadfence();    // TODO: can we remove it?
            } __syncwarp();     // TODO: can we remove it?
            return &buffer[next_slot_id * (1+slot_size)];
        } else {
            if (thread_lane == 0) {
                *next_slot_id_addr = next_slot_id + 1;  // TODO: multi-warp version
                __threadfence();    // TODO: can we remove it?
            } __syncwarp();     // TODO: can we remove it?
        }
    }
}

// caller: server warp
__device__ void SendRecvBuffer::server_quit() {     // server quit is signalling recvers
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    if (thread_lane == 0) {
        (*end_counter)--;     // TODO: change it to a multi-warp case, which means that it should consider race
        __threadfence();    // TODO: can we remove it?
    } __syncwarp();     // TODO: can we remove it?
}

// caller: recver warp
__device__ vidType* SendRecvBuffer::check_msg(int recver_warp_id, int *slot_id_per_warp) {
    for (int i = 0; i < n_slots_per_warp; i++) {
        if (isvalid(i, recver_warp_id)) {
            *slot_id_per_warp = i;
            return get_msg(i, recver_warp_id);
        }
    }
    return NULL;
}

// caller: recver warp
__device__ void SendRecvBuffer::turn_invalid(int slot_id_per_warp, int recver_warp_id) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    if (thread_lane == 0) {
        // turn invalid
        buffer[slot_id_per_warp * n_recver_warps * (1+slot_size) + recver_warp_id * (1+slot_size)] = 0;
        // turn not occupied
        occupied[slot_id_per_warp * n_recver_warps + recver_warp_id] = 0;
        __threadfence();    // TODO: can we remove it?
        nvshmem_fence();    // TODO: can we remove it?
    } __syncwarp();     // TODO: can we remove it?
}

// caller: recver warp
__device__ int SendRecvBuffer::finished() {
    nvshmem_fence();    // TODO: can we remove it?
    if (*end_counter == 0) {
        return 1;
    }
    return 0;
}
