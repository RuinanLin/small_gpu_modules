#include "common.h"

// format:
// | type | u | u_degree | num_push_tasks | u_list | tasks |

class SendRecvBuffer {
protected:
    int ndevices;
    int slot_size;
    int n_slots_per_warp;
    int n_recver_warps;

    int *valid;         // [ n_slots_per_warp, n_recver_warps ]
    vidType *buffer;    // [ n_slots_per_warp, n_recver_warps, slot_size ]
    int *end_counter;   // a counter that counts down. when reaching 0, all the servers have ended.

    int next_slot_id;

public:
    SendRecvBuffer(int _ndevices, int _slot_size, int _n_slots_per_warp, int _n_recver_warps)
    : ndevices(_ndevices), slot_size(_slot_size), n_slots_per_warp(_n_slots_per_warp), n_recver_warps(_n_recver_warps), next_slot_id(0) {
        valid = (int *)nvshmem_malloc(n_slots_per_warp * n_recver_warps * sizeof(int));
        int *valid_h = (int *)malloc(n_slots_per_warp * n_recver_warps * sizeof(int));
        for (int i = 0; i < n_slots_per_warp * n_recver_warps; i++) {
            valid_h[i] = 0;
        }
        CUDA_SAFE_CALL(cudaMemcpy(valid, valid_h, n_slots_per_warp * n_recver_warps * sizeof(int), cudaMemcpyHostToDevice));

        buffer = (vidType *)nvshmem_malloc(n_slots_per_warp * n_recver_warps * slot_size * sizeof(vidType));

        end_counter = (int *)nvshmem_malloc(sizeof(int));
        int end_counter_h = ndevices-1;
        CUDA_SAFE_CALL(cudaMemcpy(end_counter, &end_counter_h, sizeof(int), cudaMemcpyHostToDevice));
    }

    __device__ int isvalid(int slot_id_per_warp, int recver_warp_id) { return valid[slot_id_per_warp * n_recver_warps + recver_warp_id]; }
    __device__ vidType *get_msg(int slot_id_per_warp, int recver_warp_id) { return &buffer[slot_id_per_warp * n_recver_warps * slot_size + recver_warp_id * slot_size]; }

    __device__ vidType *find_empty() {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        while (1) {
            nvshmem_fence();    // TODO: can we remove it?
            if (valid[next_slot_id] == 0) {
                vidType *return_addr = &buffer[next_slot_id * slot_size];
                if (thread_lane == 0) {
                    valid[next_slot_id] = 1;
                    nvshmem_fence();    // TODO: can we remove it?
                    next_slot_id++;
                    __threadfence();    // TODO: can we remove it?
                } __syncwarp(); // TODO: can we remove it?
                return return_addr;
            } else {
                if (thread_lane == 0) {
                    next_slot_id++;
                    __threadfence();    // TODO: can we remove it?
                } __syncwarp();
            }
        }
    }

    __device__ void server_quit() {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);

        if (thread_lane == 0) {
            for (int dest_id = 0; dest_id < ndevices; dest_id++) {
                if (dest_id == mype_id) continue;
                nvshmem_int_atomic_add(end_counter, -1, dest_id);
            }
        } __syncwarp();     // TODO: can we remove it?
    }

    __device__ int finished() {
        nvshmem_fence();
        if (*end_counter == 0) {
            return 1;
        }
        return 0;
    }

    __device__ vidType *check_msg(int recver_warp_id, int *slot_id_per_warp) {
        for (int i = 0; i < n_slots_per_warp; i++) {
            if (isvalid(i, recver_warp_id)) {
                *slot_id_per_warp = i;
                return get_msg(i, recver_warp_id);
            }
        }
        return NULL;
    }

    __device__ void turn_invalid(int slot_id_per_warp, int recver_warp_id) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);

        if (thread_lane == 0) {
            valid[slot_id_per_warp * n_recver_warps + recver_warp_id] = 0;
            __threadfence();
        } __syncwarp(); // TODO: can we remove it?
    }
};
