#include "common.h"
#include "request_list.h"

// format:
// valid || type | u | u_degree | num_push_tasks | u_list | tasks |

#define TYPE_POS            0
#define U_POS               1
#define U_DEGREE_POS        2
#define NUM_PUSH_TASKS_POS  3
#define U_LIST_START        4

typedef struct {
    vidType type;
    vidType u;
    vidType u_degree;
    vidType *u_list;
    vidType v;
} update_info_t;

class SenderLocalPushBuffer{
protected:
    int ndevices;
    int slot_size;
    int n_sender_warps;
    int mype_id;

    int *valid;         // [ n_sender_warps, (ndevices-1) ]
    vidType *buffer;    // [ n_sender_warps, (ndevices-1), slot_size ]

    __device__ int get_dest_id_here(int v_partition_num) { return (v_partition_num < mype_id)? v_partition_num : (v_partition_num-1); }
    __device__ vidType *get_source(int sender_warp_id, int dest_id_here) { return &buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size]; }
    __device__ size_t get_msg_size(vidType *source) {
        vidType u_degree = source[U_DEGREE_POS];
        vidType num_push_tasks = source[NUM_PUSH_TASKS_POS];
        return 4 + u_degree + num_push_tasks;
    }
    __device__ void prepare_meta(update_info_t update_info, int sender_warp_id, int dest_id);
    __device__ void update_v(vidType v, int sender_warp_id, int dest_id);

public:
    SenderLocalPushBuffer(int _ndevices, int _slot_size, int _n_sender_warps, int _mype_id);
    __device__ void update(update_info_t update_info, int sender_warp_id, int v_partition_num);
    __device__ void send(int sender_warp_id, RequestList request_list);
};

SenderLocalPushBuffer::SenderLocalPushBuffer(int _ndevices, int _slot_size, int _n_sender_warps, int _mype_id)
: ndevices(_ndevices), slot_size(_slot_size), n_sender_warps(_n_sender_warps), mype_id(_mype_id) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&valid, n_sender_warps * (ndevices-1) * sizeof(int)));
    int *valid_h = (int *)malloc(n_sender_warps * (ndevices-1) * sizeof(int));
    for (int i = 0; i < n_sender_warps * (ndevices-1); i++) {
        valid_h[i] = 0;
    }
    CUDA_SAFE_CALL(cudaMemcpy(valid, valid_h, n_sender_warps * (ndevices-1) * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void **)&buffer, n_sender_warps * (ndevices-1) * slot_size * sizeof(vidType)));
}

__device__ void SenderLocalPushBuffer::prepare_meta(update_info_t update_info, int sender_warp_id, int dest_id) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    int dest_id_here = get_dest_id_here(dest_id);
    if (thread_lane == 0) {
        valid[sender_warp_id * (ndevices-1) + dest_id_here] = 1;
        buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + TYPE_POS] = update_info.type;
        buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + U_POS] = update_info.u;
        buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + U_DEGREE_POS] = update_info.u_degree;
        buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + NUM_PUSH_TASKS_POS] = 0;
        for (int i = 0; i < update_info.u_degree; i++) {    // TODO: find a better way to do this
            buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + U_LIST_START + i] = update_info.u_list[i];
        }
    } __syncwarp(); // TODO: can we remove it?
}

__device__ void SenderLocalPushBuffer::update_v(vidType v, int sender_warp_id, int dest_id) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    int dest_id_here = get_dest_id_here(dest_id);
    vidType* slot_addr = &buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size];

    vidType u_degree = slot_addr[U_DEGREE_POS];
    vidType num_push_tasks = slot_addr[NUM_PUSH_TASKS_POS];

    if (thread_lane == 0) {
        slot_addr[U_LIST_START + u_degree + num_push_tasks] = v;
        slot_addr[NUM_PUSH_TASKS_POS]++;
    } __syncwarp(); // TODO: can we remove it?
}

__device__ void SenderLocalPushBuffer::update(update_info_t update_info, int sender_warp_id, int v_partition_num) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int dest_id_here = get_dest_id_here(v_partition_num);
    // if (thread_lane == 0) {
    //     printf("sender [%d, %d] going to prepare meta.\n", nvshmem_my_pe(), sender_warp_id);
    // } __syncwarp();
    __threadfence();    // TODO: can we remove it?
    if (valid[sender_warp_id * (ndevices-1) + dest_id_here] == 0) {
        prepare_meta(update_info, sender_warp_id, v_partition_num);
    }
    // if (thread_lane == 0) {
    //     printf("sender [%d, %d] successfully prepared meta.\n", nvshmem_my_pe(), sender_warp_id);
    // } __syncwarp();
    update_v(update_info.v, sender_warp_id, v_partition_num);
}

__device__ void SenderLocalPushBuffer::send(int sender_warp_id, RequestList request_list) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    for (int dest_id = 0; dest_id < ndevices; dest_id++) {
        if (dest_id == mype_id) continue;
        int dest_id_here = get_dest_id_here(dest_id);
        __threadfence();    // TODO: can we remove it?
        if (valid[sender_warp_id * (ndevices-1) + dest_id_here] == 1) {
            vidType *send_addr = request_list.request(sender_warp_id, dest_id);
            vidType *source = get_source(sender_warp_id, dest_id_here);
            // put message content
            nvshmemx_int32_put_warp(send_addr + 1, source, get_msg_size(source), dest_id);
            nvshmem_fence();    // TODO: can we remove it?
            // put valid
            if (thread_lane == 0) {
                nvshmem_int32_p(send_addr, 1, dest_id);
            } __syncwarp();     // TODO: can we remove it?
            nvshmem_fence();    // TODO: can we remove it?

            // turn invalid
            if (thread_lane == 0) {
                valid[sender_warp_id * (ndevices-1) + dest_id_here] = 0;
            } __syncwarp();     // TODO: can we remove it?
        }
    }
}
