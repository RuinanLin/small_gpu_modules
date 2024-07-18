#include "common.h"
#include "request_list.h"

// format:
// | type | u | u_degree | num_push_tasks | u_list | tasks |
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

    RequestList request_list;

public:
    SenderLocalPushBuffer(int _ndevices, int _slot_size, int _n_sender_warps, int _mype_id, RequestList _request_list)
    : ndevices(_ndevices), slot_size(_slot_size), n_sender_warps(_n_sender_warps), mype_id(_mype_id), request_list(_request_list) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&valid, n_sender_warps * (ndevices-1) * sizeof(int)));
        int *valid_h = (int *)malloc(n_sender_warps * (ndevices-1) * sizeof(int));
        for (int i = 0; i < n_sender_warps * (ndevices-1) * slot_size; i++) {
            valid_h[i] = 0;
        }
        CUDA_SAFE_CALL(cudaMemcpy(valid, valid_h, n_sender_warps * (ndevices-1) * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMalloc((void **)&buffer, n_sender_warps * (ndevices-1) * slot_size * sizeof(vidType)));
    }

    __device__ int get_dest_id_here(int v_partition_num) { return (v_partition_num < mype_id)? v_partition_num : (v_partition_num-1); }
    __device__ vidType *get_source(int sender_warp_id, int dest_id_here) { return &buffer[sender_warp_id * (ndevices-1) + dest_id_here]; }
    __device__ size_t get_msg_size(vidType *source) {
        vidType u_degree = source[U_DEGREE_POS];
        vidType num_push_tasks = source[NUM_PUSH_TASKS_POS];
        return 4 + u_degree + num_push_tasks;
    }

    __device__ void prepare_meta(update_info_t update_info, int sender_warp_id, int dest_id_here) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            valid[sender_warp_id * (ndevices-1) + dest_id_here] = 1;
            buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + TYPE_POS] = update_info.type;
            buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + U_POS] = update_info.u;
            buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + U_DEGREE_POS] = update_info.u_degree;
            buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + NUM_PUSH_TASKS_POS] = 0;
            for (int i = 0; i < update_info.u_degree; i++) {    // TODO: find a better way to do this
                buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + U_LIST_START + i] = update_info.u_list[i];
            }
        } __syncwarp();
    }

    __device__ void update_v(vidType v, int sender_warp_id, int dest_id_here) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);

        vidType u_degree = buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + U_DEGREE_POS];
        vidType num_push_tasks = buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + NUM_PUSH_TASKS_POS];

        if (thread_lane == 0) {
            buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + U_LIST_START + u_degree + num_push_tasks] = v;
            buffer[sender_warp_id * (ndevices-1) * slot_size + dest_id_here * slot_size + NUM_PUSH_TASKS_POS]++;
        } __syncwarp();
    }

    __device__ void update(update_info_t update_info, int sender_warp_id, int v_partition_num) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        int dest_id_here = get_dest_id_here(v_partition_num);
        if (!isvalid(sender_warp_id, dest_id_here)) {
            prepare_meta(update_info, sender_warp_id, dest_id_here);
        }
        update_v(v, sender_warp_id, dest_id_here);
    }

    __device__ int isvalid(int sender_warp_id, int dest_id) {
        return valid[sender_warp_id * (ndevices-1) + dest_id];
    }

    __device__ void send(int sender_warp_id) {
        for (int dest_id = 0; dest_id < ndevices; dest_id++) {
            if (dest_id == mype_id) continue;
            int dest_id_here = get_dest_id_here(dest_id);
            if (isvalid(sender_warp_id, dest_id_here)) {
                vidType *send_addr = request_list.request(sender_warp_id, dest_id);
                vidType *source = get_source(sender_warp_id, dest_id_here);
                nvshmemx_int64_put_warp(send_addr, source, get_msg_size(source), dest_id);
                nvshmem_fence();    // TODO: can we remove it?
            }
        }
    }
};
