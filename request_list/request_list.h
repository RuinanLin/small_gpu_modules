#include "common.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cutil_subset.h"

class RequestList{
protected:
    int ndevices;
    int n_sender_warps;
    int mype_id;

    int *signal;    // [ ndevices-1, n_sender_warps ]
    vidType **resp; // [ ndevices-1, n_sender_warps ]
    int *end_counter;   // a counter that counts down. when reaching 0, all the senders have ended.

    __device__ int get_mype_id_there(int dest_id) { return (mype_id < dest_id)? mype_id : (mype_id-1); }
    __device__ int get_src_id_here(int src_id) { return (src_id < mype_id)? src_id : (src_id-1); }
    __device__ int *get_signal_slot(int src_id_here, int sender_warp_id) { return &signal[src_id_here * n_sender_warps + sender_warp_id]; }
    __device__ int *get_remote_signal_slot(int mype_id_there, int sender_warp_id) { return &signal[mype_id_there * n_sender_warps + sender_warp_id]; }
    __device__ vidType **get_resp_slot(int mype_id_there, int sender_warp_id) { return &resp[mype_id_there * n_sender_warps + sender_warp_id]; }
    __device__ vidType **get_remote_resp_slot(int mype_id_there, int sender_warp_id) { return &resp[mype_id_there * n_sender_warps + sender_warp_id]; }

public:
    // init
    RequestList(int _ndevices, int _n_sender_warps, int _mype_id);

    // facing senders
    __device__ vidType *request(int sender_warp_id, int dest_id);
    __device__ void sender_quit();

    // facing servers
    __device__ int check(int src_id, int sender_warp_id);
    __device__ void respond(int src_id, int sender_warp_id, vidType *send_recv_slot);
    __device__ int finished();
};

RequestList::RequestList(int _ndevices, int _n_sender_warps, int _mype_id)
: ndevices(_ndevices), n_sender_warps(_n_sender_warps), mype_id(_mype_id) {
    signal = (int *)nvshmem_malloc((ndevices-1) * n_sender_warps * sizeof(int));
    int *signal_h = (int *)malloc((ndevices-1) * n_sender_warps * sizeof(int));
    for (int i = 0; i < (ndevices-1) * n_sender_warps; i++) {
        signal_h[i] = 0;
    }
    CUDA_SAFE_CALL(cudaMemcpy(signal, signal_h, (ndevices-1) * n_sender_warps * sizeof(int), cudaMemcpyHostToDevice));

    resp = (vidType **)nvshmem_malloc((ndevices-1) * n_sender_warps * sizeof(vidType *));
    vidType **resp_h = (vidType **)malloc((ndevices-1) * n_sender_warps * sizeof(vidType *));
    for (int i = 0; i < (ndevices-1) * n_sender_warps; i++) {
        resp_h[i] = NULL;
    }
    CUDA_SAFE_CALL(cudaMemcpy(resp, resp_h, (ndevices-1) * n_sender_warps * sizeof(vidType *), cudaMemcpyHostToDevice));

    end_counter = (int *)nvshmem_malloc(sizeof(int));
    int end_counter_h = (ndevices-1) * n_sender_warps;
    CUDA_SAFE_CALL(cudaMemcpy(end_counter, &end_counter_h, sizeof(int), cudaMemcpyHostToDevice));
}

__device__ vidType* RequestList::request(int sender_warp_id, int dest_id) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp

    // send request
    int mype_id_there = get_mype_id_there(dest_id);
    int *put_addr = get_remote_signal_slot(mype_id_there, sender_warp_id);
    if (thread_lane == 0) {
        nvshmem_int_p(put_addr, 1, dest_id);
    } __syncwarp();     // TODO: can we remove it?

    if (thread_lane == 0) {
        printf("[%d, %d]: put request to (%d, %d)\n", nvshmem_my_pe(), sender_warp_id, dest_id, (int)put_addr);
    } __syncwarp();     // TODO: can we remove it?

    // wait for server's response
    vidType *resp_addr = NULL;
    vidType **get_addr;
    do {
        nvshmem_fence();    // TODO: can we remove it? how to improve this?
        get_addr = get_remote_resp_slot(mype_id_there, sender_warp_id);
        nvshmem_get64((void *)&resp_addr, (void *)get_addr, 1, dest_id);
        // resp_addr = (vidType *)nvshmem_uint64_g((const uint64_t *)get_addr, dest_id);
    } while (resp_addr == NULL);

    if (thread_lane == 1) {
        printf("[%d, %d]: got address from (%d, %d), addr = %ld\n", nvshmem_my_pe(), sender_warp_id, dest_id, (int)get_addr, (long int)resp_addr);
    } __syncwarp();

    // reset resp slot
    vidType *null_p = NULL;
    if (thread_lane == 0) {
        nvshmem_put64(get_addr, &null_p, 1, dest_id);
    } __syncwarp();     // TODO: can we remove it?

    return resp_addr;
}

__device__ void RequestList::sender_quit() {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    if (thread_lane == 0) {
        for (int dest_id = 0; dest_id < ndevices; dest_id++) {
            if (dest_id == mype_id) continue;
            nvshmem_int_atomic_add(end_counter, -1, dest_id);
        }
    } __syncwarp();     // TODO: can we remove it?
}

__device__ int RequestList::check(int src_id, int sender_warp_id) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int src_id_here = get_src_id_here(src_id);

    int *signal_slot = get_signal_slot(src_id_here, sender_warp_id);
    __threadfence();    // TODO: can we remove it?
    if (*signal_slot == 1) {
        // reset signal_slot
        if (thread_lane == 0) {
            *signal_slot = 0;
        } __syncwarp();     // TODO: can we remove it?

        if (thread_lane == 0) {
            printf("[%d]: checked and found ([%d, %d], %d)\n", nvshmem_my_pe(), src_id, sender_warp_id, (int)signal_slot);
        } __syncwarp();

        return 1;
    }
    return 0;
}

__device__ void RequestList::respond(int src_id, int sender_warp_id, vidType *send_recv_slot) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int src_id_here = get_src_id_here(src_id);

    vidType **resp_slot = get_resp_slot(src_id_here, sender_warp_id);
    if (thread_lane == 0) {
        *resp_slot = send_recv_slot;
    } __syncwarp(); // TODO: can we remove it?

    if (thread_lane == 0) {
        printf("[%d]: respond to ([%d, %d], %d), addr = %d\n", nvshmem_my_pe(), src_id, sender_warp_id, (int)resp_slot, (int)send_recv_slot);
    } __syncwarp(); // TODO: can we remove it?
}

__device__ int RequestList::finished() {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    
    if (*end_counter == 0) {
        if (thread_lane == 0) {
            printf("finished!!!\n");
        } __syncwarp();

        return 1;
    }

    return 0;
}
