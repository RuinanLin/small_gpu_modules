#include "common.h"

class RequestList{
protected:
    int ndevices;
    int n_sender_warps;
    int mype_id;

    int *signal;    // [ ndevices-1, n_sender_warps ]
    vidType **resp; // [ ndevices-1, n_sender_warps ]
    int *end_counter;   // a counter that counts down. when reaching 0, all the senders have ended.

public:
    RequestList(int _ndevices, int _n_sender_warps, int _mype_id)
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

    __device__ int get_mype_id_there(int dest_id) { return (mype_id < dest_id)? mype_id : (mype_id-1); }
    __device__ int get_dest_id_here(int dest_id) { return (dest_id < mype_id)? dest_id : (dest_id-1); }
    __device__ int *get_signal_slot(int src_id_here, int sender_warp_id) { return &signal[src_id_here * (ndevices-1) + sender_warp_id]; }
    __device__ vidType **get_resp_slot(int mype_id_there, int sender_warp_id) { return &resp[mype_id_there * (ndevices-1) + sender_warp_id]; }

    __device__ vidType *request(int sender_warp_id, int dest_id) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
        
        // send request
        int mype_id_there = get_mype_id_there(dest_id);
        int *put_addr = &signal[mype_id_there * n_sender_warps + sender_warp_id];
        if (thread_lane == 0) {
            nvshmem_int_p(put_addr, 1, dest_id);
        } __syncwarp();     // TODO: can we remove it?
        nvshmem_fence();    // TODO: can we remove it?

        // wait for server's response
        int dest_id_here = get_dest_id_here(dest_id);
        vidType *resp_addr;
        do {
            nvshmem_fence();
            resp_addr = resp[dest_id_here * n_sender_warps + sender_warp_id];
        } while (resp_addr != NULL);

        // reset resp slot
        resp[dest_id_here * n_sender_warps + sender_warp_id] = NULL;

        return resp_addr;
    }

    __device__ int check(int src_id, int sender_warp_id) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
        int src_id_here = get_dest_id_here(src_id);

        int *signal_slot = get_signal_slot(src_id_here, sender_warp_id);
        __threadfence();    // TODO: can we remove it?
        if (*signal_slot == 1) {
            // reset signal_slot
            if (thread_lane == 0) {
                *signal_slot = 0;
            } __syncwarp();     // TODO: can we remove it?

            return 1;
        }
        return 0;
    }

    __device__ void sender_quit() {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);

        if (thread_lane == 0) {
            for (int dest_id = 0; dest_id < ndevices; dest_id++) {
                if (dest_id == mype_id) continue;
                nvshmem_int_atomic_add(end_counter, -1, dest_id);
            }
        } __syncwarp();     // TODO: can we remove it?
    }

    __device__ void respond(int src_id, int sender_warp_id, vidType *send_recv_slot) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        int mype_id_there = get_mype_id_there(src_id);

        vidType **resp_slot = get_resp_slot(mype_id_there, sender_warp_id);
        if (thread_lane == 0) {
            *resp_slot = send_recv_slot;
            nvshmem_fence();
        } __syncwarp(); // TODO: can we remove it?
    }

    __device__ int finished() {
        nvshmem_fence();
        if (*end_counter == 0) {
            return 1;
        }
        return 0;
    }
};
