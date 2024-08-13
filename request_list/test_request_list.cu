#include "request_list.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <iostream>

#define BLOCK_SIZE  256
#define WARP_SIZE   32

#define N_REQUESTS_EACH_WARP    100

__device__ void server_launch(RequestList request_list) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    int n_warps_each_pe = 1 * BLOCK_SIZE / WARP_SIZE;
    int npes = nvshmem_n_pes();

    // vidType *resp_addr = (vidType *)36;
    vidType *resp_addr = new vidType[16];

    int warp_id = threadIdx.x / WARP_SIZE;
    if (warp_id == 0) {
        while (1) {
            for (int src_id = 1; src_id < npes; src_id++) {
                for (int sender_warp_id = 0; sender_warp_id < n_warps_each_pe; sender_warp_id++) {
                    if (request_list.check(src_id, sender_warp_id) == 1) {
                        request_list.respond(src_id, sender_warp_id, resp_addr);
                        resp_addr++;
                    }
                }
            }
            if (request_list.finished() == 1) {
                if (thread_lane == 0) {
                    printf("Server finished.\n");
                } __syncwarp(); // TODO: can we remove it?

                break;
            }
        }
    }
}

__device__ void sender_launch(RequestList request_list) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    
    int sender_warp_id = threadIdx.x / WARP_SIZE;
    vidType *send_addr;
    for (int r = 0; r < N_REQUESTS_EACH_WARP; r++) {
        send_addr = request_list.request(sender_warp_id, 0);

        if (thread_lane == 0) {
            printf("[%d, %d]: got send_addr = %d, request_round = %d\n", nvshmem_my_pe(), sender_warp_id, (int)send_addr, r);
        } __syncwarp(); // TODO: can we remove it?
    }

    if (thread_lane == 1) {
        printf("[%d, %d]: sender quit\n", nvshmem_my_pe(), sender_warp_id);
    } __syncwarp();

    request_list.sender_quit();
}

__global__ void test_request_list(RequestList request_list) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    
    int mype = nvshmem_my_pe();
    if (mype == 0) {
        server_launch(request_list);
    } else {
        sender_launch(request_list);
    }
}

int main() {
    // initialize nvshmem
    nvshmem_init();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    std::cout << "mype_node = " << mype_node << "\n";
    cudaSetDevice(mype_node);

    // get device information
    int ndevices;
    cudaGetDeviceCount(&ndevices);

    // initialize request_list
    int mype = nvshmem_my_pe();
    RequestList request_list(ndevices, BLOCK_SIZE / WARP_SIZE, mype);
    cudaDeviceSynchronize();    // TODO: can we remove this?
    nvshmem_barrier_all();      // TODO: can we remove this?

    // launch kernel
    test_request_list<<<1, BLOCK_SIZE>>>(request_list);     // warps in device 1~3 are senders. warp 0 in device 0 deals with the request.
    cudaDeviceSynchronize();    // TODO: can we remove this?

    return 0;
}
