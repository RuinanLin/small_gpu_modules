#include "request_list.h"
#include "send_recv_buffer.h"
#include <iostream>

// format:
// valid || type | u | u_degree | num_push_tasks | u_list | tasks |

#define BLOCK_SIZE  256
#define WARP_SIZE   32

#define N_REQUESTS_EACH_WARP    2

#define SLOT_SIZE   2
#define N_SLOTS_PER_WARP    5

__device__ void server_launch(RequestList request_list, SendRecvBuffer send_recv_buffer) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int n_warps_each_pe = 1 * BLOCK_SIZE / WARP_SIZE;
    int npes = nvshmem_n_pes();

    while (1) {
        // check and respond
        for (int src_id = 1; src_id < npes; src_id++) {
            for (int sender_warp_id = 0; sender_warp_id < n_warps_each_pe; sender_warp_id++) {
                if (request_list.check(src_id, sender_warp_id) == 1) {
                    if (thread_lane == 0) {
                        printf("server got request from [%d, %d]\n", src_id, sender_warp_id);
                    } __syncwarp();

                    vidType *resp_addr = send_recv_buffer.find_empty();
                    if (thread_lane == 0) {
                        printf("server found empty address %d\n", resp_addr);
                    } __syncwarp();

                    request_list.respond(src_id, sender_warp_id, resp_addr);
                    if (thread_lane == 0) {
                        printf("server respond to [%d, %d] with resp_addr %d\n", src_id, sender_warp_id, resp_addr);
                    } __syncwarp();
                }
            }
        }

        // terminate
        if (request_list.finished() == 1) {
            send_recv_buffer.server_quit();
            if (thread_lane == 0) {
                printf("server quit\n");
            } __syncwarp();
            break;
        }
    }
}

__device__ void recver_launch(SendRecvBuffer send_recv_buffer) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int recver_warp_id = threadIdx.x / WARP_SIZE - 1;
    
    while (1) {
        int slot_id_per_warp;
        vidType *msg = send_recv_buffer.check_msg(recver_warp_id, &slot_id_per_warp);
        if (msg) {
            if (thread_lane == 2) {
                printf("recver warp [%d] got from slot_id_per_warp = %d, pe[%d], sender_warp[%d]\n", recver_warp_id, slot_id_per_warp, msg[1], msg[2]);
            } __syncwarp();

            send_recv_buffer.turn_invalid(slot_id_per_warp, recver_warp_id);
            if (thread_lane == 0) {
                printf("recver warp [%d] turned slot %d invalid\n", recver_warp_id, slot_id_per_warp);
            } __syncwarp();
        }

        // check whether finished
        if (send_recv_buffer.finished() == 1) {
            if (thread_lane == 0) {
                printf("recver warp [%d] quit\n", recver_warp_id);
            } __syncwarp();
            return;
        }
    }
}

__device__ void sender_launch(RequestList request_list) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int sender_warp_id = threadIdx.x / WARP_SIZE;
    
    // make a request and send a message
    for (int r = 0; r < N_REQUESTS_EACH_WARP; r++) {
        // destination pe: 0
        // get address
        vidType *send_addr = request_list.request(sender_warp_id, 0);
        if (thread_lane == 0) {
            printf("sender warp [%d, %d] got send_addr = %d\n", nvshmem_my_pe(), sender_warp_id, send_addr);
        } __syncwarp();
        // put message
        nvshmem_int32_p(send_addr+1, nvshmem_my_pe(), 0);
        nvshmem_int32_p(send_addr+2, sender_warp_id, 0);
        nvshmem_fence();    // TODO: can we remove it?
        // put valid
        nvshmem_int32_p(send_addr, 1, 0);
        nvshmem_fence();    // TODO: can we remove it?
        if (thread_lane == 0) {
            printf("sender warp [%d, %d] put message to %d\n", nvshmem_my_pe(), sender_warp_id, send_addr);
        } __syncwarp();
    }

    // quit
    request_list.sender_quit();
    if (thread_lane == 0) {
        printf("sender warp [%d, %d] quit\n", nvshmem_my_pe(), sender_warp_id);
    } __syncwarp();
}

__global__ void test_send_recv_buffer(RequestList request_list, SendRecvBuffer send_recv_buffer) {
    int mype = nvshmem_my_pe();
    int warp_id = threadIdx.x / WARP_SIZE;

    if (mype == 0) {
        if (warp_id == 0) {
            server_launch(request_list, send_recv_buffer);
        } else {
            recver_launch(send_recv_buffer);
        }
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

    int mype = nvshmem_my_pe();

    // initialize request_list
    RequestList request_list(ndevices, BLOCK_SIZE/WARP_SIZE, mype);
    
    // initialize send_recv_buffer
    SendRecvBuffer send_recv_buffer(ndevices, SLOT_SIZE, N_SLOTS_PER_WARP, BLOCK_SIZE / WARP_SIZE - 1);

    cudaDeviceSynchronize();    // TODO: can we remove it?
    nvshmem_barrier_all();      // TODO: can we remove it?

    // launch kernel
    // warps in device 1~3 are senders
    // warp 0 in device 0 is a server
    // warp 1~7 in device 0 are recvers
    test_send_recv_buffer<<<1, BLOCK_SIZE>>>(request_list, send_recv_buffer);
    cudaDeviceSynchronize();    // TODO: can we remove it?

    return 0;
}
