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
                    vidType *resp_addr = send_recv_buffer.find_empty();
                    request_list.respond(src_id, sender_warp_id, resp_addr);
                }
            }
        }
        // terminate
        if (request_list.finished() == 1) {
            send_recv_buffer.server_quit();
            break;
        }
    }
}

__device__ void recver_launch(SendRecvBuffer send_recv_buffer) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int recver_warp_id = threadIdx.x / WARP_SIZE - 1;
    
    while (1) {
        // check whether server has finished
        int finished = 0;
        if (send_recv_buffer.finished() == 1) {
            finished = 1;
        }
        // check a round
        vidType *msg;
        for (int slot_id_per_warp = 0; slot_id_per_warp < N_SLOTS_PER_WARP; slot_id_per_warp++) {
            msg = send_recv_buffer.check_msg(recver_warp_id, slot_id_per_warp);
            if (msg) {
                if (thread_lane == 2) {
                    printf("recver warp [%d] got from slot_id_per_warp = %d, pe[%d], sender_warp[%d]\n", recver_warp_id, slot_id_per_warp, msg[1], msg[2]);
                } __syncwarp();

                send_recv_buffer.turn_invalid(slot_id_per_warp, recver_warp_id);
                if (thread_lane == 0) {
                    printf("recver warp [%d] turned slot %d invalid\n", recver_warp_id, slot_id_per_warp);
                } __syncwarp();
            }
        }
        // terminate
        if (finished == 1) {
            return;
        }
    }
}

__device__ void sender_launch(RequestList request_list) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int sender_warp_id = threadIdx.x / WARP_SIZE;
    for (int round = 0; round < N_REQUESTS_EACH_WARP; round++) {
        // destination pe: 0
        // get address
        vidType *send_addr = request_list.request(sender_warp_id, 0);
        // put message
        nvshmem_int32_p(send_addr+1, nvshmem_my_pe(), 0);
        nvshmem_int32_p(send_addr+2, sender_warp_id, 0);
        nvshmem_fence();    // TODO: can we remove it?
        // put valid
        nvshmem_int32_p(send_addr, 1, 0);
        nvshmem_fence();    // TODO: can we remove it?
        if (thread_lane == 3) {
            printf("sender warp [%d, %d] put message to %d, round: %d\n", nvshmem_my_pe(), sender_warp_id, (int)send_addr, round);
        } __syncwarp();
    }
    // quit
    request_list.sender_quit();
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
