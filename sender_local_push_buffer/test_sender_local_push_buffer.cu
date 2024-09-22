#include "request_list.h"
#include "send_recv_buffer.h"
#include "sender_local_push_buffer.h"
#include <iostream>

// format:
// valid || type | u | u_degree | num_push_tasks | u_list | tasks |

#define BLOCK_SIZE  256
#define WARP_SIZE   32

#define N_REQUESTS_EACH_WARP    1

#define SLOT_SIZE   20
#define N_SLOTS_PER_WARP    5

__device__ void server_launch(RequestList request_list, SendRecvBuffer send_recv_buffer) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int n_warps_each_pe = 1 * BLOCK_SIZE / WARP_SIZE;
    int npes = nvshmem_n_pes();

    if (thread_lane == 0) {
        printf("server entered.\n");
    } __syncwarp();

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
            if (thread_lane == 0) {
                printf("server quit.\n");
            } __syncwarp();
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
                    printf("recver warp [%d] got msg, u = %d, u_degree = %d, num_push_tasks = %d\n", recver_warp_id, msg[2], msg[3], msg[4]);
                } __syncwarp();
                send_recv_buffer.turn_invalid(slot_id_per_warp, recver_warp_id);
            }
        }
        // terminate
        if (finished == 1) {
            if (thread_lane == 0) {
                printf("recver warp [%d] quit.\n", recver_warp_id);
            } __syncwarp();
            return;
        }
    }
}

__device__ void sender_launch(RequestList request_list, SenderLocalPushBuffer sender_local_push_buffer, vidType* u_list) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int sender_warp_id = threadIdx.x / WARP_SIZE;
    if (thread_lane == 0) {
        printf("sender [%d, %d] entered.\n", nvshmem_my_pe(), sender_warp_id);
    } __syncwarp();
    for (int round = 0; round < N_REQUESTS_EACH_WARP; round++) {
        // destination pe: 0
        // prepare update information
        update_info_t update_info;
        update_info.type = 0;
        update_info.u = nvshmem_my_pe() * sender_warp_id;
        update_info.u_degree = (nvshmem_my_pe() * sender_warp_id) % 5 + 3;
        vidType* u_list_for_this_warp = &u_list[sender_warp_id * 7];
        if (thread_lane == 0) {
            for (int i = 0; i < update_info.u_degree; i++) {
                u_list_for_this_warp[i] = i;
            }
        } __syncwarp();     // TODO
        __threadfence();    // TODO
        update_info.u_list = u_list_for_this_warp;
        for (int i = 0; i < 3; i++) {
            update_info.v = i;
            if (thread_lane == 0) {
                printf("sender [%d, %d] going to update %d.\n", nvshmem_my_pe(), sender_warp_id, i);
            } __syncwarp();
            sender_local_push_buffer.update(update_info, sender_warp_id, 0);
            if (thread_lane == 0) {
                printf("sender [%d, %d] update %d.\n", nvshmem_my_pe(), sender_warp_id, i);
            } __syncwarp();
        }
        sender_local_push_buffer.send(sender_warp_id, request_list);
    }
    // quit
    request_list.sender_quit();
    if (thread_lane == 0) {
        printf("sender [%d, %d] quit.\n", nvshmem_my_pe(), sender_warp_id);
    } __syncwarp();
}

__global__ void test_send_recv_buffer(RequestList request_list, SendRecvBuffer send_recv_buffer, SenderLocalPushBuffer sender_local_push_buffer, vidType* u_list) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int mype = nvshmem_my_pe();
    if (mype == 0) {
        if (warp_id == 0) {
            server_launch(request_list, send_recv_buffer);
        } else {
            recver_launch(send_recv_buffer);
        }
    } else {
        sender_launch(request_list, sender_local_push_buffer, u_list);
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

    // initialize gpu memory for each sender
    vidType *u_list;
    CUDA_SAFE_CALL(cudaMalloc((void**)&u_list, 8*7*sizeof(vidType)));

    // components
    RequestList request_list(ndevices, BLOCK_SIZE/WARP_SIZE, mype);
    SendRecvBuffer send_recv_buffer(ndevices, SLOT_SIZE, N_SLOTS_PER_WARP, BLOCK_SIZE/WARP_SIZE-1);
    SenderLocalPushBuffer sender_local_push_buffer(ndevices, SLOT_SIZE, BLOCK_SIZE/WARP_SIZE, mype);

    cudaDeviceSynchronize();    // TODO: can we remove it?
    nvshmem_barrier_all();      // TODO: can we remove it?

    // launch kernel
    // warps in device 1~3 are senders
    // warp 0 in device 0 is a server
    // warp 1~7 in device 0 are recvers
    test_send_recv_buffer<<<1, BLOCK_SIZE>>>(request_list, send_recv_buffer, sender_local_push_buffer, u_list);
    cudaDeviceSynchronize();    // TODO: can we remove it?

    return 0;
}
