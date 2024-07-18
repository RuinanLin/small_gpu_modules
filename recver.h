#include "graph_nvshmem.h"
#include "common.h"
#include "send_recv_buffer.h"

// format:
// | type | u | u_degree | num_push_tasks | u_list | tasks |
#define TYPE_POS            0
#define U_POS               1
#define U_DEGREE_POS        2
#define NUM_PUSH_TASKS_POS  3
#define U_LIST_START        4

class Recver {
protected:
    GraphNVSHMEM g;
    SendRecvBuffer send_recv_buffer;

    int recver_warp_start_id;

public:
    Recver(GraphNVSHMEM _g, SendRecvBuffer _send_recv_buffer, int _recver_warp_start_id)
    : g(_g), send_recv_buffer(_send_recv_buffer), recver_warp_start_id(_recver_warp_start_id) {
    }

    __device__ int get_recver_warp_id() {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
        int warp_id = thread_id / WARP_SIZE;                    // global warp index
        return warp_id - recver_warp_start_id;
    }

    __device__ AccType do_msg_intersection(vidType *msg) {
        vidType u_degree = msg[U_DEGREE_POS];
        vidType num_push_tasks = msg[NUM_PUSH_TASKS_POS];
        vidType *u_list = &msg[U_LIST_START];
        vidType *tasks = &msg[U_LIST_START + u_degree];

        AccType result = 0;
        for (int n = 0; n < u_degree; n++) {
            vidType v = tasks[n];
            result += intersect_num(u_list, u_degree, g.N(v), g.get_degree(v));
        }
        return result;
    }

    __device__ AccType launch() {
        AccType count = 0;

        // get its warp_id in all recver warps
        int recver_warp_id = get_recver_warp_id();

        // start receiving messages
        while (1) {
            int slot_id_per_warp;
            vidType *msg = send_recv_buffer.check_msg(recver_warp_id, &slot_id_per_warp);
            if (msg != NULL) {
                // do set intersection
                count += do_msg_intersection(msg);

                // reset valid
                send_recv_buffer.turn_invalid(slot_id_per_warp, recver_warp_id);
            }
            if (send_recv_buffer.finished()) {
                return count;
            }
        }
    }
};
