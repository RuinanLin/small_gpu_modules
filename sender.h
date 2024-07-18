#include "graph_nvshmem.h"
#include "common.h"
#include "sender_local_push_buffer.h"
#include "sender_local_pull_buffer.h"
#include "request_list.h"

class Sender {
protected:
    GraphNVSHMEM g;
    int ndevices;
    vidType n_real_vertices;
    int u_partition_num;
    int stride;
    int sender_warp_start_id;

    SenderLocalPushBuffer sender_local_push_buffer;
    SenderLocalPullBuffer sender_local_pull_buffer;
    RequestList request_list;

public:
    Sender(GraphNVSHMEM _g, int _ndevices, int u_partition_num, int _stride, int _sender_warp_start_id, SenderLocalPushBuffer _sender_local_push_buffer, SenderLocalPullBuffer _sender_local_pull_buffer, RequestList _request_list)
    : g(_g), ndevices(_ndevices), u_partition_num(_u_partition_num), stride(_stride), sender_warp_start_id(_sender_warp_start_id), sender_local_push_buffer(_sender_local_push_buffer), sender_local_pull_buffer(_sender_local_pull_buffer), request_list(_request_list) {
        n_real_vertices = g.get_n_real_vertices();
    }

    __device__ int get_sender_warp_id() {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
        int warp_id = thread_id / WARP_SIZE;                    // global warp index
        return warp_id - sender_warp_start_id;
    }

    // TODO: design a cleverer data format to get rid of this stupid function
    __device__ vidType get_v_degree(vidType v, int v_partition_num) {
        eidType v_list_begin = nvshmem_int64_g(g.get_rowptr()+v, v_partition_num);
        eidType v_list_end = nvshmem_int64_g(g.get_rowptr()+v+1, v_partition_num);
        return v_list_end - v_list_begin;
    }

    __device__ AccType launch() {
        AccType count = 0;

        // get its warp_id in all sender warps
        int sender_warp_id = get_sender_warp_id();

        // start calculation
        for (int u_local_id = sender_warp_start_id; u_local_id < n_real_vertices; u_local_id += stride) {
            vidType u = g.get_vertex_in_vertex_list(u_local_id);
            vidType u_degree = g.get_degree(u);
            for (int n = 0; n < u_degree; n++) {
                vidType v = g.N(u, n);
                int v_partition_num = g.get_vertex_partition_number(v);
                vidType v_degree = get_v_degree(v, v_partition_num);    // TODO: design a cleverer data format to be able to look up the degree of remote vertices on local devices
                if (v_partition_num != u_partition_num) {
                    if (u_degree < v_degree) {  // push
                        update_info_t update_info = { .type = 0, .u = u, .u_degree = u_degree, .u_list = g.N(u), .v = v };
                        sender_local_push_buffer.update(update_info, sender_warp_id, v_partition_num);
                    } else {    // pull
                        // TODO: design a cleverer format to avoid reading this every time
                        eidType v_list_begin = nvshmem_int64_g(g.get_rowptr()+v, v_partition_num);
                        vidType *v_list = sender_local_pull_buffer.pull(sender_warp_id, g.get_colidx()+v_list_begin, v_degree, v_partition_num);
                        count += intersect_num(g.N(u), u_degree, v_list, v_degree);
                    }
                }
            }
            sender_local_push_buffer.send(sender_warp_id);
        }

        // send the finish signal
        request_list.sender_quit();

        return count;
    }
};
