#include "graph_nvshmem.h"
#include "common.h"

class Worker {
protected:
    GraphNVSHMEM g;
    vidType n_real_vertices;
    int u_partition_num;
    int stride;
    int worker_warp_start_id;

public:
    Worker(GraphNVSHMEM _g, int _stride, int _worker_warp_start_id)
    : g(_g), u_partition_num(_u_partition_num), stride(_stride), worker_warp_start_id(_worker_warp_start_id) {
        n_real_vertices = g.get_n_real_vertices();
    }

    __device__ int get_worker_warp_id() {
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
        int warp_id = thread_id / WARP_SIZE;                    // global warp index
        return warp_id - worker_warp_start_id;
    }

    __device__ int is_local(vidType v) {
        if (g.get_vertex_partition_number(v) == u_partition_num) {
            return 1;
        } else {
            return 0;
        }
    }

    __device__ AccType launch() {
        AccType count = 0;

        // get its warp_id in all worker warps
        int worker_warp_id = get_worker_warp_id();

        // start calculation
        for (int u_local_id = worker_warp_id; u_local_id < n_real_vertices; u_local_id += stride) {
            vidType u = g.get_vertex_in_vertex_list(u_local_id);
            vidType u_degree = g.get_degree(u);
            for (int n = 0; n < u_degree; n++) {
                vidType v = g.N(u, n);
                if (is_local(v)) {
                    count += intersect_num(g.N(u), u_degree, g.N(v), g.get_degree(v));
                }
            }
        }
        return count;
    }
};