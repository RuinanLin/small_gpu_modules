#include "common.h"

class Server {
protected:
    int ndevices;
    int n_sender_warps;
    int mype_id;

    RequestList request_list;
    SendRecvBuffer send_recv_buffer;

public:
    Server(int _ndevices, int _n_sender_warps, int _mype_id, RequestList _request_list)
    : ndevices(_ndevices), n_sender_warps(_n_sender_warps), mype_id(_mype_id), request_list(_request_list) { }

    __device__ int get_src_id_here(int src_id) { return (src_id < mype_id)? src_id : (src_id-1); }

    __device__ void launch() {
        while (1) {
            for (int src_id = 0; src_id < ndevices; src_id++) {
                if (src_id == mype_id) continue;
                for (int sender_warp_id = 0; sender_warp_id < n_sender_warps; sender_warp_id++) {
                    if (request_list.check(src_id, sender_warp_id) == 1) {
                        vidType *send_recv_slot = send_recv_buffer.find_empty();
                        request_list.respond(src_id, sender_warp_id, send_recv_slot);
                    }
                }
            }
            if (request_list.finished() == 1) {
                send_recv_buffer.server_quit();
                return;
            }
        }
    }
};
