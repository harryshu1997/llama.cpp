#define main resident_router_main
#include "examples/layersplit/ffn-split-resident-router.cpp"
#undef main

#include <cassert>
#include <thread>

class worker {
public:
    worker(uint64_t mask, const std::string & fault) {
        listener = socket(AF_INET, SOCK_STREAM, 0);
        assert(listener >= 0);
        sockaddr_in address = {};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        assert(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0);
        socklen_t size = sizeof(address);
        assert(getsockname(listener, reinterpret_cast<sockaddr *>(&address), &size) == 0);
        port = ntohs(address.sin_port);
        assert(listen(listener, 1) == 0);
        thread = std::thread([this, mask, fault]() {
            const int fd = accept(listener, nullptr, nullptr);
            if (fd < 0) {
                return;
            }
            ffn_split::hello_request request = {};
            assert(receive_exact(fd, &request, sizeof(request)));
            ffn_split::hello_response response = {};
            response.magic = request.magic;
            response.version = request.version;
            response.message = static_cast<uint16_t>(ffn_split::message_type::hello_response);
            response.flags = request.flags;
            response.layer_mask = mask;
            response.layer_count = __builtin_popcountll(mask);
            response.n_embd = request.n_embd;
            response.max_columns = request.max_columns;
            response.n_ff = request.max_columns;
            response.max_tokens = request.max_tokens;
            response.column_quantum = 32;
            response.weight_hash = mask + 1;
            std::memcpy(response.artifact_sha256, request.artifact_sha256, 32);
            if (fault == "artifact") {
                response.artifact_sha256[0] ^= 1;
            } else if (fault == "geometry") {
                ++response.n_embd;
            } else if (fault == "width") {
                --response.max_columns;
            } else if (fault == "mask") {
                response.layer_mask = mask << 1;
            }
            assert(send_exact(fd, &response, sizeof(response)));
            close(fd);
        });
    }

    ~worker() {
        shutdown(listener, SHUT_RDWR);
        thread.join();
        close(listener);
    }

    int port = 0;

private:
    int listener = -1;
    std::thread thread;
};

int main(int argc, char ** argv) {
    assert(argc == 2);
    const std::string test = argv[1];
    const std::string artifact = "sha256:" + std::string(64, 'a');
    worker first(3, test);
    worker second(12, "");
    std::vector<ffn_split::resident_session_shard> shards(3);
    for (size_t index = 0; index < shards.size(); ++index) {
        auto & shard = shards[index];
        shard.session_id = "session-" + std::to_string(index);
        shard.artifact_sha256 = artifact;
        shard.layer_mask = UINT64_C(3) << (index * 2);
        shard.columns = 128;
        shard.session_generation = index + 1;
    }
    shards[0].port = first.port;
    shards[1].port = second.port;
    // The unrequested third session has no listening worker while loading.
    shards[2].port = 0;
    ffn_split::hello_request hello = {};
    hello.magic = ffn_split::protocol_magic;
    hello.version = ffn_split::protocol_version;
    hello.message = static_cast<uint16_t>(ffn_split::message_type::hello_request);
    hello.flags = ffn_split::flag_f16_io | ffn_split::flag_swiglu;
    hello.n_embd = 64;
    hello.max_columns = 128;
    hello.max_tokens = 4;
    hello.layer_mask = test == "one" ? 3 : test == "partial" ? 1 :
            test == "missing" ? 63 : 15;
    assert(ffn_split::parse_artifact_sha256(artifact, hello.artifact_sha256));
    target_selection selected;
    std::vector<session_proof_totals> proofs;
    const bool accepted = select_targets(shards, hello, selected, proofs);
    const bool expected = test == "subset" || test == "one";
    assert(accepted == expected);
    if (accepted) {
        const size_t count = test == "one" ? 1 : 2;
        assert(selected.targets.size() == count);
        assert(proofs.size() == count);
        assert(selected.response.layer_mask == hello.layer_mask);
        for (size_t index = 0; index < count; ++index) {
            assert(proofs[index].shard.session_generation == index + 1);
        }
    }
    close_selection(selected);
    return 0;
}
