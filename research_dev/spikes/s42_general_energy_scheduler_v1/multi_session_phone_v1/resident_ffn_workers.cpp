#include "ffn-split-worker-entry.h"
#include "ggml-backend.h"

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

constexpr const char * layout_balanced = "gemma23-qwen12-full-v1";
constexpr const char * layout_gemma_heavy = "gemma46-qwen6-full-v1";

bool parse_port(const char * text, int & port) {
    errno = 0;
    char * end = nullptr;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value <= 0 || value > 65535) {
        return false;
    }
    port = static_cast<int>(value);
    return true;
}

bool parse_layer_spec(const char * text, uint64_t & mask) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    uint64_t parsed = 0;
    std::string remaining(text);
    while (!remaining.empty()) {
        const size_t comma = remaining.find(',');
        const std::string item = remaining.substr(0, comma);
        const size_t dash = item.find('-');
        const std::string first_text = item.substr(0, dash);
        const std::string last_text = dash == std::string::npos ?
                first_text : item.substr(dash + 1);
        errno = 0;
        char * first_end = nullptr;
        char * last_end = nullptr;
        const long first = strtol(first_text.c_str(), &first_end, 10);
        const long last = strtol(last_text.c_str(), &last_end, 10);
        if (errno != 0 || first_end == first_text.c_str() ||
            last_end == last_text.c_str() || *first_end != '\0' ||
            *last_end != '\0' || first < 0 || last < first || last >= 64) {
            return false;
        }
        for (long layer = first; layer <= last; ++layer) {
            parsed |= UINT64_C(1) << layer;
        }
        if (comma == std::string::npos) {
            break;
        }
        remaining.erase(0, comma + 1);
    }
    mask = parsed;
    return mask != 0;
}

bool port_ready(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ready = fd >= 0 &&
            connect(fd, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) == 0;
    if (fd >= 0) {
        close(fd);
    }
    return ready;
}

bool wait_ready(int port, const std::atomic<int> & status) {
    const auto deadline = steady_clock::now() + std::chrono::minutes(6);
    while (steady_clock::now() < deadline) {
        if (port_ready(port)) {
            return true;
        }
        if (status.load() != INT_MIN) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

std::thread launch_worker(
        std::string label,
        std::string model,
        std::string backend,
        std::string layers,
        std::string columns,
        std::string column_quantum,
        std::string bind,
        int port,
        std::atomic<int> & status) {
    return std::thread([
            label = std::move(label), model = std::move(model),
            backend = std::move(backend), layers = std::move(layers),
            columns = std::move(columns),
            column_quantum = std::move(column_quantum),
            bind = std::move(bind),
            port, &status]() mutable {
        std::vector<std::string> arguments = {
            std::move(label),
            "-m", std::move(model),
            "--layers", std::move(layers),
            "--columns", std::move(columns),
            "--backend", std::move(backend),
            "--bind", std::move(bind),
            "--port", std::to_string(port),
            "--max-tokens", "512",
            "--column-quantum", std::move(column_quantum),
            "--max-requests", "0",
            "--f16-io",
        };
        std::vector<char *> argv;
        argv.reserve(arguments.size());
        for (std::string & value : arguments) {
            argv.push_back(value.data());
        }
        status.store(ffn_split_worker_main(
                static_cast<int>(argv.size()), argv.data()));
    });
}

[[noreturn]] void fail(const char * message) {
    fprintf(stderr, "[resident-workers] %s\n", message);
    fflush(stderr);
    std::_Exit(1);
}

int run_qwen_only(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s <qwen-model> <qwen-low-port> "
                "<qwen-high-port>\n",
                argv[0]);
        return 2;
    }
    int qwen_low_port = 0;
    int qwen_high_port = 0;
    if (!parse_port(argv[2], qwen_low_port) ||
        !parse_port(argv[3], qwen_high_port) ||
        qwen_low_port == qwen_high_port) {
        fprintf(stderr, "[resident-workers] invalid Qwen worker ports\n");
        return 2;
    }

    ggml_backend_load_all();
    std::atomic<int> qwen_low_status(INT_MIN);
    std::atomic<int> qwen_high_status(INT_MIN);
    std::thread qwen_low = launch_worker(
            "resident-qwen-low", argv[1], "HTP0", "0-5", "17408",
            "17408", "127.0.0.1", qwen_low_port, qwen_low_status);
    if (!wait_ready(qwen_low_port, qwen_low_status)) {
        fail("HTP0 Qwen low-layer worker failed to become warm");
    }
    fprintf(stderr, "[resident-workers] HTP0 Qwen layers 0-5 WARM\n");
    fflush(stderr);

    std::thread qwen_high = launch_worker(
            "resident-qwen-high", argv[1], "HTP1", "6-11", "17408",
            "17408", "127.0.0.1", qwen_high_port, qwen_high_status);
    if (!wait_ready(qwen_high_port, qwen_high_status)) {
        fail("HTP1 Qwen high-layer worker failed to become warm");
    }
    fprintf(stderr,
            "RESIDENTWORKERS {\"status\":\"WARM\","
            "\"layout\":\"qwen12-full-v1\","
            "\"sessions\":[\"HTP0\",\"HTP1\"],"
            "\"qwen_layers\":\"0-11\","
            "\"qwen_columns\":17408,"
            "\"resident_mib_approx\":6120}\n");
    fflush(stderr);

    qwen_low.join();
    qwen_high.join();
    return qwen_low_status.load() == 0 && qwen_high_status.load() == 0 ?
            0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 4) {
        return run_qwen_only(argc, argv);
    }
    if (argc != 6 && argc != 8) {
        fprintf(stderr,
                "usage: %s <gemma-model> <qwen-model> "
                "[<llama-model>] <gemma-port> <qwen-low-port> "
                "<qwen-high-port> [<llama-port>]\n",
                argv[0]);
        return 2;
    }
    const bool llama_enabled = argc == 8;
    const char * layout_value = getenv("S42_RESIDENT_LAYOUT");
    const std::string layout = layout_value != nullptr ?
            layout_value : layout_balanced;
    if (layout != layout_balanced && layout != layout_gemma_heavy) {
        fprintf(stderr, "[resident-workers] invalid layout: %s\n",
                layout.c_str());
        return 2;
    }
    int gemma_port = 0;
    int qwen_low_port = 0;
    int qwen_high_port = 0;
    int llama_port = 0;
    const int port_offset = llama_enabled ? 4 : 3;
    if (!parse_port(argv[port_offset], gemma_port) ||
        !parse_port(argv[port_offset + 1], qwen_low_port) ||
        !parse_port(argv[port_offset + 2], qwen_high_port) ||
        (llama_enabled && !parse_port(argv[7], llama_port)) ||
        gemma_port == qwen_low_port || gemma_port == qwen_high_port ||
        qwen_low_port == qwen_high_port ||
        (llama_enabled &&
         (llama_port == gemma_port || llama_port == qwen_low_port ||
          llama_port == qwen_high_port))) {
        fprintf(stderr, "[resident-workers] invalid worker ports\n");
        return 2;
    }

    ggml_backend_load_all();
    std::atomic<int> first_status(INT_MIN);
    std::atomic<int> second_status(INT_MIN);
    std::atomic<int> third_status(INT_MIN);
    std::thread first = launch_worker(
            "resident-gemma-low", argv[1], "HTP0", "0-22", "6144", "512",
            "127.0.0.1", gemma_port, first_status);
    if (!wait_ready(gemma_port, first_status)) {
        fail("HTP0 Gemma low-layer worker failed to become warm");
    }
    fprintf(stderr, "[resident-workers] HTP0 Gemma layers 0-22 WARM\n");
    fflush(stderr);

    const bool gemma_heavy = layout == layout_gemma_heavy;
    std::thread second = launch_worker(
            gemma_heavy ? "resident-gemma-high" : "resident-qwen-low",
            gemma_heavy ? argv[1] : argv[2], "HTP1",
            gemma_heavy ? "23-45" : "0-5",
            gemma_heavy ? "6144" : "17408",
            gemma_heavy ? "512" : "17408",
            "127.0.0.1", qwen_low_port, second_status);
    if (!wait_ready(qwen_low_port, second_status)) {
        fail(gemma_heavy ?
                "HTP1 Gemma high-layer worker failed to become warm" :
                "HTP1 Qwen low-layer worker failed to become warm");
    }
    fprintf(stderr, gemma_heavy ?
            "[resident-workers] HTP1 Gemma layers 23-45 WARM\n" :
            "[resident-workers] HTP1 Qwen layers 0-5 WARM\n");
    fflush(stderr);

    std::thread third = launch_worker(
            gemma_heavy ? "resident-qwen-low" : "resident-qwen-high",
            argv[2], "HTP2", gemma_heavy ? "0-5" : "6-11",
            "17408", "17408", "127.0.0.1", qwen_high_port, third_status);
    if (!wait_ready(qwen_high_port, third_status)) {
        fail(gemma_heavy ?
                "HTP2 Qwen low-layer worker failed to become warm" :
                "HTP2 Qwen high-layer worker failed to become warm");
    }
    fprintf(stderr, gemma_heavy ?
            "[resident-workers] HTP2 Qwen layers 0-5 WARM\n" :
            "[resident-workers] HTP2 Qwen layers 6-11 WARM\n");
    std::atomic<int> llama_status(INT_MIN);
    std::thread llama;
    std::string llama_layers;
    int llama_mib_approx = 0;
    if (llama_enabled) {
        const char * llama_layers_value = getenv("S42_LLAMA_FFN_LAYERS");
        llama_layers = llama_layers_value != nullptr ?
                llama_layers_value : "0-15";
        uint64_t llama_layer_mask = 0;
        if (!parse_layer_spec(llama_layers.c_str(), llama_layer_mask)) {
            fail("invalid Llama resident layer specification");
        }
        llama_mib_approx =
                (453 * __builtin_popcountll(llama_layer_mask) + 15) / 16;
        llama = launch_worker(
                "resident-llama-ffn", argv[3], "HTP3", llama_layers, "8192",
                "1024", "0.0.0.0", llama_port, llama_status);
        if (!wait_ready(llama_port, llama_status)) {
            fail("HTP3 Llama FFN worker failed to become warm");
        }
        fprintf(stderr, "[resident-workers] HTP3 Llama layers %s WARM\n",
                llama_layers.c_str());
    }
    fprintf(stderr,
            "RESIDENTWORKERS {\"status\":\"WARM\"," 
            "\"layout\":\"%s\"," 
            "\"sessions\":[\"HTP0\",\"HTP1\",\"HTP2\"%s],"
            "\"gemma_layers\":\"%s\",\"qwen_layers\":\"%s\"," 
            "\"llama_layers\":\"%s\",\"qwen_columns\":17408,"
            "\"resident_mib_approx\":%d}\n",
            layout.c_str(), llama_enabled ? ",\"HTP3\"" : "",
            gemma_heavy ? "0-45" : "0-22",
            gemma_heavy ? "0-5" : "0-11",
            llama_layers.c_str(),
            (gemma_heavy ? 9270 : 9225) + llama_mib_approx);
    fflush(stderr);

    first.join();
    second.join();
    third.join();
    if (llama_enabled) {
        llama.join();
    }
    return first_status.load() == 0 && second_status.load() == 0 &&
            third_status.load() == 0 &&
            (!llama_enabled || llama_status.load() == 0) ? 0 : 1;
}
