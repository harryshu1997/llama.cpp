#include "../aoa_async_transport_v1/libusb_abi.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static constexpr uint16_t AOA_VENDOR_ID = 0x18d1;
static constexpr uint16_t AOA_PRODUCT_IDS[] = {
    0x2d00, 0x2d01, 0x2d04, 0x2d05,
};
static constexpr unsigned char OUT_ENDPOINT = 0x01;
static constexpr unsigned char IN_ENDPOINT = 0x81;
static constexpr size_t TRANSFER_BYTES = 64U * 1024U;
static constexpr unsigned int OUT_TIMEOUT_MS = 10000;
static constexpr unsigned int IN_TIMEOUT_MS = 250;
static constexpr int LIBUSB_ERROR_TIMEOUT_VALUE = -7;

struct bridge_context {
    libusb_device_handle * usb = nullptr;
    int socket = -1;
    std::atomic<bool> stopping{false};
    std::atomic<int> failure{0};
    uint64_t usb_out_bytes = 0;
    uint64_t usb_in_bytes = 0;
    uint64_t usb_out_transfers = 0;
    uint64_t usb_in_transfers = 0;
};

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int parse_port(const char * text) {
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 1 || value > 65535) {
        throw std::runtime_error("invalid TCP port");
    }
    return (int) value;
}

static libusb_device_handle * open_accessory(
        libusb_context * context, uint16_t & selected_product) {
    for (unsigned int attempt = 0; attempt < 100; ++attempt) {
        for (const uint16_t product : AOA_PRODUCT_IDS) {
            libusb_device_handle * handle = libusb_open_device_with_vid_pid(
                context, AOA_VENDOR_ID, product);
            if (handle != nullptr) {
                selected_product = product;
                return handle;
            }
        }
        usleep(100000);
    }
    return nullptr;
}

static int create_listener(int port) {
    const int descriptor = socket(AF_INET, SOCK_STREAM, 0);
    if (descriptor < 0) {
        throw std::runtime_error(std::string("socket failed: ") +
                                 std::strerror(errno));
    }
    const int enabled = 1;
    setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t) port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(descriptor, (sockaddr *) &address, sizeof(address)) != 0 ||
            listen(descriptor, 1) != 0) {
        const std::string message = std::strerror(errno);
        close(descriptor);
        throw std::runtime_error("listen failed: " + message);
    }
    return descriptor;
}

static bool send_socket_all(int descriptor, const unsigned char * data,
        size_t size) {
    while (size > 0) {
        const ssize_t count = send(descriptor, data, size, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        data += count;
        size -= (size_t) count;
    }
    return true;
}

static void stop_bridge(bridge_context & context, int failure) {
    if (failure != 0) {
        int expected = 0;
        context.failure.compare_exchange_strong(expected, failure);
    }
    context.stopping.store(true);
    shutdown(context.socket, SHUT_RDWR);
}

static bool transfer_out(bridge_context & context, unsigned char * data,
        size_t size) {
    size_t offset = 0;
    while (offset < size) {
        int transferred = 0;
        const int status = libusb_bulk_transfer(
            context.usb, OUT_ENDPOINT, data + offset,
            (int) (size - offset), &transferred, OUT_TIMEOUT_MS);
        if (status != 0 || transferred <= 0) {
            if (!context.stopping.load()) {
                std::fprintf(stderr, "AOA OUT failed: %s transferred=%d\n",
                             libusb_error_name(status), transferred);
            }
            return false;
        }
        offset += (size_t) transferred;
        context.usb_out_bytes += (uint64_t) transferred;
        ++context.usb_out_transfers;
    }
    return true;
}

static void socket_to_usb(bridge_context & context) {
    std::vector<unsigned char> buffer(TRANSFER_BYTES);
    while (!context.stopping.load()) {
        const ssize_t count = recv(context.socket, buffer.data(), buffer.size(), 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count == 0) {
            stop_bridge(context, 0);
            return;
        }
        if (count < 0) {
            if (!context.stopping.load()) {
                std::fprintf(stderr, "TCP receive failed: %s\n", std::strerror(errno));
                stop_bridge(context, 1);
            }
            return;
        }
        if (!transfer_out(context, buffer.data(), (size_t) count)) {
            stop_bridge(context, 1);
            return;
        }
    }
}

static void usb_to_socket(bridge_context & context) {
    std::vector<unsigned char> buffer(TRANSFER_BYTES);
    while (!context.stopping.load()) {
        int transferred = 0;
        const int status = libusb_bulk_transfer(
            context.usb, IN_ENDPOINT, buffer.data(), (int) buffer.size(),
            &transferred, IN_TIMEOUT_MS);
        if (status == LIBUSB_ERROR_TIMEOUT_VALUE) {
            continue;
        }
        if (status != 0) {
            if (!context.stopping.load()) {
                std::fprintf(stderr, "AOA IN ended: %s\n", libusb_error_name(status));
            }
            stop_bridge(context, context.usb_in_bytes == 0 ? 1 : 0);
            return;
        }
        if (transferred <= 0) {
            continue;
        }
        context.usb_in_bytes += (uint64_t) transferred;
        ++context.usb_in_transfers;
        if (!send_socket_all(context.socket, buffer.data(), (size_t) transferred)) {
            if (!context.stopping.load()) {
                std::fprintf(stderr, "TCP send failed: %s\n", std::strerror(errno));
                stop_bridge(context, 1);
            }
            return;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s LOCAL_PORT\n", argv[0]);
        return 2;
    }
    libusb_context * usb_context = nullptr;
    libusb_device_handle * usb = nullptr;
    int listener = -1;
    int client = -1;
    try {
        const int port = parse_port(argv[1]);
        int status = libusb_init(&usb_context);
        if (status != 0) {
            throw std::runtime_error(std::string("libusb_init failed: ") +
                                     libusb_error_name(status));
        }
        uint16_t product = 0;
        usb = open_accessory(usb_context, product);
        if (usb == nullptr) {
            throw std::runtime_error("AOA device did not appear");
        }
        libusb_detach_kernel_driver(usb, 0);
        status = libusb_claim_interface(usb, 0);
        if (status != 0) {
            throw std::runtime_error(std::string("AOA claim failed: ") +
                                     libusb_error_name(status));
        }
        listener = create_listener(port);
        std::fprintf(stderr,
                     "[aoa-host-bridge] ready product=18d1:%04x port=%d\n",
                     product, port);
        std::fflush(stderr);
        client = accept(listener, nullptr, nullptr);
        if (client < 0) {
            throw std::runtime_error(std::string("accept failed: ") +
                                     std::strerror(errno));
        }
        close(listener);
        listener = -1;
        const int socket_buffer = 1024 * 1024;
        setsockopt(client, SOL_SOCKET, SO_SNDBUF, &socket_buffer, sizeof(socket_buffer));
        setsockopt(client, SOL_SOCKET, SO_RCVBUF, &socket_buffer, sizeof(socket_buffer));
        const int no_delay = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
        std::fprintf(stderr, "[aoa-host-bridge] client connected\n");
        std::fflush(stderr);

        bridge_context context;
        context.usb = usb;
        context.socket = client;
        const uint64_t started = now_ns();
        std::thread input_thread(usb_to_socket, std::ref(context));
        std::thread output_thread(socket_to_usb, std::ref(context));
        output_thread.join();
        context.stopping.store(true);
        input_thread.join();
        const uint64_t elapsed = now_ns() - started;
        std::fprintf(stderr,
                     "AOABRIDGE {\"role\":\"host\",\"status\":%d,"
                     "\"usb_out_bytes\":%llu,\"usb_in_bytes\":%llu,"
                     "\"usb_out_transfers\":%llu,\"usb_in_transfers\":%llu,"
                     "\"elapsed_ms\":%.3f}\n",
                     context.failure.load(),
                     (unsigned long long) context.usb_out_bytes,
                     (unsigned long long) context.usb_in_bytes,
                     (unsigned long long) context.usb_out_transfers,
                     (unsigned long long) context.usb_in_transfers,
                     elapsed / 1e6);
        std::fflush(stderr);
        close(client);
        client = -1;
        libusb_release_interface(usb, 0);
        libusb_close(usb);
        usb = nullptr;
        libusb_exit(usb_context);
        usb_context = nullptr;
        return context.failure.load() == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        if (client >= 0) {
            close(client);
        }
        if (listener >= 0) {
            close(listener);
        }
        if (usb != nullptr) {
            libusb_release_interface(usb, 0);
            libusb_close(usb);
        }
        if (usb_context != nullptr) {
            libusb_exit(usb_context);
        }
        return 1;
    }
}
