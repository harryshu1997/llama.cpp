#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/usb/ch9.h>
#include <linux/usb/functionfs.h>
#include <poll.h>
#include <signal.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#ifndef FUNCTIONFS_DMABUF_ATTACH
struct usb_ffs_dmabuf_transfer_req {
    int fd;
    uint32_t flags;
    uint64_t length;
} __attribute__((packed));

#define FUNCTIONFS_DMABUF_ATTACH _IOW('g', 131, int)
#define FUNCTIONFS_DMABUF_DETACH _IOW('g', 132, int)
#define FUNCTIONFS_DMABUF_TRANSFER \
    _IOW('g', 133, struct usb_ffs_dmabuf_transfer_req)
#endif

#define ENABLE_TIMEOUT_MS 30000
#define S41_INTERFACE_STRING "S41 FunctionFS HTP DMA-BUF"

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define cpu_to_le16(value) (value)
#define cpu_to_le32(value) (value)
#else
#define cpu_to_le16(value) __builtin_bswap16(value)
#define cpu_to_le32(value) __builtin_bswap32(value)
#endif

static const struct {
    struct usb_functionfs_descs_head_v2 header;
    uint32_t fs_count;
    uint32_t hs_count;
    uint32_t ss_count;
    struct {
        struct usb_interface_descriptor interface;
        struct usb_endpoint_descriptor_no_audio device_to_host;
        struct usb_endpoint_descriptor_no_audio host_to_device;
    } __attribute__((packed)) fs, hs;
    struct {
        struct usb_interface_descriptor interface;
        struct usb_endpoint_descriptor_no_audio device_to_host;
        struct usb_ss_ep_comp_descriptor device_to_host_companion;
        struct usb_endpoint_descriptor_no_audio host_to_device;
        struct usb_ss_ep_comp_descriptor host_to_device_companion;
    } __attribute__((packed)) ss;
} __attribute__((packed)) descriptors = {
    .header = {
        .magic = cpu_to_le32(FUNCTIONFS_DESCRIPTORS_MAGIC_V2),
        .length = cpu_to_le32(sizeof(descriptors)),
        .flags = cpu_to_le32(FUNCTIONFS_HAS_FS_DESC |
                FUNCTIONFS_HAS_HS_DESC | FUNCTIONFS_HAS_SS_DESC),
    },
    .fs_count = cpu_to_le32(3),
    .hs_count = cpu_to_le32(3),
    .ss_count = cpu_to_le32(5),
    .fs = {
        .interface = {
            .bLength = sizeof(struct usb_interface_descriptor),
            .bDescriptorType = USB_DT_INTERFACE,
            .bInterfaceNumber = 0,
            .bAlternateSetting = 0,
            .bNumEndpoints = 2,
            .bInterfaceClass = USB_CLASS_VENDOR_SPEC,
            .bInterfaceSubClass = 0,
            .bInterfaceProtocol = 0,
            .iInterface = 1,
        },
        .device_to_host = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = 0,
            .bInterval = 0,
        },
        .host_to_device = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = 0,
            .bInterval = 0,
        },
    },
    .hs = {
        .interface = {
            .bLength = sizeof(struct usb_interface_descriptor),
            .bDescriptorType = USB_DT_INTERFACE,
            .bInterfaceNumber = 0,
            .bAlternateSetting = 0,
            .bNumEndpoints = 2,
            .bInterfaceClass = USB_CLASS_VENDOR_SPEC,
            .bInterfaceSubClass = 0,
            .bInterfaceProtocol = 0,
            .iInterface = 1,
        },
        .device_to_host = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = cpu_to_le16(512),
            .bInterval = 0,
        },
        .host_to_device = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = cpu_to_le16(512),
            .bInterval = 0,
        },
    },
    .ss = {
        .interface = {
            .bLength = sizeof(struct usb_interface_descriptor),
            .bDescriptorType = USB_DT_INTERFACE,
            .bInterfaceNumber = 0,
            .bAlternateSetting = 0,
            .bNumEndpoints = 2,
            .bInterfaceClass = USB_CLASS_VENDOR_SPEC,
            .bInterfaceSubClass = 0,
            .bInterfaceProtocol = 0,
            .iInterface = 1,
        },
        .device_to_host = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = cpu_to_le16(1024),
            .bInterval = 0,
        },
        .device_to_host_companion = {
            .bLength = sizeof(struct usb_ss_ep_comp_descriptor),
            .bDescriptorType = USB_DT_SS_ENDPOINT_COMP,
            .bMaxBurst = 15,
            .bmAttributes = 0,
            .wBytesPerInterval = 0,
        },
        .host_to_device = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = cpu_to_le16(1024),
            .bInterval = 0,
        },
        .host_to_device_companion = {
            .bLength = sizeof(struct usb_ss_ep_comp_descriptor),
            .bDescriptorType = USB_DT_SS_ENDPOINT_COMP,
            .bMaxBurst = 15,
            .bmAttributes = 0,
            .wBytesPerInterval = 0,
        },
    },
};

static const struct {
    struct usb_functionfs_strings_head header;
    struct {
        uint16_t language;
        char interface[sizeof(S41_INTERFACE_STRING)];
    } __attribute__((packed)) language;
} __attribute__((packed)) strings = {
    .header = {
        .magic = cpu_to_le32(FUNCTIONFS_STRINGS_MAGIC),
        .length = cpu_to_le32(sizeof(strings)),
        .str_count = cpu_to_le32(1),
        .lang_count = cpu_to_le32(1),
    },
    .language = {
        .language = cpu_to_le16(0x0409),
        .interface = S41_INTERFACE_STRING,
    },
};

static uint64_t now_ns() {
    struct timespec value = {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<uint64_t>(value.tv_sec) * UINT64_C(1000000000) +
            static_cast<uint64_t>(value.tv_nsec);
}

static uint64_t process_cpu_ns() {
    struct timespec value = {};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value);
    return static_cast<uint64_t>(value.tv_sec) * UINT64_C(1000000000) +
            static_cast<uint64_t>(value.tv_nsec);
}

static int write_exact(int descriptor, const void * source, size_t size) {
    const uint8_t * cursor = static_cast<const uint8_t *>(source);
    while (size > 0) {
        const ssize_t count = write(descriptor, cursor, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        cursor += count;
        size -= static_cast<size_t>(count);
    }
    return 0;
}

static int read_exact(int descriptor, void * destination, size_t size) {
    uint8_t * cursor = static_cast<uint8_t *>(destination);
    while (size > 0) {
        const ssize_t count = read(descriptor, cursor, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        cursor += count;
        size -= static_cast<size_t>(count);
    }
    return 0;
}

static int touch_file(const char * path, const char * text) {
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0660);
    if (descriptor < 0) {
        return -1;
    }
    const int status = write_exact(descriptor, text, strlen(text));
    const int saved_errno = errno;
    close(descriptor);
    errno = saved_errno;
    return status;
}

static int open_endpoint(const char * root, const char * name) {
    char path[256];
    if (snprintf(path, sizeof(path), "%s/%s", root, name) >=
            static_cast<int>(sizeof(path))) {
        errno = ENAMETOOLONG;
        return -1;
    }
    return open(path, O_RDWR | O_CLOEXEC);
}

static int wait_for_enable(int ep0) {
    const uint64_t deadline = now_ns() +
            static_cast<uint64_t>(ENABLE_TIMEOUT_MS) * UINT64_C(1000000);
    while (now_ns() < deadline) {
        struct pollfd item = {
            .fd = ep0,
            .events = POLLIN,
            .revents = 0,
        };
        const int status = poll(&item, 1, 250);
        if (status < 0 && errno == EINTR) {
            continue;
        }
        if (status < 0) {
            return -1;
        }
        if (status == 0 || !(item.revents & POLLIN)) {
            continue;
        }
        struct usb_functionfs_event event = {};
        const ssize_t count = read(ep0, &event, sizeof(event));
        if (count != sizeof(event)) {
            return -1;
        }
        if (event.type == FUNCTIONFS_SETUP) {
            const int control_status = event.u.setup.bRequestType & USB_DIR_IN
                    ? static_cast<int>(write(ep0, nullptr, 0))
                    : static_cast<int>(read(ep0, nullptr, 0));
            if (control_status < 0) {
                return -1;
            }
        } else if (event.type == FUNCTIONFS_ENABLE) {
            return 0;
        }
    }
    errno = ETIMEDOUT;
    return -1;
}

static int dmabuf_attach(int endpoint, int descriptor) {
    return ioctl(endpoint, FUNCTIONFS_DMABUF_ATTACH, &descriptor);
}

static int dmabuf_detach(int endpoint, int descriptor) {
    return ioctl(endpoint, FUNCTIONFS_DMABUF_DETACH, &descriptor);
}

static int dmabuf_queue(int endpoint, int descriptor, size_t size) {
    const struct usb_ffs_dmabuf_transfer_req transfer = {
        .fd = descriptor,
        .flags = 0,
        .length = size,
    };
    return ioctl(endpoint, FUNCTIONFS_DMABUF_TRANSFER, &transfer);
}

static int dmabuf_sync(int descriptor, uint64_t flags) {
    struct dma_buf_sync sync = {
        .flags = flags,
    };
    return ioctl(descriptor, DMA_BUF_IOCTL_SYNC, &sync);
}

static int dmabuf_wait(int descriptor) {
    if (dmabuf_sync(descriptor, DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW) != 0) {
        return -1;
    }
    return dmabuf_sync(descriptor, DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW);
}

static uint64_t median(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0
            ? (values[middle - 1] + values[middle]) / 2
            : values[middle];
}

static bool parse_u64(const char * text, uint64_t & value) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s <htp-sqr|htp-copy-sqr|htp-staged-sqr> "
                "<ffs-root> <request-bytes> "
                "<response-bytes> <total> <warmup> 1 <ready-file>\n",
                argv[0]);
        return 2;
    }

    uint64_t request_bytes = 0;
    uint64_t response_bytes = 0;
    uint64_t total = 0;
    uint64_t warmup = 0;
    uint64_t depth = 0;
    const bool dmabuf_mode = strcmp(argv[1], "htp-sqr") == 0;
    const bool copy_mode = strcmp(argv[1], "htp-copy-sqr") == 0;
    const bool staged_mode = strcmp(argv[1], "htp-staged-sqr") == 0;
    if ((!dmabuf_mode && !copy_mode && !staged_mode) ||
            !parse_u64(argv[3], request_bytes) ||
            !parse_u64(argv[4], response_bytes) ||
            !parse_u64(argv[5], total) ||
            !parse_u64(argv[6], warmup) ||
            !parse_u64(argv[7], depth) ||
            request_bytes != response_bytes || request_bytes == 0 ||
            request_bytes % sizeof(float) != 0 ||
            request_bytes > 16U * 1024U * 1024U || total == 0 ||
            warmup >= total || depth != 1) {
        fprintf(stderr, "[ffs-htp] invalid arguments\n");
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);

    ggml_backend_t backend = nullptr;
    const char * selected_name = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const char * name = ggml_backend_dev_name(device);
        if (strstr(name, "HTP") != nullptr) {
            backend = ggml_backend_dev_init(device, nullptr);
            selected_name = name;
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[ffs-htp] HTP backend not found\n");
        return 1;
    }

    struct ggml_init_params parameters = {};
    parameters.mem_size = ggml_tensor_overhead() * 4 +
            ggml_graph_overhead_custom(4, false);
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        fprintf(stderr, "[ffs-htp] context allocation failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    const int64_t elements = static_cast<int64_t>(
            request_bytes / sizeof(float));
    ggml_tensor * input = ggml_new_tensor_1d(
            context, GGML_TYPE_F32, elements);
    ggml_tensor * output = ggml_sqr(context, input);
    ggml_set_input(input);
    ggml_set_output(output);
    ggml_backend_buffer_type_t buffer_type =
            ggml_backend_get_default_buffer_type(backend);
    const size_t input_size = ggml_backend_buft_get_alloc_size(
            buffer_type, input);
    const size_t output_size = ggml_backend_buft_get_alloc_size(
            buffer_type, output);
    ggml_backend_buffer_t input_buffer =
            ggml_backend_buft_alloc_buffer(buffer_type, input_size);
    ggml_backend_buffer_t output_buffer =
            ggml_backend_buft_alloc_buffer(buffer_type, output_size);
    if (input_buffer == nullptr || output_buffer == nullptr) {
        fprintf(stderr, "[ffs-htp] HTP buffer allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(
            input_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_set_usage(
            output_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (ggml_backend_tensor_alloc(input_buffer, input,
                ggml_backend_buffer_get_base(input_buffer)) !=
                    GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(output_buffer, output,
                ggml_backend_buffer_get_base(output_buffer)) !=
                    GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ffs-htp] tensor allocation failed\n");
        return 1;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context, 4, false);
    ggml_build_forward_expand(graph, output);

    const int input_fd = ggml_backend_hexagon_buffer_get_fd(input_buffer);
    const int output_fd = ggml_backend_hexagon_buffer_get_fd(output_buffer);
    if (dmabuf_mode && (input_fd < 0 || output_fd < 0)) {
        fprintf(stderr, "[ffs-htp] HTP DMA-BUF FD unavailable\n");
        return 1;
    }

    std::vector<uint8_t> request_staging(
            staged_mode ? static_cast<size_t>(request_bytes) : 0);
    std::vector<uint8_t> response_staging(
            staged_mode ? static_cast<size_t>(response_bytes) : 0);

    const int ep0 = open_endpoint(argv[2], "ep0");
    if (ep0 < 0 || write_exact(ep0, &descriptors, sizeof(descriptors)) != 0 ||
            write_exact(ep0, &strings, sizeof(strings)) != 0) {
        fprintf(stderr, "[ffs-htp] descriptor setup failed: %s\n",
                strerror(errno));
        return 1;
    }
    const int device_to_host = open_endpoint(argv[2], "ep1");
    const int host_to_device = open_endpoint(argv[2], "ep2");
    if (device_to_host < 0 || host_to_device < 0) {
        fprintf(stderr, "[ffs-htp] endpoint open failed: %s\n",
                strerror(errno));
        return 1;
    }
    if (touch_file(argv[8], "descriptors_ready\n") != 0) {
        fprintf(stderr, "[ffs-htp] ready marker failed: %s\n",
                strerror(errno));
        return 1;
    }
    fprintf(stderr,
            "[ffs-htp] descriptors_ready mode=%s backend=%s elements=%lld "
            "input_fd=%d output_fd=%d total=%llu warmup=%llu\n",
            argv[1], selected_name, static_cast<long long>(elements),
            input_fd, output_fd, static_cast<unsigned long long>(total),
            static_cast<unsigned long long>(warmup));
    fflush(stderr);
    if (wait_for_enable(ep0) != 0) {
        fprintf(stderr, "[ffs-htp] enable failed: %s\n", strerror(errno));
        return 1;
    }
    if (dmabuf_mode &&
            (dmabuf_attach(host_to_device, input_fd) != 0 ||
             dmabuf_attach(device_to_host, output_fd) != 0 ||
             dmabuf_queue(host_to_device, input_fd, request_bytes) != 0)) {
        fprintf(stderr, "[ffs-htp] DMA-BUF setup failed: %s\n",
                strerror(errno));
        return 1;
    }

    std::vector<uint64_t> wait_values;
    std::vector<uint64_t> submit_values;
    std::vector<uint64_t> sync_values;
    std::vector<uint64_t> transmit_values;
    std::vector<uint64_t> cpu_values;
    wait_values.reserve(total - warmup);
    submit_values.reserve(total - warmup);
    sync_values.reserve(total - warmup);
    transmit_values.reserve(total - warmup);
    cpu_values.reserve(total - warmup);

    int status = 0;
    for (uint64_t sequence = 1; sequence <= total; ++sequence) {
        const uint64_t cpu_started = process_cpu_ns();
        uint64_t started = now_ns();
        void * input_destination = staged_mode
                ? static_cast<void *>(request_staging.data()) : input->data;
        const int input_status = dmabuf_mode
                ? dmabuf_wait(input_fd)
                : read_exact(host_to_device, input_destination,
                        static_cast<size_t>(request_bytes));
        if (input_status != 0) {
            fprintf(stderr, "[ffs-htp] input transfer failed: %s\n",
                    strerror(errno));
            status = 1;
            break;
        }
        if (staged_mode) {
            ggml_backend_tensor_set(input, request_staging.data(), 0,
                    static_cast<size_t>(request_bytes));
        }
        const uint64_t wait_ns = now_ns() - started;

        started = now_ns();
        const enum ggml_status compute_status =
                ggml_backend_graph_compute_async(backend, graph);
        const uint64_t submit_ns = now_ns() - started;
        if (compute_status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[ffs-htp] graph submit failed: %d\n",
                    static_cast<int>(compute_status));
            status = 1;
            break;
        }
        started = now_ns();
        ggml_backend_synchronize(backend);
        const uint64_t sync_ns = now_ns() - started;

        started = now_ns();
        int output_status = 0;
        if (dmabuf_mode) {
            output_status = dmabuf_queue(device_to_host, output_fd,
                    response_bytes) != 0 || dmabuf_wait(output_fd) != 0
                    ? -1 : 0;
        } else {
            const void * output_source = output->data;
            if (staged_mode) {
                ggml_backend_tensor_get(output, response_staging.data(), 0,
                        static_cast<size_t>(response_bytes));
                output_source = response_staging.data();
            }
            output_status = write_exact(device_to_host, output_source,
                    static_cast<size_t>(response_bytes));
        }
        if (output_status != 0) {
            fprintf(stderr, "[ffs-htp] output transfer failed: %s\n",
                    strerror(errno));
            status = 1;
            break;
        }
        const uint64_t transmit_ns = now_ns() - started;
        const uint64_t cpu_ns = process_cpu_ns() - cpu_started;

        if (sequence > warmup) {
            wait_values.push_back(wait_ns);
            submit_values.push_back(submit_ns);
            sync_values.push_back(sync_ns);
            transmit_values.push_back(transmit_ns);
            cpu_values.push_back(cpu_ns);
        }
        if (dmabuf_mode && sequence < total &&
                dmabuf_queue(host_to_device, input_fd, request_bytes) != 0) {
            fprintf(stderr, "[ffs-htp] input requeue failed: %s\n",
                    strerror(errno));
            status = 1;
            break;
        }
    }

    if (dmabuf_mode) {
        dmabuf_detach(host_to_device, input_fd);
        dmabuf_detach(device_to_host, output_fd);
    }
    close(host_to_device);
    close(device_to_host);
    close(ep0);

    if (status == 0) {
        fprintf(stderr,
                "[ffs-htp] complete mode=%s paid=%zu input_wait_us=%.3f "
                "submit_us=%.3f sync_us=%.3f output_wait_us=%.3f "
                "process_cpu_us=%.3f\n",
                argv[1], wait_values.size(), median(wait_values) / 1e3,
                median(submit_values) / 1e3, median(sync_values) / 1e3,
                median(transmit_values) / 1e3, median(cpu_values) / 1e3);
    }
    ggml_backend_buffer_free(output_buffer);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    return status;
}
