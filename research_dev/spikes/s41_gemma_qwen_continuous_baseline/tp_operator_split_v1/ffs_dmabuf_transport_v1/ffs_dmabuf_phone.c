#define _GNU_SOURCE

#include "../aoa_async_transport_v1/aoa_async_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <linux/usb/ch9.h>
#include <linux/usb/functionfs.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

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

#define MAX_TRANSFER_BYTES (16U * 1024U * 1024U)
#define MAX_RING_DEPTH 16U
#define ENABLE_TIMEOUT_MS 30000
#define S41_INTERFACE_STRING "S41 FunctionFS DMA-BUF"

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
            .bNumEndpoints = 2,
            .bInterfaceClass = USB_CLASS_VENDOR_SPEC,
            .iInterface = 1,
        },
        .device_to_host = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
        },
        .host_to_device = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
        },
    },
    .hs = {
        .interface = {
            .bLength = sizeof(struct usb_interface_descriptor),
            .bDescriptorType = USB_DT_INTERFACE,
            .bNumEndpoints = 2,
            .bInterfaceClass = USB_CLASS_VENDOR_SPEC,
            .iInterface = 1,
        },
        .device_to_host = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = cpu_to_le16(512),
        },
        .host_to_device = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = cpu_to_le16(512),
        },
    },
    .ss = {
        .interface = {
            .bLength = sizeof(struct usb_interface_descriptor),
            .bDescriptorType = USB_DT_INTERFACE,
            .bNumEndpoints = 2,
            .bInterfaceClass = USB_CLASS_VENDOR_SPEC,
            .iInterface = 1,
        },
        .device_to_host = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = cpu_to_le16(1024),
        },
        .device_to_host_companion = {
            .bLength = sizeof(struct usb_ss_ep_comp_descriptor),
            .bDescriptorType = USB_DT_SS_ENDPOINT_COMP,
            .bMaxBurst = 15,
        },
        .host_to_device = {
            .bLength = sizeof(struct usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = cpu_to_le16(1024),
        },
        .host_to_device_companion = {
            .bLength = sizeof(struct usb_ss_ep_comp_descriptor),
            .bDescriptorType = USB_DT_SS_ENDPOINT_COMP,
            .bMaxBurst = 15,
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

struct dmabuf_slot {
    int receive_fd;
    int transmit_fd;
    uint8_t * receive;
    uint8_t * transmit;
    int transmit_pending;
};

static uint64_t now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t) value.tv_sec * UINT64_C(1000000000) +
        (uint64_t) value.tv_nsec;
}

static uint64_t boot_time_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_BOOTTIME, &value);
    return (uint64_t) value.tv_sec * UINT64_C(1000000000) +
        (uint64_t) value.tv_nsec;
}

static int write_exact(int descriptor, const void * source, size_t size) {
    const uint8_t * cursor = source;
    while (size > 0) {
        const ssize_t count = write(descriptor, cursor, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        cursor += count;
        size -= (size_t) count;
    }
    return 0;
}

static int read_exact(int descriptor, void * destination, size_t size) {
    uint8_t * cursor = destination;
    while (size > 0) {
        const ssize_t count = read(descriptor, cursor, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        cursor += count;
        size -= (size_t) count;
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

static int validate_request(const uint8_t * data, size_t size,
        size_t request_bytes, size_t response_bytes, uint64_t sequence) {
    struct s41_aoa_frame_header header;
    uint64_t trailing;
    if (size < sizeof(header) + sizeof(trailing)) {
        return -1;
    }
    memcpy(&header, data, sizeof(header));
    memcpy(&trailing, data + size - sizeof(trailing), sizeof(trailing));
    return header.magic == S41_AOA_REQUEST_MAGIC &&
        header.sequence == sequence &&
        header.request_bytes == request_bytes &&
        header.response_bytes == response_bytes &&
        header.sentinel == s41_aoa_sentinel(sequence) &&
        trailing == s41_aoa_sentinel(sequence) ? 0 : -1;
}

static void prepare_response(uint8_t * data, size_t size,
        size_t request_bytes, size_t response_bytes, uint64_t sequence) {
    const struct s41_aoa_frame_header header = {
        .magic = S41_AOA_RESPONSE_MAGIC,
        .sequence = sequence,
        .request_bytes = (uint32_t) request_bytes,
        .response_bytes = (uint32_t) response_bytes,
        .sentinel = s41_aoa_sentinel(sequence),
    };
    const uint64_t trailing = s41_aoa_sentinel(sequence);
    memcpy(data, &header, sizeof(header));
    memcpy(data + size - sizeof(trailing), &trailing, sizeof(trailing));
}

static int wait_for_enable(int ep0) {
    const uint64_t deadline = now_ns() +
        (uint64_t) ENABLE_TIMEOUT_MS * UINT64_C(1000000);
    while (now_ns() < deadline) {
        struct pollfd item = {
            .fd = ep0,
            .events = POLLIN,
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
        struct usb_functionfs_event event;
        const ssize_t count = read(ep0, &event, sizeof(event));
        if (count != sizeof(event)) {
            return -1;
        }
        if (event.type == FUNCTIONFS_SETUP) {
            const int control_status = event.u.setup.bRequestType & USB_DIR_IN
                ? (int) write(ep0, NULL, 0) : (int) read(ep0, NULL, 0);
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

static int dmabuf_allocate(int heap, size_t size, int * buffer_fd,
        uint8_t ** mapping) {
    struct dma_heap_allocation_data allocation = {
        .len = size,
        .fd_flags = O_RDWR | O_CLOEXEC,
    };
    if (ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation) != 0) {
        return -1;
    }
    void * address = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
            (int) allocation.fd, 0);
    if (address == MAP_FAILED) {
        const int saved_errno = errno;
        close((int) allocation.fd);
        errno = saved_errno;
        return -1;
    }
    *buffer_fd = (int) allocation.fd;
    *mapping = address;
    return 0;
}

static int dmabuf_sync(int descriptor, uint64_t flags) {
    struct dma_buf_sync sync = {
        .flags = flags,
    };
    return ioctl(descriptor, DMA_BUF_IOCTL_SYNC, &sync);
}

static int dmabuf_attach(int endpoint, int descriptor) {
    return ioctl(endpoint, FUNCTIONFS_DMABUF_ATTACH, &descriptor);
}

static int dmabuf_queue(int endpoint, int descriptor, size_t size) {
    const struct usb_ffs_dmabuf_transfer_req transfer = {
        .fd = descriptor,
        .length = size,
    };
    return ioctl(endpoint, FUNCTIONFS_DMABUF_TRANSFER, &transfer);
}

static int dmabuf_wait(int descriptor) {
    if (dmabuf_sync(descriptor, DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW) != 0) {
        return -1;
    }
    return dmabuf_sync(descriptor, DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW);
}

static int run_copy(int device_to_host, int host_to_device,
        size_t request_bytes, size_t response_bytes, uint64_t total,
        uint64_t warmup) {
    uint8_t * request = calloc(1, request_bytes);
    uint8_t * response = calloc(1, response_bytes);
    uint64_t measured_started = 0;
    if (request == NULL || response == NULL) {
        fprintf(stderr, "[ffs-dmabuf] copy allocation failed\n");
        return 1;
    }
    for (uint64_t index = 0; index < total; ++index) {
        const uint64_t sequence = index + 1;
        if (sequence == warmup + 1) {
            measured_started = now_ns();
            fprintf(stderr,
                    "PHONE_ENERGY_WINDOW_START phone_uptime_ns=%llu "
                    "label=usb-copy\n",
                    (unsigned long long) boot_time_ns());
            fflush(stderr);
        }
        if (read_exact(host_to_device, request, request_bytes) != 0) {
            fprintf(stderr, "[ffs-dmabuf] copy read failed at %llu: %s\n",
                    (unsigned long long) sequence, strerror(errno));
            return 1;
        }
        if (validate_request(request, request_bytes, request_bytes,
                response_bytes, sequence) != 0) {
            fprintf(stderr, "[ffs-dmabuf] copy validation failed at %llu\n",
                    (unsigned long long) sequence);
            return 1;
        }
        prepare_response(response, response_bytes, request_bytes,
                response_bytes, sequence);
        if (write_exact(device_to_host, response, response_bytes) != 0) {
            fprintf(stderr, "[ffs-dmabuf] copy write failed at %llu: %s\n",
                    (unsigned long long) sequence, strerror(errno));
            return 1;
        }
    }
    fprintf(stderr,
            "PHONE_ENERGY_WINDOW_END phone_uptime_ns=%llu label=usb-copy\n",
            (unsigned long long) boot_time_ns());
    fflush(stderr);
    const double seconds = (now_ns() - measured_started) / 1e9;
    fprintf(stderr,
            "[ffs-dmabuf] copy paid_seconds=%.9f aggregate_MBps=%.3f\n",
            seconds, (request_bytes + response_bytes) *
                (double) (total - warmup) / seconds / 1e6);
    free(response);
    free(request);
    return 0;
}

static void release_slots(struct dmabuf_slot * slots, unsigned int depth,
        size_t request_bytes, size_t response_bytes) {
    if (slots == NULL) {
        return;
    }
    for (unsigned int index = 0; index < depth; ++index) {
        if (slots[index].receive != NULL) {
            munmap(slots[index].receive, request_bytes);
        }
        if (slots[index].transmit != NULL) {
            munmap(slots[index].transmit, response_bytes);
        }
        if (slots[index].receive_fd >= 0) {
            close(slots[index].receive_fd);
        }
        if (slots[index].transmit_fd >= 0) {
            close(slots[index].transmit_fd);
        }
    }
    free(slots);
}

static int run_dmabuf(int device_to_host, int host_to_device,
        size_t request_bytes, size_t response_bytes, uint64_t total,
        uint64_t warmup, unsigned int depth) {
    int status = 1;
    const int heap = open("/dev/dma_heap/qcom,system", O_RDONLY | O_CLOEXEC);
    if (heap < 0) {
        fprintf(stderr, "[ffs-dmabuf] open DMA heap failed: %s\n",
                strerror(errno));
        return 1;
    }
    struct dmabuf_slot * slots = calloc(depth, sizeof(*slots));
    if (slots == NULL) {
        fprintf(stderr, "[ffs-dmabuf] slot allocation failed\n");
        close(heap);
        return 1;
    }
    for (unsigned int index = 0; index < depth; ++index) {
        slots[index].receive_fd = -1;
        slots[index].transmit_fd = -1;
    }
    for (unsigned int index = 0; index < depth; ++index) {
        struct dmabuf_slot * slot = &slots[index];
        if (dmabuf_allocate(heap, request_bytes, &slot->receive_fd,
                &slot->receive) != 0 ||
                dmabuf_allocate(heap, response_bytes, &slot->transmit_fd,
                &slot->transmit) != 0) {
            fprintf(stderr, "[ffs-dmabuf] DMA allocation failed: %s\n",
                    strerror(errno));
            goto done;
        }
        if (dmabuf_attach(host_to_device, slot->receive_fd) != 0 ||
                dmabuf_attach(device_to_host, slot->transmit_fd) != 0) {
            fprintf(stderr, "[ffs-dmabuf] endpoint attach failed: %s\n",
                    strerror(errno));
            goto done;
        }
        if (dmabuf_sync(slot->receive_fd,
                DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW) != 0) {
            fprintf(stderr, "[ffs-dmabuf] initial CPU sync failed: %s\n",
                    strerror(errno));
            goto done;
        }
        memset(slot->receive, 0, request_bytes);
        if (dmabuf_sync(slot->receive_fd,
                DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW) != 0 ||
                dmabuf_queue(host_to_device, slot->receive_fd,
                    request_bytes) != 0) {
            fprintf(stderr, "[ffs-dmabuf] initial RX queue failed: %s\n",
                    strerror(errno));
            goto done;
        }
    }

    uint64_t measured_started = 0;
    for (uint64_t index = 0; index < total; ++index) {
        const uint64_t sequence = index + 1;
        struct dmabuf_slot * slot = &slots[index % depth];
        if (sequence == warmup + 1) {
            measured_started = now_ns();
            fprintf(stderr,
                    "PHONE_ENERGY_WINDOW_START phone_uptime_ns=%llu "
                    "label=usb-dmabuf\n",
                    (unsigned long long) boot_time_ns());
            fflush(stderr);
        }
        if (dmabuf_wait(slot->receive_fd) != 0) {
            fprintf(stderr, "[ffs-dmabuf] RX wait failed at %llu: %s\n",
                    (unsigned long long) sequence, strerror(errno));
            goto done;
        }
        if (validate_request(slot->receive, request_bytes, request_bytes,
                response_bytes, sequence) != 0) {
            fprintf(stderr, "[ffs-dmabuf] validation failed at %llu\n",
                    (unsigned long long) sequence);
            goto done;
        }
        if (slot->transmit_pending) {
            if (dmabuf_wait(slot->transmit_fd) != 0) {
                fprintf(stderr,
                        "[ffs-dmabuf] TX wait failed at %llu: %s\n",
                        (unsigned long long) sequence, strerror(errno));
                goto done;
            }
            slot->transmit_pending = 0;
        }
        if (dmabuf_sync(slot->transmit_fd,
                DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW) != 0) {
            fprintf(stderr, "[ffs-dmabuf] TX CPU sync failed: %s\n",
                    strerror(errno));
            goto done;
        }
        prepare_response(slot->transmit, response_bytes, request_bytes,
                response_bytes, sequence);
        if (dmabuf_sync(slot->transmit_fd,
                DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW) != 0 ||
                dmabuf_queue(device_to_host, slot->transmit_fd,
                    response_bytes) != 0) {
            fprintf(stderr, "[ffs-dmabuf] TX queue failed at %llu: %s\n",
                    (unsigned long long) sequence, strerror(errno));
            goto done;
        }
        slot->transmit_pending = 1;

        if (sequence + depth <= total) {
            if (dmabuf_queue(host_to_device, slot->receive_fd,
                    request_bytes) != 0) {
                fprintf(stderr,
                        "[ffs-dmabuf] RX requeue failed at %llu: %s\n",
                        (unsigned long long) sequence, strerror(errno));
                goto done;
            }
        }
    }
    for (unsigned int index = 0; index < depth; ++index) {
        if (slots[index].transmit_pending &&
                dmabuf_wait(slots[index].transmit_fd) != 0) {
            fprintf(stderr, "[ffs-dmabuf] final TX wait failed: %s\n",
                    strerror(errno));
            goto done;
        }
    }
    fprintf(stderr,
            "PHONE_ENERGY_WINDOW_END phone_uptime_ns=%llu "
            "label=usb-dmabuf\n",
            (unsigned long long) boot_time_ns());
    fflush(stderr);
    {
        const double seconds = (now_ns() - measured_started) / 1e9;
        fprintf(stderr,
                "[ffs-dmabuf] dmabuf paid_seconds=%.9f "
                "aggregate_MBps=%.3f\n",
                seconds, (request_bytes + response_bytes) *
                    (double) (total - warmup) / seconds / 1e6);
    }
    status = 0;

done:
    release_slots(slots, depth, request_bytes, response_bytes);
    close(heap);
    return status;
}

static int parse_size(const char * text, size_t * value) {
    errno = 0;
    char * end = NULL;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
            parsed > SIZE_MAX) {
        return -1;
    }
    *value = (size_t) parsed;
    return 0;
}

static int open_endpoint(const char * root, const char * name) {
    char path[256];
    if (snprintf(path, sizeof(path), "%s/%s", root, name) >=
            (int) sizeof(path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    return open(path, O_RDWR | O_CLOEXEC);
}

int main(int argc, char ** argv) {
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s <copy|dmabuf> <ffs-root> <request-bytes> "
                "<response-bytes> <total> <warmup> <depth> <ready-file>\n",
                argv[0]);
        return 2;
    }
    size_t request_bytes;
    size_t response_bytes;
    size_t total_value;
    size_t warmup_value;
    size_t depth_value;
    if ((strcmp(argv[1], "copy") != 0 && strcmp(argv[1], "dmabuf") != 0) ||
            parse_size(argv[3], &request_bytes) != 0 ||
            parse_size(argv[4], &response_bytes) != 0 ||
            parse_size(argv[5], &total_value) != 0 ||
            parse_size(argv[6], &warmup_value) != 0 ||
            parse_size(argv[7], &depth_value) != 0 ||
            request_bytes < sizeof(struct s41_aoa_frame_header) + 8 ||
            response_bytes < sizeof(struct s41_aoa_frame_header) + 8 ||
            request_bytes > MAX_TRANSFER_BYTES ||
            response_bytes > MAX_TRANSFER_BYTES || total_value == 0 ||
            warmup_value >= total_value || depth_value == 0 ||
            depth_value > MAX_RING_DEPTH) {
        fprintf(stderr, "[ffs-dmabuf] invalid arguments\n");
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    const int ep0 = open_endpoint(argv[2], "ep0");
    if (ep0 < 0 || write_exact(ep0, &descriptors, sizeof(descriptors)) != 0 ||
            write_exact(ep0, &strings, sizeof(strings)) != 0) {
        fprintf(stderr, "[ffs-dmabuf] descriptor setup failed: %s\n",
                strerror(errno));
        return 1;
    }
    const int device_to_host = open_endpoint(argv[2], "ep1");
    const int host_to_device = open_endpoint(argv[2], "ep2");
    if (device_to_host < 0 || host_to_device < 0) {
        fprintf(stderr, "[ffs-dmabuf] endpoint open failed: %s\n",
                strerror(errno));
        return 1;
    }
    if (touch_file(argv[8], "descriptors_ready\n") != 0) {
        fprintf(stderr, "[ffs-dmabuf] ready marker failed: %s\n",
                strerror(errno));
        return 1;
    }
    fprintf(stderr,
            "[ffs-dmabuf] descriptors_ready mode=%s request=%zu "
            "response=%zu total=%zu warmup=%zu depth=%zu\n",
            argv[1], request_bytes, response_bytes, total_value,
            warmup_value, depth_value);
    fflush(stderr);
    if (wait_for_enable(ep0) != 0) {
        fprintf(stderr, "[ffs-dmabuf] enable failed: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "[ffs-dmabuf] enabled\n");
    fflush(stderr);
    const int status = strcmp(argv[1], "copy") == 0
        ? run_copy(device_to_host, host_to_device, request_bytes,
                response_bytes, total_value, warmup_value)
        : run_dmabuf(device_to_host, host_to_device, request_bytes,
                response_bytes, total_value, warmup_value,
                (unsigned int) depth_value);
    close(host_to_device);
    close(device_to_host);
    close(ep0);
    fprintf(stderr, "[ffs-dmabuf] complete status=%d\n", status);
    return status;
}
