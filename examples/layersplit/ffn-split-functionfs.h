#pragma once

#if !defined(__ANDROID__)
#error "FunctionFS DMA-BUF transport is Android-only"
#endif

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/usb/ch9.h>
#include <linux/usb/functionfs.h>
#include <poll.h>
#include <mutex>
#include <string>
#include <sys/ioctl.h>
#include <thread>
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

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define FFN_CPU_TO_LE16(value) (value)
#define FFN_CPU_TO_LE32(value) (value)
#else
#define FFN_CPU_TO_LE16(value) __builtin_bswap16(value)
#define FFN_CPU_TO_LE32(value) __builtin_bswap32(value)
#endif

namespace ffn_split {

static constexpr int functionfs_enable_timeout_ms = 30000;
static constexpr char functionfs_interface_string[] =
        "Gemma4 FFN HTP DMA-BUF";

static const struct {
    usb_functionfs_descs_head_v2 header;
    uint32_t fs_count;
    uint32_t hs_count;
    uint32_t ss_count;
    struct {
        usb_interface_descriptor interface;
        usb_endpoint_descriptor_no_audio device_to_host;
        usb_endpoint_descriptor_no_audio host_to_device;
    } __attribute__((packed)) fs, hs;
    struct {
        usb_interface_descriptor interface;
        usb_endpoint_descriptor_no_audio device_to_host;
        usb_ss_ep_comp_descriptor device_to_host_companion;
        usb_endpoint_descriptor_no_audio host_to_device;
        usb_ss_ep_comp_descriptor host_to_device_companion;
    } __attribute__((packed)) ss;
} __attribute__((packed)) functionfs_descriptors = {
    .header = {
        .magic = FFN_CPU_TO_LE32(FUNCTIONFS_DESCRIPTORS_MAGIC_V2),
        .length = FFN_CPU_TO_LE32(sizeof(functionfs_descriptors)),
        .flags = FFN_CPU_TO_LE32(FUNCTIONFS_HAS_FS_DESC |
                FUNCTIONFS_HAS_HS_DESC | FUNCTIONFS_HAS_SS_DESC),
    },
    .fs_count = FFN_CPU_TO_LE32(3),
    .hs_count = FFN_CPU_TO_LE32(3),
    .ss_count = FFN_CPU_TO_LE32(5),
    .fs = {
        .interface = {
            .bLength = sizeof(usb_interface_descriptor),
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
            .bLength = sizeof(usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = 0,
            .bInterval = 0,
        },
        .host_to_device = {
            .bLength = sizeof(usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = 0,
            .bInterval = 0,
        },
    },
    .hs = {
        .interface = {
            .bLength = sizeof(usb_interface_descriptor),
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
            .bLength = sizeof(usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = FFN_CPU_TO_LE16(512),
            .bInterval = 0,
        },
        .host_to_device = {
            .bLength = sizeof(usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = FFN_CPU_TO_LE16(512),
            .bInterval = 0,
        },
    },
    .ss = {
        .interface = {
            .bLength = sizeof(usb_interface_descriptor),
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
            .bLength = sizeof(usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_IN | 1,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = FFN_CPU_TO_LE16(1024),
            .bInterval = 0,
        },
        .device_to_host_companion = {
            .bLength = sizeof(usb_ss_ep_comp_descriptor),
            .bDescriptorType = USB_DT_SS_ENDPOINT_COMP,
            .bMaxBurst = 15,
            .bmAttributes = 0,
            .wBytesPerInterval = 0,
        },
        .host_to_device = {
            .bLength = sizeof(usb_endpoint_descriptor_no_audio),
            .bDescriptorType = USB_DT_ENDPOINT,
            .bEndpointAddress = USB_DIR_OUT | 2,
            .bmAttributes = USB_ENDPOINT_XFER_BULK,
            .wMaxPacketSize = FFN_CPU_TO_LE16(1024),
            .bInterval = 0,
        },
        .host_to_device_companion = {
            .bLength = sizeof(usb_ss_ep_comp_descriptor),
            .bDescriptorType = USB_DT_SS_ENDPOINT_COMP,
            .bMaxBurst = 15,
            .bmAttributes = 0,
            .wBytesPerInterval = 0,
        },
    },
};

static const struct {
    usb_functionfs_strings_head header;
    struct {
        uint16_t language;
        char interface[sizeof(functionfs_interface_string)];
    } __attribute__((packed)) language;
} __attribute__((packed)) functionfs_strings = {
    .header = {
        .magic = FFN_CPU_TO_LE32(FUNCTIONFS_STRINGS_MAGIC),
        .length = FFN_CPU_TO_LE32(sizeof(functionfs_strings)),
        .str_count = FFN_CPU_TO_LE32(1),
        .lang_count = FFN_CPU_TO_LE32(1),
    },
    .language = {
        .language = FFN_CPU_TO_LE16(0x0409),
        .interface = "Gemma4 FFN HTP DMA-BUF",
    },
};

static inline uint64_t functionfs_now_ns() {
    timespec value = {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<uint64_t>(value.tv_sec) * UINT64_C(1000000000) +
            static_cast<uint64_t>(value.tv_nsec);
}

static inline bool functionfs_write_exact(
        int descriptor, const void * source, size_t size) {
    const uint8_t * cursor = static_cast<const uint8_t *>(source);
    while (size > 0) {
        const ssize_t count = write(descriptor, cursor, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        cursor += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

static inline bool functionfs_read_exact(
        int descriptor, void * destination, size_t size) {
    uint8_t * cursor = static_cast<uint8_t *>(destination);
    while (size > 0) {
        const ssize_t count = read(descriptor, cursor, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        cursor += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

static inline int functionfs_open_endpoint(
        const std::string & root, const char * name) {
    const std::string path = root + "/" + name;
    return open(path.c_str(), O_RDWR | O_CLOEXEC);
}

static inline bool functionfs_touch(
        const std::string & path, const char * text) {
    const int descriptor = open(
            path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0660);
    if (descriptor < 0) {
        return false;
    }
    const bool ok = functionfs_write_exact(descriptor, text, strlen(text));
    const int saved_errno = errno;
    close(descriptor);
    errno = saved_errno;
    return ok;
}

static inline bool functionfs_wait_for_enable(int ep0) {
    const uint64_t deadline = functionfs_now_ns() +
            static_cast<uint64_t>(functionfs_enable_timeout_ms) *
                    UINT64_C(1000000);
    while (functionfs_now_ns() < deadline) {
        pollfd item = { ep0, POLLIN, 0 };
        const int status = poll(&item, 1, 250);
        if (status < 0 && errno == EINTR) {
            continue;
        }
        if (status < 0) {
            return false;
        }
        if (status == 0 || !(item.revents & POLLIN)) {
            continue;
        }
        usb_functionfs_event event = {};
        const ssize_t count = read(ep0, &event, sizeof(event));
        if (count != sizeof(event)) {
            return false;
        }
        if (event.type == FUNCTIONFS_SETUP) {
            const int control_status = event.u.setup.bRequestType & USB_DIR_IN
                    ? static_cast<int>(write(ep0, nullptr, 0))
                    : static_cast<int>(read(ep0, nullptr, 0));
            if (control_status < 0) {
                return false;
            }
        } else if (event.type == FUNCTIONFS_ENABLE) {
            return true;
        }
    }
    errno = ETIMEDOUT;
    return false;
}

class functionfs_event_monitor {
public:
    explicit functionfs_event_monitor(int ep0) : ep0_(ep0) {
        thread_ = std::thread(&functionfs_event_monitor::run, this);
    }

    ~functionfs_event_monitor() {
        stop();
    }

    functionfs_event_monitor(const functionfs_event_monitor &) = delete;
    functionfs_event_monitor & operator=(
            const functionfs_event_monitor &) = delete;

    bool wait_enabled(uint64_t & generation, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!condition_.wait_for(
                    lock, std::chrono::milliseconds(timeout_ms),
                    [this]() { return enabled_ || failed_; })) {
            errno = ETIMEDOUT;
            return false;
        }
        if (failed_) {
            errno = event_errno_;
            return false;
        }
        generation = generation_;
        return true;
    }

    bool wait_reenabled(
            uint64_t previous_generation, uint64_t & generation,
            int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!condition_.wait_for(
                    lock, std::chrono::milliseconds(timeout_ms),
                    [this, previous_generation]() {
                        return failed_ ||
                                (enabled_ && generation_ != previous_generation);
                    })) {
            errno = ETIMEDOUT;
            return false;
        }
        if (failed_) {
            errno = event_errno_;
            return false;
        }
        generation = generation_;
        return true;
    }

    bool current(uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        return !failed_ && enabled_ && generation_ == generation;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                return;
            }
            stop_ = true;
        }
        condition_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void fail(int error) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
            event_errno_ = error != 0 ? error : EIO;
        }
        condition_.notify_all();
    }

    bool stopping() {
        std::lock_guard<std::mutex> lock(mutex_);
        return stop_;
    }

    void run() {
        while (!stopping()) {
            pollfd item = { ep0_, POLLIN, 0 };
            const int status = poll(&item, 1, 100);
            if (status < 0 && errno == EINTR) {
                continue;
            }
            if (status < 0) {
                fail(errno);
                return;
            }
            if (status == 0) {
                continue;
            }
            if (!(item.revents & POLLIN)) {
                if (!stopping()) {
                    fail(EIO);
                }
                return;
            }

            usb_functionfs_event event = {};
            const ssize_t count = read(ep0_, &event, sizeof(event));
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count != sizeof(event)) {
                fail(count < 0 ? errno : EIO);
                return;
            }
            if (event.type == FUNCTIONFS_SETUP) {
                const int control_status = event.u.setup.bRequestType & USB_DIR_IN
                        ? static_cast<int>(write(ep0_, nullptr, 0))
                        : static_cast<int>(read(ep0_, nullptr, 0));
                if (control_status < 0) {
                    fail(errno);
                    return;
                }
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (event.type == FUNCTIONFS_ENABLE) {
                    if (enabled_) {
                        ++generation_;
                    }
                    enabled_ = true;
                } else if (event.type == FUNCTIONFS_DISABLE ||
                           event.type == FUNCTIONFS_UNBIND) {
                    if (enabled_) {
                        ++generation_;
                    }
                    enabled_ = false;
                }
            }
            condition_.notify_all();
        }
    }

    int ep0_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    bool enabled_ = false;
    bool failed_ = false;
    bool stop_ = false;
    int event_errno_ = 0;
    uint64_t generation_ = 0;
};

static inline bool functionfs_dmabuf_attach(int endpoint, int descriptor) {
    return ioctl(endpoint, FUNCTIONFS_DMABUF_ATTACH, &descriptor) == 0;
}

static inline bool functionfs_dmabuf_detach(int endpoint, int descriptor) {
    return ioctl(endpoint, FUNCTIONFS_DMABUF_DETACH, &descriptor) == 0;
}

static inline bool functionfs_dmabuf_queue(
        int endpoint, int descriptor, size_t size) {
    const usb_ffs_dmabuf_transfer_req transfer = {
        .fd = descriptor,
        .flags = 0,
        .length = size,
    };
    return ioctl(endpoint, FUNCTIONFS_DMABUF_TRANSFER, &transfer) == 0;
}

static inline bool functionfs_dmabuf_cpu_start(int descriptor) {
    dma_buf_sync sync = {
        .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW,
    };
    return ioctl(descriptor, DMA_BUF_IOCTL_SYNC, &sync) == 0;
}

static inline bool functionfs_dmabuf_cpu_end(int descriptor) {
    dma_buf_sync sync = {
        .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW,
    };
    return ioctl(descriptor, DMA_BUF_IOCTL_SYNC, &sync) == 0;
}

static inline bool functionfs_dmabuf_wait(int descriptor) {
    return functionfs_dmabuf_cpu_start(descriptor) &&
            functionfs_dmabuf_cpu_end(descriptor);
}

} // namespace ffn_split

#undef FFN_CPU_TO_LE16
#undef FFN_CPU_TO_LE32
