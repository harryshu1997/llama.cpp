#include "ffn-split-usb-client.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

extern "C" {

struct libusb_context;
struct libusb_device_handle;
struct libusb_transfer;

enum libusb_transfer_type {
    LIBUSB_TRANSFER_TYPE_CONTROL = 0,
    LIBUSB_TRANSFER_TYPE_ISOCHRONOUS = 1,
    LIBUSB_TRANSFER_TYPE_BULK = 2,
    LIBUSB_TRANSFER_TYPE_INTERRUPT = 3,
    LIBUSB_TRANSFER_TYPE_BULK_STREAM = 4,
};

enum libusb_transfer_status {
    LIBUSB_TRANSFER_COMPLETED = 0,
    LIBUSB_TRANSFER_ERROR,
    LIBUSB_TRANSFER_TIMED_OUT,
    LIBUSB_TRANSFER_CANCELLED,
    LIBUSB_TRANSFER_STALL,
    LIBUSB_TRANSFER_NO_DEVICE,
    LIBUSB_TRANSFER_OVERFLOW,
};

using libusb_transfer_cb_fn = void (*)(libusb_transfer * transfer);

struct libusb_iso_packet_descriptor {
    unsigned int length;
    unsigned int actual_length;
    libusb_transfer_status status;
};

struct libusb_transfer {
    libusb_device_handle * dev_handle;
    uint8_t flags;
    unsigned char endpoint;
    unsigned char type;
    unsigned int timeout;
    libusb_transfer_status status;
    int length;
    int actual_length;
    libusb_transfer_cb_fn callback;
    void * user_data;
    unsigned char * buffer;
    int num_iso_packets;
    libusb_iso_packet_descriptor iso_packet_desc[];
};

int libusb_init(libusb_context ** context);
void libusb_exit(libusb_context * context);
libusb_device_handle * libusb_open_device_with_vid_pid(
        libusb_context * context, uint16_t vendor_id, uint16_t product_id);
void libusb_close(libusb_device_handle * handle);
int libusb_set_auto_detach_kernel_driver(
        libusb_device_handle * handle, int enable);
int libusb_detach_kernel_driver(
        libusb_device_handle * handle, int interface_number);
int libusb_claim_interface(
        libusb_device_handle * handle, int interface_number);
int libusb_release_interface(
        libusb_device_handle * handle, int interface_number);
libusb_transfer * libusb_alloc_transfer(int iso_packets);
void libusb_free_transfer(libusb_transfer * transfer);
int libusb_submit_transfer(libusb_transfer * transfer);
int libusb_cancel_transfer(libusb_transfer * transfer);
int libusb_handle_events(libusb_context * context);
const char * libusb_error_name(int error_code);
unsigned char * libusb_dev_mem_alloc(
        libusb_device_handle * handle, size_t length);
int libusb_dev_mem_free(
        libusb_device_handle * handle, unsigned char * buffer, size_t length);

} // extern "C"

namespace ffn_split {
namespace {

uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

class transfer_buffer {
public:
    transfer_buffer(
            libusb_device_handle * handle, size_t size,
            usb_host_allocator allocator) :
            handle_(handle), size_(size), allocator_(allocator) {
        if (allocator_ == usb_host_allocator::persistent) {
            data_ = libusb_dev_mem_alloc(handle_, size_);
        } else {
            owned_.resize(size_);
            data_ = owned_.data();
        }
        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
    }

    ~transfer_buffer() {
        if (allocator_ == usb_host_allocator::persistent && data_ != nullptr) {
            libusb_dev_mem_free(handle_, data_, size_);
        }
    }

    transfer_buffer(const transfer_buffer &) = delete;
    transfer_buffer & operator=(const transfer_buffer &) = delete;

    unsigned char * data() { return data_; }
    size_t size() const { return size_; }

private:
    libusb_device_handle * handle_ = nullptr;
    size_t size_ = 0;
    usb_host_allocator allocator_ = usb_host_allocator::malloc_buffer;
    unsigned char * data_ = nullptr;
    std::vector<unsigned char> owned_;
};

struct transfer_slot;

struct transfer_tag {
    transfer_slot * slot = nullptr;
    bool device_to_host = false;
};

struct transfer_slot {
    transfer_slot(
            libusb_device_handle * handle, const usb_client_config & config,
            unsigned int index) :
            host_to_device(
                    handle, config.host_to_device_slot_bytes,
                    config.allocator),
            device_to_host(
                    handle, config.device_to_host_slot_bytes,
                    config.allocator),
            slot_index(index) {
        host_to_device_tag = { this, false };
        device_to_host_tag = { this, true };
    }

    transfer_buffer host_to_device;
    transfer_buffer device_to_host;
    libusb_transfer * host_to_device_transfer = nullptr;
    libusb_transfer * device_to_host_transfer = nullptr;
    transfer_tag host_to_device_tag;
    transfer_tag device_to_host_tag;
    usb_transfer_identity identity;
    size_t host_to_device_bytes = 0;
    size_t device_to_host_bytes = 0;
    uint64_t started_ns = 0;
    uint64_t host_to_device_completed_ns = 0;
    uint64_t device_to_host_completed_ns = 0;
    unsigned int slot_index = 0;
    bool active = false;
    bool reserved = false;
    bool host_to_device_done = false;
    bool device_to_host_done = false;
    bool failed = false;
    int failure_status = 0;
};

void transfer_completed(libusb_transfer * transfer) {
    auto * tag = static_cast<transfer_tag *>(transfer->user_data);
    transfer_slot * slot = tag->slot;
    const uint64_t completed_ns = now_ns();
    if (transfer->status != LIBUSB_TRANSFER_COMPLETED ||
        transfer->actual_length != transfer->length) {
        slot->failed = true;
        slot->failure_status = static_cast<int>(transfer->status);
    }
    if (tag->device_to_host) {
        slot->device_to_host_done = true;
        slot->device_to_host_completed_ns = completed_ns;
    } else {
        slot->host_to_device_done = true;
        slot->host_to_device_completed_ns = completed_ns;
    }
}

void configure_transfer(
        libusb_transfer * transfer, libusb_device_handle * handle,
        uint8_t endpoint, unsigned char * buffer, size_t size,
        unsigned int timeout_ms, transfer_tag * tag) {
    transfer->dev_handle = handle;
    transfer->flags = 0;
    transfer->endpoint = endpoint;
    transfer->type = LIBUSB_TRANSFER_TYPE_BULK;
    transfer->timeout = timeout_ms;
    transfer->length = static_cast<int>(size);
    transfer->callback = transfer_completed;
    transfer->user_data = tag;
    transfer->buffer = buffer;
    transfer->num_iso_packets = 0;
}

bool valid_config(const usb_client_config & config) {
    return config.vendor_id != 0 && config.product_id != 0 &&
            config.host_to_device_endpoint != 0 &&
            config.device_to_host_endpoint != 0 &&
            config.host_to_device_endpoint != config.device_to_host_endpoint &&
            config.host_to_device_slot_bytes > 0 &&
            config.device_to_host_slot_bytes > 0 &&
            config.host_to_device_slot_bytes <=
                    static_cast<size_t>(INT_MAX) &&
            config.device_to_host_slot_bytes <=
                    static_cast<size_t>(INT_MAX) &&
            config.configured_max_queue_depth > 0 &&
            config.configured_max_queue_depth <= 64 &&
            config.timeout_ms > 0 && !config.transport_generation.empty();
}

std::string usb_error(const char * operation, int status) {
    return std::string(operation) + ": " + libusb_error_name(status);
}

} // namespace

struct usb_client::implementation {
    explicit implementation(usb_client_config value) :
            config(std::move(value)) {}

    usb_client_config config;
    libusb_context * context = nullptr;
    libusb_device_handle * handle = nullptr;
    std::vector<std::unique_ptr<transfer_slot>> slots;
    unsigned int next_exchange_slot = 0;
    unsigned int maximum_active_slots = 0;
    unsigned int maximum_outstanding_transfers = 0;

    bool complete(const transfer_slot & slot) const {
        return slot.active && slot.host_to_device_done &&
                slot.device_to_host_done;
    }

    void cancel_all() {
        if (context == nullptr) {
            return;
        }
        for (const auto & slot : slots) {
            if (slot->active && !slot->host_to_device_done &&
                slot->host_to_device_bytes != 0) {
                libusb_cancel_transfer(slot->host_to_device_transfer);
            }
            if (slot->active && !slot->device_to_host_done &&
                slot->device_to_host_bytes != 0) {
                libusb_cancel_transfer(slot->device_to_host_transfer);
            }
        }
        for (;;) {
            bool pending = false;
            for (const auto & slot : slots) {
                pending = pending || (slot->active &&
                        (!slot->host_to_device_done ||
                         !slot->device_to_host_done));
            }
            if (!pending) {
                return;
            }
            if (libusb_handle_events(context) != 0) {
                return;
            }
        }
    }

    void reset_slot(transfer_slot & slot, bool release_reservation = true) {
        slot.identity = {};
        slot.host_to_device_bytes = 0;
        slot.device_to_host_bytes = 0;
        slot.started_ns = 0;
        slot.host_to_device_completed_ns = 0;
        slot.device_to_host_completed_ns = 0;
        slot.active = false;
        slot.host_to_device_done = false;
        slot.device_to_host_done = false;
        slot.failed = false;
        slot.failure_status = 0;
        if (release_reservation) {
            slot.reserved = false;
        }
    }

    bool start(
            transfer_slot & slot,
            const void * host_to_device, size_t host_to_device_bytes,
            size_t device_to_host_bytes,
            usb_transfer_identity identity, bool copy_input,
            std::string & error) {
        if (slot.active ||
            (host_to_device_bytes == 0 && device_to_host_bytes == 0) ||
            (copy_input && host_to_device_bytes != 0 &&
             host_to_device == nullptr) ||
            host_to_device_bytes > slot.host_to_device.size() ||
            device_to_host_bytes > slot.device_to_host.size()) {
            error = "invalid USB transfer submission";
            return false;
        }
        if (copy_input && host_to_device_bytes != 0) {
            std::memcpy(
                    slot.host_to_device.data(), host_to_device,
                    host_to_device_bytes);
        }
        slot.identity = identity;
        slot.host_to_device_bytes = host_to_device_bytes;
        slot.device_to_host_bytes = device_to_host_bytes;
        slot.started_ns = now_ns();
        slot.host_to_device_completed_ns = host_to_device_bytes == 0 ?
                slot.started_ns : 0;
        slot.device_to_host_completed_ns = device_to_host_bytes == 0 ?
                slot.started_ns : 0;
        slot.host_to_device_done = host_to_device_bytes == 0;
        slot.device_to_host_done = device_to_host_bytes == 0;
        slot.failed = false;
        slot.failure_status = 0;
        slot.active = true;

        if (device_to_host_bytes != 0) {
            configure_transfer(
                    slot.device_to_host_transfer, handle,
                    config.device_to_host_endpoint,
                    slot.device_to_host.data(), device_to_host_bytes,
                    config.timeout_ms, &slot.device_to_host_tag);
            const int status = libusb_submit_transfer(
                    slot.device_to_host_transfer);
            if (status != 0) {
                reset_slot(slot, !slot.reserved);
                error = usb_error("submit USB D2H failed", status);
                return false;
            }
        }

        if (host_to_device_bytes != 0) {
            configure_transfer(
                    slot.host_to_device_transfer, handle,
                    config.host_to_device_endpoint,
                    slot.host_to_device.data(), host_to_device_bytes,
                    config.timeout_ms, &slot.host_to_device_tag);
            const int status = libusb_submit_transfer(
                    slot.host_to_device_transfer);
            if (status != 0) {
                if (device_to_host_bytes != 0) {
                    libusb_cancel_transfer(slot.device_to_host_transfer);
                    while (!slot.device_to_host_done) {
                        libusb_handle_events(context);
                    }
                }
                reset_slot(slot, !slot.reserved);
                error = usb_error("submit USB H2D failed", status);
                return false;
            }
        }
        unsigned int active_slots = 0;
        unsigned int outstanding_transfers = 0;
        for (const auto & current : slots) {
            if (!current->active) {
                continue;
            }
            ++active_slots;
            outstanding_transfers += current->host_to_device_done ? 0 : 1;
            outstanding_transfers += current->device_to_host_done ? 0 : 1;
        }
        maximum_active_slots = std::max(maximum_active_slots, active_slots);
        maximum_outstanding_transfers = std::max(
                maximum_outstanding_transfers, outstanding_transfers);
        return true;
    }
};

size_t usbfs_memory_bytes() {
    std::ifstream input("/sys/module/usbcore/parameters/usbfs_memory_mb");
    unsigned long long memory_mb = 0;
    if (!(input >> memory_mb) ||
        memory_mb > std::numeric_limits<size_t>::max() / (1024U * 1024U)) {
        return 0;
    }
    return static_cast<size_t>(memory_mb) * 1024U * 1024U;
}

unsigned int usb_adaptive_queue_depth(const usb_client_config & config) {
    if (config.configured_max_queue_depth == 0) {
        return 0;
    }
    const size_t available = config.usbfs_available_bytes != 0 ?
            config.usbfs_available_bytes : usbfs_memory_bytes();
    if (available == 0) {
        return 1;
    }
    if (config.host_to_device_slot_bytes >
            std::numeric_limits<size_t>::max() -
                    config.device_to_host_slot_bytes) {
        return 0;
    }
    const size_t payload_bytes = config.host_to_device_slot_bytes +
            config.device_to_host_slot_bytes;
    if (payload_bytes > std::numeric_limits<size_t>::max() -
            config.slot_safety_bytes) {
        return 0;
    }
    const size_t slot_bytes = payload_bytes + config.slot_safety_bytes;
    if (slot_bytes == 0 || available < slot_bytes) {
        return 0;
    }
    return std::min<unsigned int>(
            config.configured_max_queue_depth,
            static_cast<unsigned int>(std::min<size_t>(
                    available / slot_bytes,
                    std::numeric_limits<unsigned int>::max())));
}

const char * usb_host_allocator_name(usb_host_allocator allocator) {
    switch (allocator) {
        case usb_host_allocator::malloc_buffer:
            return "malloc";
        case usb_host_allocator::persistent:
            return "devmem";
    }
    return "invalid";
}

bool parse_usb_host_allocator(
        const std::string & text, usb_host_allocator & allocator) {
    if (text == "malloc") {
        allocator = usb_host_allocator::malloc_buffer;
        return true;
    }
    if (text == "devmem") {
        allocator = usb_host_allocator::persistent;
        return true;
    }
    return false;
}

usb_client::usb_client(usb_client_config config) :
        impl_(std::make_unique<implementation>(std::move(config))) {}

usb_client::~usb_client() {
    close();
}

bool usb_client::connect(std::string & error) {
    if (connected()) {
        error = "USB client is already connected";
        return false;
    }
    if (!valid_config(impl_->config)) {
        error = "invalid USB client configuration";
        return false;
    }
    const unsigned int depth = usb_adaptive_queue_depth(impl_->config);
    if (depth == 0) {
        error = "USB slot buffers exceed the available usbfs memory";
        return false;
    }
    int status = libusb_init(&impl_->context);
    if (status != 0) {
        error = usb_error("libusb_init failed", status);
        close();
        return false;
    }
    for (int attempt = 0; impl_->handle == nullptr && attempt < 600; ++attempt) {
        impl_->handle = libusb_open_device_with_vid_pid(
                impl_->context, impl_->config.vendor_id,
                impl_->config.product_id);
        if (impl_->handle == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    if (impl_->handle == nullptr) {
        error = "FunctionFS USB device was not found";
        close();
        return false;
    }
    libusb_set_auto_detach_kernel_driver(impl_->handle, 1);
    libusb_detach_kernel_driver(impl_->handle, 0);
    status = libusb_claim_interface(impl_->handle, 0);
    if (status != 0) {
        error = usb_error("USB interface claim failed", status);
        close();
        return false;
    }
    try {
        impl_->slots.reserve(depth);
        for (unsigned int index = 0; index < depth; ++index) {
            auto slot = std::make_unique<transfer_slot>(
                    impl_->handle, impl_->config, index);
            slot->host_to_device_transfer = libusb_alloc_transfer(0);
            slot->device_to_host_transfer = libusb_alloc_transfer(0);
            if (slot->host_to_device_transfer == nullptr ||
                slot->device_to_host_transfer == nullptr) {
                if (slot->host_to_device_transfer != nullptr) {
                    libusb_free_transfer(slot->host_to_device_transfer);
                }
                if (slot->device_to_host_transfer != nullptr) {
                    libusb_free_transfer(slot->device_to_host_transfer);
                }
                error = "libusb transfer allocation failed";
                close();
                return false;
            }
            impl_->slots.push_back(std::move(slot));
        }
    } catch (const std::bad_alloc &) {
        error = "USB host buffer allocation failed";
        close();
        return false;
    }
    return true;
}

void usb_client::close() {
    impl_->cancel_all();
    for (auto & slot : impl_->slots) {
        if (slot->host_to_device_transfer != nullptr) {
            libusb_free_transfer(slot->host_to_device_transfer);
        }
        if (slot->device_to_host_transfer != nullptr) {
            libusb_free_transfer(slot->device_to_host_transfer);
        }
    }
    impl_->slots.clear();
    if (impl_->handle != nullptr) {
        libusb_release_interface(impl_->handle, 0);
        libusb_close(impl_->handle);
        impl_->handle = nullptr;
    }
    if (impl_->context != nullptr) {
        libusb_exit(impl_->context);
        impl_->context = nullptr;
    }
    impl_->next_exchange_slot = 0;
    impl_->maximum_active_slots = 0;
    impl_->maximum_outstanding_transfers = 0;
}

unsigned int usb_client::queue_depth() const {
    return static_cast<unsigned int>(impl_->slots.size());
}

unsigned int usb_client::maximum_active_slots() const {
    return impl_->maximum_active_slots;
}

unsigned int usb_client::maximum_outstanding_transfers() const {
    return impl_->maximum_outstanding_transfers;
}

bool usb_client::connected() const {
    return impl_->handle != nullptr && !impl_->slots.empty();
}

bool usb_client::acquire(
        usb_slot_buffers & buffers, std::string & error) {
    if (!connected()) {
        error = "USB client is not connected";
        return false;
    }
    for (unsigned int attempt = 0; attempt < queue_depth(); ++attempt) {
        const unsigned int index =
                (impl_->next_exchange_slot + attempt) % queue_depth();
        transfer_slot & slot = *impl_->slots[index];
        if (slot.active || slot.reserved) {
            continue;
        }
        slot.reserved = true;
        impl_->next_exchange_slot = (index + 1) % queue_depth();
        buffers.slot_index = index;
        buffers.host_to_device = slot.host_to_device.data();
        buffers.host_to_device_capacity = slot.host_to_device.size();
        buffers.device_to_host = slot.device_to_host.data();
        buffers.device_to_host_capacity = slot.device_to_host.size();
        return true;
    }
    error = "all USB transfer slots are busy";
    return false;
}

bool usb_client::submit_acquired(
        const usb_slot_buffers & buffers,
        size_t host_to_device_bytes, size_t device_to_host_bytes,
        usb_transfer_identity identity,
        std::string & error) {
    if (!connected() || buffers.slot_index >= impl_->slots.size()) {
        error = "USB transfer slot is unavailable";
        return false;
    }
    transfer_slot & slot = *impl_->slots[buffers.slot_index];
    if (!slot.reserved || buffers.host_to_device != slot.host_to_device.data() ||
        buffers.device_to_host != slot.device_to_host.data() ||
        buffers.host_to_device_capacity != slot.host_to_device.size() ||
        buffers.device_to_host_capacity != slot.device_to_host.size()) {
        error = "USB transfer slot identity differs";
        return false;
    }
    return impl_->start(
            slot, nullptr, host_to_device_bytes, device_to_host_bytes,
            identity, false, error);
}

bool usb_client::wait_acquired(
        const usb_slot_buffers & buffers,
        usb_transfer_record & record,
        std::string & error) {
    if (!connected() || buffers.slot_index >= impl_->slots.size()) {
        error = "USB transfer slot is unavailable";
        return false;
    }
    transfer_slot & slot = *impl_->slots[buffers.slot_index];
    if (!slot.reserved || !slot.active ||
        buffers.host_to_device != slot.host_to_device.data() ||
        buffers.device_to_host != slot.device_to_host.data()) {
        error = "USB transfer slot is not an active reservation";
        return false;
    }
    while (!impl_->complete(slot)) {
        const int status = libusb_handle_events(impl_->context);
        if (status != 0) {
            error = usb_error("USB event handling failed", status);
            impl_->cancel_all();
            impl_->reset_slot(slot, false);
            return false;
        }
    }
    if (slot.failed) {
        error = "asynchronous USB transfer failed with status " +
                std::to_string(slot.failure_status);
        impl_->reset_slot(slot, false);
        return false;
    }
    record.identity = slot.identity;
    record.host_to_device_bytes = slot.host_to_device_bytes;
    record.device_to_host_bytes = slot.device_to_host_bytes;
    record.started_ns = slot.started_ns;
    record.host_to_device_completed_ns =
            slot.host_to_device_completed_ns;
    record.device_to_host_completed_ns =
            slot.device_to_host_completed_ns;
    record.slot_index = slot.slot_index;
    impl_->reset_slot(slot, false);
    return true;
}

bool usb_client::release(
        const usb_slot_buffers & buffers, std::string & error) {
    if (!connected() || buffers.slot_index >= impl_->slots.size()) {
        error = "USB transfer slot is unavailable";
        return false;
    }
    transfer_slot & slot = *impl_->slots[buffers.slot_index];
    if (!slot.reserved || slot.active ||
        buffers.host_to_device != slot.host_to_device.data() ||
        buffers.device_to_host != slot.device_to_host.data()) {
        error = "USB transfer slot cannot be released";
        return false;
    }
    slot.reserved = false;
    return true;
}

bool usb_client::submit(
        unsigned int slot_index,
        const void * host_to_device, size_t host_to_device_bytes,
        size_t device_to_host_bytes,
        usb_transfer_identity identity,
        std::string & error) {
    if (!connected() || slot_index >= impl_->slots.size()) {
        error = "USB transfer slot is unavailable";
        return false;
    }
    transfer_slot & slot = *impl_->slots[slot_index];
    if (slot.reserved) {
        error = "USB transfer slot is reserved";
        return false;
    }
    return impl_->start(
            slot, host_to_device, host_to_device_bytes,
            device_to_host_bytes, identity, true, error);
}

bool usb_client::wait(
        unsigned int slot_index,
        void * device_to_host, size_t device_to_host_capacity,
        usb_transfer_record & record,
        std::string & error) {
    if (!connected() || slot_index >= impl_->slots.size()) {
        error = "USB transfer slot is unavailable";
        return false;
    }
    transfer_slot & slot = *impl_->slots[slot_index];
    if (!slot.active ||
        (slot.device_to_host_bytes != 0 &&
         (device_to_host == nullptr ||
          device_to_host_capacity < slot.device_to_host_bytes))) {
        error = "invalid USB transfer completion buffer";
        return false;
    }
    while (!impl_->complete(slot)) {
        const int status = libusb_handle_events(impl_->context);
        if (status != 0) {
            error = usb_error("USB event handling failed", status);
            impl_->cancel_all();
            return false;
        }
    }
    if (slot.failed) {
        error = "asynchronous USB transfer failed with status " +
                std::to_string(slot.failure_status);
        impl_->reset_slot(slot);
        return false;
    }
    if (slot.device_to_host_bytes != 0) {
        std::memcpy(
                device_to_host, slot.device_to_host.data(),
                slot.device_to_host_bytes);
    }
    record.identity = slot.identity;
    record.host_to_device_bytes = slot.host_to_device_bytes;
    record.device_to_host_bytes = slot.device_to_host_bytes;
    record.started_ns = slot.started_ns;
    record.host_to_device_completed_ns =
            slot.host_to_device_completed_ns;
    record.device_to_host_completed_ns =
            slot.device_to_host_completed_ns;
    record.slot_index = slot.slot_index;
    impl_->reset_slot(slot);
    return true;
}

bool usb_client::wait_any(
        unsigned int & slot_index,
        void * device_to_host, size_t device_to_host_capacity,
        usb_transfer_record & record,
        std::string & error) {
    if (!connected()) {
        error = "USB client is not connected";
        return false;
    }
    for (;;) {
        for (unsigned int index = 0; index < impl_->slots.size(); ++index) {
            if (impl_->complete(*impl_->slots[index])) {
                slot_index = index;
                return wait(
                        index, device_to_host, device_to_host_capacity,
                        record, error);
            }
        }
        const int status = libusb_handle_events(impl_->context);
        if (status != 0) {
            error = usb_error("USB event handling failed", status);
            impl_->cancel_all();
            return false;
        }
    }
}

bool usb_client::exchange(
        const void * host_to_device, size_t host_to_device_bytes,
        void * device_to_host, size_t device_to_host_bytes,
        usb_transfer_identity identity,
        usb_transfer_record & record,
        std::string & error) {
    if (!connected()) {
        error = "USB client is not connected";
        return false;
    }
    const unsigned int depth = queue_depth();
    for (unsigned int attempt = 0; attempt < depth; ++attempt) {
        const unsigned int index =
                (impl_->next_exchange_slot + attempt) % depth;
        if (!impl_->slots[index]->active) {
            impl_->next_exchange_slot = (index + 1) % depth;
            return submit(
                        index, host_to_device, host_to_device_bytes,
                        device_to_host_bytes, identity, error) &&
                    wait(
                        index, device_to_host, device_to_host_bytes,
                        record, error);
        }
    }
    error = "all USB transfer slots are busy";
    return false;
}

} // namespace ffn_split
