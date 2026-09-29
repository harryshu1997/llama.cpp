#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ffn_split {

enum class usb_host_allocator {
    malloc_buffer,
    persistent,
};

struct usb_client_config {
    uint16_t vendor_id = 0;
    uint16_t product_id = 0;
    uint8_t host_to_device_endpoint = 0;
    uint8_t device_to_host_endpoint = 0;
    size_t host_to_device_slot_bytes = 0;
    size_t device_to_host_slot_bytes = 0;
    size_t usbfs_available_bytes = 0;
    size_t slot_safety_bytes = 0;
    unsigned int configured_max_queue_depth = 1;
    unsigned int timeout_ms = 10000;
    usb_host_allocator allocator = usb_host_allocator::malloc_buffer;
    std::string transport_generation;
};

struct usb_transfer_identity {
    uint64_t request_id = 0;
    uint64_t model_id = 0;
    uint64_t operator_id = 0;
};

struct usb_transfer_record {
    usb_transfer_identity identity;
    size_t host_to_device_bytes = 0;
    size_t device_to_host_bytes = 0;
    uint64_t started_ns = 0;
    uint64_t host_to_device_completed_ns = 0;
    uint64_t device_to_host_completed_ns = 0;
    unsigned int slot_index = 0;
};

struct usb_slot_buffers {
    unsigned int slot_index = 0;
    unsigned char * host_to_device = nullptr;
    size_t host_to_device_capacity = 0;
    unsigned char * device_to_host = nullptr;
    size_t device_to_host_capacity = 0;
};

class usb_client {
public:
    explicit usb_client(usb_client_config config);
    ~usb_client();

    usb_client(const usb_client &) = delete;
    usb_client & operator=(const usb_client &) = delete;

    bool connect(std::string & error);
    void close();

    unsigned int queue_depth() const;
    unsigned int maximum_active_slots() const;
    unsigned int maximum_outstanding_transfers() const;
    bool connected() const;

    bool acquire(usb_slot_buffers & buffers, std::string & error);
    bool submit_acquired(
            const usb_slot_buffers & buffers,
            size_t host_to_device_bytes, size_t device_to_host_bytes,
            usb_transfer_identity identity,
            std::string & error);
    bool wait_acquired(
            const usb_slot_buffers & buffers,
            usb_transfer_record & record,
            std::string & error);
    bool release(const usb_slot_buffers & buffers, std::string & error);

    bool submit(
            unsigned int slot_index,
            const void * host_to_device, size_t host_to_device_bytes,
            size_t device_to_host_bytes,
            usb_transfer_identity identity,
            std::string & error);
    bool wait(
            unsigned int slot_index,
            void * device_to_host, size_t device_to_host_capacity,
            usb_transfer_record & record,
            std::string & error);
    bool wait_any(
            unsigned int & slot_index,
            void * device_to_host, size_t device_to_host_capacity,
            usb_transfer_record & record,
            std::string & error);
    bool exchange(
            const void * host_to_device, size_t host_to_device_bytes,
            void * device_to_host, size_t device_to_host_bytes,
            usb_transfer_identity identity,
            usb_transfer_record & record,
            std::string & error);

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

size_t usbfs_memory_bytes();
unsigned int usb_adaptive_queue_depth(const usb_client_config & config);
const char * usb_host_allocator_name(usb_host_allocator allocator);
bool parse_usb_host_allocator(
        const std::string & text, usb_host_allocator & allocator);

} // namespace ffn_split
