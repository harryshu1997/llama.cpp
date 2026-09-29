#ifndef S41_LIBUSB_ABI_H
#define S41_LIBUSB_ABI_H

// Minimal public libusb 1.0 ABI used by the bounded AOA transport probe.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

typedef void (*libusb_transfer_cb_fn)(struct libusb_transfer * transfer);

struct libusb_iso_packet_descriptor {
    unsigned int length;
    unsigned int actual_length;
    enum libusb_transfer_status status;
};

struct libusb_transfer {
    struct libusb_device_handle * dev_handle;
    uint8_t flags;
    unsigned char endpoint;
    unsigned char type;
    unsigned int timeout;
    enum libusb_transfer_status status;
    int length;
    int actual_length;
    libusb_transfer_cb_fn callback;
    void * user_data;
    unsigned char * buffer;
    int num_iso_packets;
    struct libusb_iso_packet_descriptor iso_packet_desc[];
};

int libusb_init(struct libusb_context ** context);
void libusb_exit(struct libusb_context * context);
struct libusb_device_handle * libusb_open_device_with_vid_pid(
        struct libusb_context * context, uint16_t vendor_id,
        uint16_t product_id);
void libusb_close(struct libusb_device_handle * handle);
int libusb_detach_kernel_driver(struct libusb_device_handle * handle,
        int interface_number);
int libusb_set_auto_detach_kernel_driver(
        struct libusb_device_handle * handle, int enable);
int libusb_claim_interface(struct libusb_device_handle * handle,
        int interface_number);
int libusb_release_interface(struct libusb_device_handle * handle,
        int interface_number);
int libusb_bulk_transfer(struct libusb_device_handle * handle,
        unsigned char endpoint, unsigned char * data, int length,
        int * transferred, unsigned int timeout);
struct libusb_transfer * libusb_alloc_transfer(int iso_packets);
void libusb_free_transfer(struct libusb_transfer * transfer);
int libusb_submit_transfer(struct libusb_transfer * transfer);
int libusb_cancel_transfer(struct libusb_transfer * transfer);
int libusb_handle_events(struct libusb_context * context);
const char * libusb_error_name(int error_code);

#ifdef __cplusplus
}
#endif

#endif
