#define _GNU_SOURCE

#include <arpa/inet.h>
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/usb/ch9.h>
#include <linux/usb/functionfs.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_PAYLOAD_BYTES (16U * 1024U * 1024U)
#define STOP_LENGTH UINT32_MAX

struct __attribute__((packed)) descriptors {
    struct usb_functionfs_descs_head_v2 header;
    uint32_t fs_count;
    uint32_t hs_count;
    uint32_t ss_count;
    struct {
        struct usb_interface_descriptor interface;
        struct usb_endpoint_descriptor_no_audio out;
        struct usb_endpoint_descriptor_no_audio in;
    } fs, hs;
    struct {
        struct usb_interface_descriptor interface;
        struct usb_endpoint_descriptor_no_audio out;
        struct usb_ss_ep_comp_descriptor out_companion;
        struct usb_endpoint_descriptor_no_audio in;
        struct usb_ss_ep_comp_descriptor in_companion;
    } ss;
};

struct __attribute__((packed)) strings {
    struct usb_functionfs_strings_head header;
    struct {
        uint16_t code;
        char interface_name[20];
    } language;
};

static struct usb_interface_descriptor interface_descriptor(void) {
    const struct usb_interface_descriptor descriptor = {
        .bLength = USB_DT_INTERFACE_SIZE,
        .bDescriptorType = USB_DT_INTERFACE,
        .bInterfaceNumber = 0,
        .bAlternateSetting = 0,
        .bNumEndpoints = 2,
        .bInterfaceClass = USB_CLASS_VENDOR_SPEC,
        .bInterfaceSubClass = 0,
        .bInterfaceProtocol = 0,
        .iInterface = 1,
    };
    return descriptor;
}

static struct usb_endpoint_descriptor_no_audio endpoint_descriptor(
        uint8_t address, uint16_t packet_size) {
    const struct usb_endpoint_descriptor_no_audio descriptor = {
        .bLength = USB_DT_ENDPOINT_SIZE,
        .bDescriptorType = USB_DT_ENDPOINT,
        .bEndpointAddress = address,
        .bmAttributes = USB_ENDPOINT_XFER_BULK,
        .wMaxPacketSize = htole16(packet_size),
        .bInterval = 0,
    };
    return descriptor;
}

static int write_descriptors(int control) {
    struct descriptors descriptors;
    memset(&descriptors, 0, sizeof(descriptors));
    descriptors.header.magic = htole32(FUNCTIONFS_DESCRIPTORS_MAGIC_V2);
    descriptors.header.length = htole32(sizeof(descriptors));
    descriptors.header.flags = htole32(
            FUNCTIONFS_HAS_FS_DESC |
            FUNCTIONFS_HAS_HS_DESC |
            FUNCTIONFS_HAS_SS_DESC);
    descriptors.fs_count = htole32(3);
    descriptors.hs_count = htole32(3);
    descriptors.ss_count = htole32(5);

    descriptors.fs.interface = interface_descriptor();
    descriptors.fs.out = endpoint_descriptor(0x01, 64);
    descriptors.fs.in = endpoint_descriptor(0x82, 64);
    descriptors.hs.interface = interface_descriptor();
    descriptors.hs.out = endpoint_descriptor(0x01, 512);
    descriptors.hs.in = endpoint_descriptor(0x82, 512);
    descriptors.ss.interface = interface_descriptor();
    descriptors.ss.out = endpoint_descriptor(0x01, 1024);
    descriptors.ss.out_companion.bLength = USB_DT_SS_EP_COMP_SIZE;
    descriptors.ss.out_companion.bDescriptorType = USB_DT_SS_ENDPOINT_COMP;
    descriptors.ss.out_companion.bMaxBurst = 15;
    descriptors.ss.in = endpoint_descriptor(0x82, 1024);
    descriptors.ss.in_companion.bLength = USB_DT_SS_EP_COMP_SIZE;
    descriptors.ss.in_companion.bDescriptorType = USB_DT_SS_ENDPOINT_COMP;
    descriptors.ss.in_companion.bMaxBurst = 15;

    if (write(control, &descriptors, sizeof(descriptors)) !=
            (ssize_t) sizeof(descriptors)) {
        perror("write descriptors");
        return -1;
    }

    struct strings strings;
    memset(&strings, 0, sizeof(strings));
    strings.header.magic = htole32(FUNCTIONFS_STRINGS_MAGIC);
    strings.header.length = htole32(sizeof(strings));
    strings.header.str_count = htole32(1);
    strings.header.lang_count = htole32(1);
    strings.language.code = htole16(0x0409);
    memcpy(strings.language.interface_name, "S41 FunctionFS Echo", 19);
    if (write(control, &strings, sizeof(strings)) != (ssize_t) sizeof(strings)) {
        perror("write strings");
        return -1;
    }
    return 0;
}

static void * control_loop(void * argument) {
    const int control = *(const int *) argument;
    struct usb_functionfs_event events[4];
    for (;;) {
        const ssize_t count = read(control, events, sizeof(events));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return NULL;
        }
        const size_t event_count = (size_t) count / sizeof(events[0]);
        for (size_t index = 0; index < event_count; ++index) {
            fprintf(stderr, "[ffs-echo] event=%u\n", events[index].type);
        }
        fflush(stderr);
    }
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

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <functionfs-directory>\n", argv[0]);
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);

    char path[256];
    snprintf(path, sizeof(path), "%s/ep0", argv[1]);
    const int control = open(path, O_RDWR);
    if (control < 0) {
        perror("open ep0");
        return 1;
    }
    if (write_descriptors(control) != 0) {
        return 1;
    }

    snprintf(path, sizeof(path), "%s/ep1", argv[1]);
    const int out = open(path, O_RDWR);
    if (out < 0) {
        perror("open ep1");
        return 1;
    }
    snprintf(path, sizeof(path), "%s/ep2", argv[1]);
    const int in = open(path, O_RDWR);
    if (in < 0) {
        perror("open ep2");
        return 1;
    }

    pthread_t control_thread;
    if (pthread_create(&control_thread, NULL, control_loop,
                (void *) &control) != 0) {
        perror("pthread_create");
        return 1;
    }
    fprintf(stderr, "[ffs-echo] endpoints ready out=ep1 in=ep2\n");
    fflush(stderr);

    uint8_t * request = malloc(MAX_PAYLOAD_BYTES);
    uint8_t * response = calloc(1, MAX_PAYLOAD_BYTES + sizeof(uint32_t));
    if (request == NULL || response == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    uint64_t completed = 0;
    for (;;) {
        uint32_t header[2];
        if (read_exact(out, header, sizeof(header)) != 0) {
            perror("read header");
            continue;
        }
        const uint32_t request_bytes = ntohl(header[0]);
        const uint32_t response_bytes = ntohl(header[1]);
        if (request_bytes == STOP_LENGTH && response_bytes == STOP_LENGTH) {
            break;
        }
        if (request_bytes > MAX_PAYLOAD_BYTES ||
                response_bytes > MAX_PAYLOAD_BYTES) {
            fprintf(stderr, "[ffs-echo] invalid frame req=%u rsp=%u\n",
                    request_bytes, response_bytes);
            continue;
        }
        if (read_exact(out, request, request_bytes) != 0) {
            perror("read payload");
            continue;
        }
        const uint32_t encoded_response = htonl(response_bytes);
        memcpy(response, &encoded_response, sizeof(encoded_response));
        const size_t response_size = sizeof(encoded_response) + response_bytes;
        if (write(in, response, response_size) != (ssize_t) response_size) {
            perror("write ep2");
            break;
        }
        ++completed;
    }

    fprintf(stderr, "[ffs-echo] complete requests=%llu\n",
            (unsigned long long) completed);
    return 0;
}
