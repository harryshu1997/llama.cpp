#define _GNU_SOURCE

#include "aoa_async_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_TRANSFER_BYTES (16U * 1024U * 1024U)
#define MAX_RING_DEPTH 32U

enum slot_state {
    SLOT_EMPTY = 0,
    SLOT_READY = 1,
};

struct slot {
    uint8_t * request;
    uint8_t * response;
    enum slot_state state;
};

struct buffered_context {
    int descriptor;
    size_t request_bytes;
    size_t response_bytes;
    uint64_t total_requests;
    uint64_t warmup;
    unsigned int depth;
    struct slot * slots;
    uint64_t * read_ns;
    uint64_t * write_ns;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int failed;
};

static uint64_t now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t) value.tv_sec * UINT64_C(1000000000) +
        (uint64_t) value.tv_nsec;
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

static int validate_request(const uint8_t * data, size_t size,
        size_t request_bytes, size_t response_bytes, uint64_t sequence) {
    if (size < sizeof(struct s41_aoa_frame_header)) {
        return -1;
    }
    struct s41_aoa_frame_header header;
    memcpy(&header, data, sizeof(header));
    uint64_t trailing;
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
    struct s41_aoa_frame_header header = {
        .magic = S41_AOA_RESPONSE_MAGIC,
        .sequence = sequence,
        .request_bytes = (uint32_t) request_bytes,
        .response_bytes = (uint32_t) response_bytes,
        .sentinel = s41_aoa_sentinel(sequence),
    };
    memcpy(data, &header, sizeof(header));
    const uint64_t trailing = s41_aoa_sentinel(sequence);
    memcpy(data + size - sizeof(trailing), &trailing, sizeof(trailing));
}

static int compare_u64(const void * left, const void * right) {
    const uint64_t a = *(const uint64_t *) left;
    const uint64_t b = *(const uint64_t *) right;
    return (a > b) - (a < b);
}

static void print_distribution(const char * name, uint64_t * values,
        uint64_t count) {
    qsort(values, (size_t) count, sizeof(*values), compare_u64);
    const uint64_t median = values[count / 2];
    const uint64_t p90 = values[(count * 90) / 100];
    const uint64_t p99 = values[(count * 99) / 100];
    fprintf(stderr, "[aoa-buffer] %s_us median=%.3f p90=%.3f p99=%.3f\n",
            name, median / 1000.0, p90 / 1000.0, p99 / 1000.0);
}

static int run_serial(int descriptor, size_t request_bytes,
        size_t response_bytes, uint64_t total_requests, uint64_t warmup) {
    uint8_t * request = calloc(1, request_bytes);
    uint8_t * response = calloc(1, response_bytes);
    uint64_t * read_ns = calloc((size_t) total_requests, sizeof(*read_ns));
    uint64_t * write_ns = calloc((size_t) total_requests, sizeof(*write_ns));
    if (request == NULL || response == NULL || read_ns == NULL ||
            write_ns == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    for (uint64_t index = 0; index < total_requests; ++index) {
        const uint64_t sequence = index + 1;
        uint64_t started = now_ns();
        if (read_exact(descriptor, request, request_bytes) != 0) {
            fprintf(stderr, "read failed at request %llu: %s\n",
                    (unsigned long long) sequence, strerror(errno));
            return 1;
        }
        read_ns[index] = now_ns() - started;
        if (validate_request(request, request_bytes, request_bytes,
                response_bytes, sequence) != 0) {
            fprintf(stderr, "request validation failed at %llu\n",
                    (unsigned long long) sequence);
            return 1;
        }
        prepare_response(response, response_bytes, request_bytes,
                response_bytes, sequence);
        started = now_ns();
        if (write_exact(descriptor, response, response_bytes) != 0) {
            fprintf(stderr, "write failed at request %llu: %s\n",
                    (unsigned long long) sequence, strerror(errno));
            return 1;
        }
        write_ns[index] = now_ns() - started;
    }
    print_distribution("read", read_ns + warmup, total_requests - warmup);
    print_distribution("write", write_ns + warmup, total_requests - warmup);
    free(write_ns);
    free(read_ns);
    free(response);
    free(request);
    return 0;
}

static void set_failed(struct buffered_context * context) {
    pthread_mutex_lock(&context->mutex);
    context->failed = 1;
    pthread_cond_broadcast(&context->condition);
    pthread_mutex_unlock(&context->mutex);
}

static void * reader_main(void * opaque) {
    struct buffered_context * context = opaque;
    for (uint64_t index = 0; index < context->total_requests; ++index) {
        struct slot * slot = &context->slots[index % context->depth];
        pthread_mutex_lock(&context->mutex);
        while (slot->state != SLOT_EMPTY && !context->failed) {
            pthread_cond_wait(&context->condition, &context->mutex);
        }
        const int failed = context->failed;
        pthread_mutex_unlock(&context->mutex);
        if (failed) {
            return NULL;
        }

        const uint64_t started = now_ns();
        if (read_exact(context->descriptor, slot->request,
                context->request_bytes) != 0) {
            fprintf(stderr, "buffered read failed at %llu: %s\n",
                    (unsigned long long) (index + 1), strerror(errno));
            set_failed(context);
            return NULL;
        }
        context->read_ns[index] = now_ns() - started;
        if (validate_request(slot->request, context->request_bytes,
                context->request_bytes, context->response_bytes,
                index + 1) != 0) {
            fprintf(stderr, "buffered validation failed at %llu\n",
                    (unsigned long long) (index + 1));
            set_failed(context);
            return NULL;
        }
        prepare_response(slot->response, context->response_bytes,
                context->request_bytes, context->response_bytes, index + 1);

        pthread_mutex_lock(&context->mutex);
        slot->state = SLOT_READY;
        pthread_cond_broadcast(&context->condition);
        pthread_mutex_unlock(&context->mutex);
    }
    return NULL;
}

static void * writer_main(void * opaque) {
    struct buffered_context * context = opaque;
    for (uint64_t index = 0; index < context->total_requests; ++index) {
        struct slot * slot = &context->slots[index % context->depth];
        pthread_mutex_lock(&context->mutex);
        while (slot->state != SLOT_READY && !context->failed) {
            pthread_cond_wait(&context->condition, &context->mutex);
        }
        const int failed = context->failed;
        pthread_mutex_unlock(&context->mutex);
        if (failed) {
            return NULL;
        }

        const uint64_t started = now_ns();
        if (write_exact(context->descriptor, slot->response,
                context->response_bytes) != 0) {
            fprintf(stderr, "buffered write failed at %llu: %s\n",
                    (unsigned long long) (index + 1), strerror(errno));
            set_failed(context);
            return NULL;
        }
        context->write_ns[index] = now_ns() - started;

        pthread_mutex_lock(&context->mutex);
        slot->state = SLOT_EMPTY;
        pthread_cond_broadcast(&context->condition);
        pthread_mutex_unlock(&context->mutex);
    }
    return NULL;
}

static int run_buffered(int descriptor, size_t request_bytes,
        size_t response_bytes, uint64_t total_requests, uint64_t warmup,
        unsigned int depth) {
    struct buffered_context context = {
        .descriptor = descriptor,
        .request_bytes = request_bytes,
        .response_bytes = response_bytes,
        .total_requests = total_requests,
        .warmup = warmup,
        .depth = depth,
    };
    context.slots = calloc(depth, sizeof(*context.slots));
    context.read_ns = calloc((size_t) total_requests, sizeof(*context.read_ns));
    context.write_ns = calloc((size_t) total_requests, sizeof(*context.write_ns));
    if (context.slots == NULL || context.read_ns == NULL ||
            context.write_ns == NULL) {
        fprintf(stderr, "buffer allocation failed\n");
        return 1;
    }
    for (unsigned int index = 0; index < depth; ++index) {
        context.slots[index].request = calloc(1, request_bytes);
        context.slots[index].response = calloc(1, response_bytes);
        if (context.slots[index].request == NULL ||
                context.slots[index].response == NULL) {
            fprintf(stderr, "ring allocation failed\n");
            return 1;
        }
    }
    pthread_mutex_init(&context.mutex, NULL);
    pthread_cond_init(&context.condition, NULL);

    pthread_t reader;
    pthread_t writer;
    if (pthread_create(&reader, NULL, reader_main, &context) != 0 ||
            pthread_create(&writer, NULL, writer_main, &context) != 0) {
        fprintf(stderr, "thread creation failed\n");
        return 1;
    }
    pthread_join(reader, NULL);
    pthread_join(writer, NULL);

    if (!context.failed) {
        print_distribution("read", context.read_ns + context.warmup,
                total_requests - context.warmup);
        print_distribution("write", context.write_ns + context.warmup,
                total_requests - context.warmup);
    }
    pthread_cond_destroy(&context.condition);
    pthread_mutex_destroy(&context.mutex);
    for (unsigned int index = 0; index < depth; ++index) {
        free(context.slots[index].response);
        free(context.slots[index].request);
    }
    free(context.write_ns);
    free(context.read_ns);
    free(context.slots);
    return context.failed ? 1 : 0;
}

static int open_accessory(void) {
    for (unsigned int attempt = 0; attempt < 600; ++attempt) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor >= 0) {
            return descriptor;
        }
        usleep(100000);
    }
    return -1;
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s <serial|buffered> <request_bytes> "
                "<response_bytes> <total_requests> <warmup> <ring_depth>\n",
                argv[0]);
        return 2;
    }
    const char * mode = argv[1];
    const size_t request_bytes = (size_t) strtoull(argv[2], NULL, 10);
    const size_t response_bytes = (size_t) strtoull(argv[3], NULL, 10);
    const uint64_t total_requests = strtoull(argv[4], NULL, 10);
    const uint64_t warmup = strtoull(argv[5], NULL, 10);
    const unsigned int depth = (unsigned int) strtoul(argv[6], NULL, 10);
    if ((strcmp(mode, "serial") != 0 && strcmp(mode, "buffered") != 0) ||
            request_bytes < sizeof(struct s41_aoa_frame_header) ||
            response_bytes < sizeof(struct s41_aoa_frame_header) ||
            request_bytes > MAX_TRANSFER_BYTES ||
            response_bytes > MAX_TRANSFER_BYTES || total_requests == 0 ||
            warmup >= total_requests ||
            depth == 0 || depth > MAX_RING_DEPTH ||
            request_bytes > UINT32_MAX || response_bytes > UINT32_MAX) {
        fprintf(stderr, "invalid arguments\n");
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);
    fprintf(stderr,
            "[aoa-buffer] configured mode=%s request=%zu response=%zu "
            "requests=%llu warmup=%llu depth=%u\n",
            mode, request_bytes, response_bytes,
            (unsigned long long) total_requests,
            (unsigned long long) warmup, depth);
    fflush(stderr);
    const int descriptor = open_accessory();
    if (descriptor < 0) {
        fprintf(stderr, "accessory open timed out: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "[aoa-buffer] endpoint open\n");
    fflush(stderr);

    const int status = strcmp(mode, "serial") == 0
        ? run_serial(descriptor, request_bytes, response_bytes,
                total_requests, warmup)
        : run_buffered(descriptor, request_bytes, response_bytes,
                total_requests, warmup, depth);
    close(descriptor);
    fprintf(stderr, "[aoa-buffer] complete status=%d requests=%llu\n",
            status, (unsigned long long) total_requests);
    return status;
}
