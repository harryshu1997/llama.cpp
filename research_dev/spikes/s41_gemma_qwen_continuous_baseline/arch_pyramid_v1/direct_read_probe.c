#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return value.tv_sec + value.tv_nsec * 1.0e-9;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FILE\n", argv[0]);
        return 2;
    }
    const size_t block_size = 4ULL * 1024 * 1024;
    const size_t target = 1024ULL * 1024 * 1024;
    void * buffer = NULL;
    if (posix_memalign(&buffer, 4096, block_size) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }
    const int fd = open(argv[1], O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) {
        fprintf(stderr, "open: %s\n", strerror(errno));
        free(buffer);
        return 1;
    }

    size_t completed = 0;
    const double start = seconds();
    while (completed < target) {
        const ssize_t count = pread(fd, buffer, block_size, (off_t) completed);
        if (count <= 0) {
            fprintf(stderr, "pread at %zu: %s\n", completed,
                    count == 0 ? "end of file" : strerror(errno));
            close(fd);
            free(buffer);
            return 1;
        }
        completed += (size_t) count;
    }
    const double elapsed = seconds() - start;
    printf("bytes=%zu elapsed_s=%.6f read_GBps=%.3f\n",
           completed, elapsed, completed / elapsed / 1.0e9);
    close(fd);
    free(buffer);
    return 0;
}
