#define _GNU_SOURCE

// Persistent NCM request/response transport probe.
//
// usage: ncm_echo_server <port> [interface link_local_address]

#include <arpa/inet.h>
#include <errno.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_PAYLOAD_BYTES (16U * 1024U * 1024U)
#define STOP_LENGTH UINT32_MAX

static int read_exact(int fd, void * destination, size_t size) {
    uint8_t * cursor = destination;
    while (size > 0) {
        const ssize_t count = read(fd, cursor, size);
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

static int write_exact(int fd, const void * source, size_t size) {
    const uint8_t * cursor = source;
    while (size > 0) {
        const ssize_t count = send(fd, cursor, size, MSG_NOSIGNAL);
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
    if (argc != 2 && argc != 4) {
        fprintf(stderr,
                "usage: %s <port> [interface link_local_address]\n",
                argv[0]);
        return 2;
    }
    const int port = atoi(argv[1]);
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "invalid port\n");
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);
    uint8_t * request = malloc(MAX_PAYLOAD_BYTES);
    uint8_t * response = calloc(1, MAX_PAYLOAD_BYTES + sizeof(uint32_t));
    if (request == NULL || response == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    const int listener = socket(AF_INET6, SOCK_STREAM, 0);
    if (listener < 0) {
        perror("socket");
        return 1;
    }
    const int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    setsockopt(listener, IPPROTO_IPV6, IPV6_V6ONLY, &one, sizeof(one));
    if (argc == 4 && setsockopt(listener, SOL_SOCKET, SO_BINDTODEVICE,
                argv[2], strlen(argv[2]) + 1) != 0) {
        perror("SO_BINDTODEVICE");
        return 1;
    }

    struct sockaddr_in6 address;
    memset(&address, 0, sizeof(address));
    address.sin6_family = AF_INET6;
    if (argc == 4) {
        if (inet_pton(AF_INET6, argv[3], &address.sin6_addr) != 1) {
            fprintf(stderr, "invalid IPv6 address\n");
            return 2;
        }
        address.sin6_scope_id = if_nametoindex(argv[2]);
        if (address.sin6_scope_id == 0) {
            fprintf(stderr, "invalid interface\n");
            return 2;
        }
    } else {
        address.sin6_addr = in6addr_any;
    }
    address.sin6_port = htons((uint16_t) port);
    if (bind(listener, (struct sockaddr *) &address, sizeof(address)) != 0 ||
        listen(listener, 4) != 0) {
        perror("bind/listen");
        return 1;
    }

    fprintf(stderr,
            "[ncm-echo] ready port=%d interface=%s max_payload=%u\n",
            port, argc == 4 ? argv[2] : "any", MAX_PAYLOAD_BYTES);
    fflush(stderr);

    uint64_t completed = 0;
    for (;;) {
        const int client = accept(listener, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("accept");
            return 1;
        }
        const int socket_buffer = 4 * 1024 * 1024;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        setsockopt(client, SOL_SOCKET, SO_RCVBUF,
                &socket_buffer, sizeof(socket_buffer));
        setsockopt(client, SOL_SOCKET, SO_SNDBUF,
                &socket_buffer, sizeof(socket_buffer));
        fprintf(stderr, "[ncm-echo] accepted\n");
        fflush(stderr);

        for (;;) {
            uint32_t header[2];
            if (read_exact(client, header, sizeof(header)) != 0) {
                break;
            }
            const uint32_t request_bytes = ntohl(header[0]);
            const uint32_t response_bytes = ntohl(header[1]);
            if (request_bytes == STOP_LENGTH && response_bytes == STOP_LENGTH) {
                fprintf(stderr, "[ncm-echo] complete requests=%llu\n",
                        (unsigned long long) completed);
                fflush(stderr);
                close(client);
                close(listener);
                free(response);
                free(request);
                return 0;
            }
            if (request_bytes > MAX_PAYLOAD_BYTES ||
                response_bytes > MAX_PAYLOAD_BYTES) {
                fprintf(stderr, "[ncm-echo] invalid lengths %u %u\n",
                        request_bytes, response_bytes);
                return 3;
            }
            if (read_exact(client, request, request_bytes) != 0) {
                break;
            }
            const uint32_t encoded_response = htonl(response_bytes);
            memcpy(response, &encoded_response, sizeof(encoded_response));
            if (write_exact(client, response,
                    sizeof(encoded_response) + response_bytes) != 0) {
                break;
            }
            ++completed;
        }
        close(client);
    }
}
