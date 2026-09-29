#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <net/if.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_PAYLOAD_BYTES 60000U
#define STOP_LENGTH UINT32_MAX

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s <port> <interface> <link_local_address>\n",
                argv[0]);
        return 2;
    }
    const int port = atoi(argv[1]);
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "invalid port\n");
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);
    uint8_t * request = malloc(MAX_PAYLOAD_BYTES + 3 * sizeof(uint32_t));
    uint8_t * response = calloc(1, MAX_PAYLOAD_BYTES + 2 * sizeof(uint32_t));
    if (request == NULL || response == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    const int descriptor = socket(AF_INET6, SOCK_DGRAM, 0);
    if (descriptor < 0) {
        perror("socket");
        return 1;
    }
    const int socket_buffer = 4 * 1024 * 1024;
    setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF,
            &socket_buffer, sizeof(socket_buffer));
    setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF,
            &socket_buffer, sizeof(socket_buffer));
    if (setsockopt(descriptor, SOL_SOCKET, SO_BINDTODEVICE,
                argv[2], strlen(argv[2]) + 1) != 0) {
        perror("SO_BINDTODEVICE");
        return 1;
    }

    struct sockaddr_in6 address;
    memset(&address, 0, sizeof(address));
    address.sin6_family = AF_INET6;
    address.sin6_port = htons((uint16_t) port);
    address.sin6_scope_id = if_nametoindex(argv[2]);
    if (address.sin6_scope_id == 0 ||
            inet_pton(AF_INET6, argv[3], &address.sin6_addr) != 1) {
        fprintf(stderr, "invalid interface or IPv6 address\n");
        return 2;
    }
    if (bind(descriptor, (struct sockaddr *) &address, sizeof(address)) != 0) {
        perror("bind");
        return 1;
    }

    fprintf(stderr, "[ncm-udp-echo] ready port=%d interface=%s\n",
            port, argv[2]);
    fflush(stderr);

    uint64_t completed = 0;
    for (;;) {
        struct sockaddr_in6 peer;
        socklen_t peer_size = sizeof(peer);
        const ssize_t count = recvfrom(descriptor, request,
                MAX_PAYLOAD_BYTES + 3 * sizeof(uint32_t), 0,
                (struct sockaddr *) &peer, &peer_size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < (ssize_t) (3 * sizeof(uint32_t))) {
            continue;
        }

        uint32_t header[3];
        memcpy(header, request, sizeof(header));
        const uint32_t request_bytes = ntohl(header[0]);
        const uint32_t response_bytes = ntohl(header[1]);
        const uint32_t sequence = ntohl(header[2]);
        if (request_bytes == STOP_LENGTH && response_bytes == STOP_LENGTH) {
            break;
        }
        if (request_bytes > MAX_PAYLOAD_BYTES ||
                response_bytes > MAX_PAYLOAD_BYTES ||
                count != (ssize_t) (sizeof(header) + request_bytes)) {
            continue;
        }

        const uint32_t encoded_response = htonl(response_bytes);
        const uint32_t encoded_sequence = htonl(sequence);
        memcpy(response, &encoded_response, sizeof(encoded_response));
        memcpy(response + sizeof(encoded_response), &encoded_sequence,
                sizeof(encoded_sequence));
        if (sendto(descriptor, response,
                    2 * sizeof(uint32_t) + response_bytes, 0,
                    (struct sockaddr *) &peer, peer_size) < 0) {
            perror("sendto");
            return 1;
        }
        ++completed;
    }

    fprintf(stderr, "[ncm-udp-echo] complete requests=%llu\n",
            (unsigned long long) completed);
    close(descriptor);
    free(response);
    free(request);
    return 0;
}
