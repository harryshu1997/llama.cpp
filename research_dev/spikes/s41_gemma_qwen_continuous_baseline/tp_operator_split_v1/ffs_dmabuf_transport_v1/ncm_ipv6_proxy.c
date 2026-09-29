#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <net/if.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int parse_port(const char * text) {
    char * end = NULL;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value <= 0 || value > 65535) {
        return -1;
    }
    return (int) value;
}

static int send_all(int fd, const uint8_t * data, size_t size) {
    while (size > 0) {
        const ssize_t count = send(fd, data, size, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        data += count;
        size -= (size_t) count;
    }
    return 0;
}

static int relay(int client, int target) {
    struct pollfd fds[2] = {
        { .fd = client, .events = POLLIN },
        { .fd = target, .events = POLLIN },
    };
    uint8_t buffer[64 * 1024];
    uint64_t uploaded = 0;
    uint64_t downloaded = 0;
    int active = 2;

    while (active > 0) {
        const int status = poll(fds, 2, -1);
        if (status < 0 && errno == EINTR) {
            continue;
        }
        if (status < 0) {
            perror("poll");
            return -1;
        }
        for (int i = 0; i < 2; ++i) {
            if ((fds[i].revents & (POLLIN | POLLHUP)) == 0) {
                continue;
            }
            ssize_t count = recv(fds[i].fd, buffer, sizeof(buffer), 0);
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count < 0 &&
                (errno == ECONNRESET || errno == ECONNABORTED)) {
                count = 0;
            }
            if (count < 0) {
                perror("recv");
                return -1;
            }
            if (count == 0) {
                fds[i].events = 0;
                shutdown(fds[1 - i].fd, SHUT_WR);
                --active;
                continue;
            }
            if (send_all(fds[1 - i].fd, buffer, (size_t) count) != 0) {
                perror("send");
                return -1;
            }
            if (i == 0) {
                uploaded += (uint64_t) count;
            } else {
                downloaded += (uint64_t) count;
            }
        }
    }
    fprintf(stderr,
            "[ncm-phone-proxy] complete upload_bytes=%llu download_bytes=%llu\n",
            (unsigned long long) uploaded,
            (unsigned long long) downloaded);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr,
                "usage: %s <listen-port> <interface> <listen-address> <target-port>\n",
                argv[0]);
        return 2;
    }
    const int listen_port = parse_port(argv[1]);
    const int target_port = parse_port(argv[4]);
    const unsigned int scope_id = if_nametoindex(argv[2]);
    if (listen_port < 0 || target_port < 0 || scope_id == 0) {
        fprintf(stderr, "invalid port or interface\n");
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);
    const int one = 1;
    const int listener = socket(AF_INET6, SOCK_STREAM, 0);
    if (listener < 0 ||
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0 ||
        setsockopt(listener, IPPROTO_IPV6, IPV6_V6ONLY, &one, sizeof(one)) != 0 ||
        setsockopt(listener, SOL_SOCKET, SO_BINDTODEVICE,
                   argv[2], strlen(argv[2]) + 1) != 0) {
        perror("listener setup");
        return 1;
    }

    struct sockaddr_in6 listen_address = { 0 };
    listen_address.sin6_family = AF_INET6;
    listen_address.sin6_port = htons((uint16_t) listen_port);
    listen_address.sin6_scope_id = scope_id;
    if (inet_pton(AF_INET6, argv[3], &listen_address.sin6_addr) != 1 ||
        bind(listener, (struct sockaddr *) &listen_address,
             sizeof(listen_address)) != 0 ||
        listen(listener, 1) != 0) {
        perror("bind/listen");
        return 1;
    }
    fprintf(stderr,
            "[ncm-phone-proxy] ready [%s%%%s]:%d -> 127.0.0.1:%d\n",
            argv[3], argv[2], listen_port, target_port);
    fflush(stderr);

    const int client = accept(listener, NULL, NULL);
    if (client < 0) {
        perror("accept");
        return 1;
    }
    close(listener);

    const int target = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in target_address = { 0 };
    target_address.sin_family = AF_INET;
    target_address.sin_port = htons((uint16_t) target_port);
    target_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (target < 0 ||
        connect(target, (struct sockaddr *) &target_address,
                sizeof(target_address)) != 0) {
        perror("target connect");
        close(client);
        return 1;
    }
    fprintf(stderr, "[ncm-phone-proxy] connected\n");
    fflush(stderr);

    const int status = relay(client, target);
    close(target);
    close(client);
    return status == 0 ? 0 : 1;
}
