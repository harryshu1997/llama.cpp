// Activation-transfer probe daemon.
//
// Models the real operator-split exchange rather than a symmetric echo: read
// exactly REQ bytes (the activation the server sends), reply with exactly RSP
// bytes (the partial result the phone returns). No compute - this isolates the
// transport cost of one activation round trip.
//
// usage: act_echo <port> <req_bytes> <rsp_bytes>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <port> <req> <rsp>\n", argv[0]); return 2; }
    const int port = atoi(argv[1]);
    const int req  = atoi(argv[2]);
    const int rsp  = atoi(argv[3]);
    signal(SIGPIPE, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);

    char * rbuf = malloc(req);
    char * sbuf = calloc(1, rsp);

    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(port);
    if (bind(ls, (struct sockaddr *)&a, sizeof a) || listen(ls, 8)) { perror("bind"); return 1; }
    if (fork() > 0) return 0;
    setsid();

    for (;;) {
        int cs = accept(ls, NULL, NULL);
        if (cs < 0) continue;
        if (fork() > 0) { close(cs); continue; }
        close(ls);
        int sz = 1 << 20;
        setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        setsockopt(cs, SOL_SOCKET, SO_RCVBUF, &sz, sizeof sz);
        setsockopt(cs, SOL_SOCKET, SO_SNDBUF, &sz, sizeof sz);
        for (;;) {
            int got = 0;
            while (got < req) {
                ssize_t r = read(cs, rbuf + got, req - got);
                if (r <= 0) goto done;
                got += r;
            }
            int put = 0;
            while (put < rsp) {
                ssize_t w = write(cs, sbuf + put, rsp - put);
                if (w <= 0) goto done;
                put += w;
            }
        }
    done:
        close(cs);
        _exit(0);
    }
}
