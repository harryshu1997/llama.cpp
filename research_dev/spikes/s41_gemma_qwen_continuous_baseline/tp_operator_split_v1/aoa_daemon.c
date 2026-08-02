// Phone-side AOA accessory daemon (must run as root: /dev/usb_accessory is
// root:usb 0660 and the shell user is not in the usb group).
//
// Serves the same request/response shape as act_echo so the AOA transport can
// be compared like-for-like against the adb-forward path: read REQ bytes from
// the accessory endpoint, write RSP bytes back. No compute.
//
// usage: aoa_daemon <req_bytes> <rsp_bytes>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <req> <rsp>\n", argv[0]); return 2; }
    const int req = atoi(argv[1]);
    const int rsp = atoi(argv[2]);
    signal(SIGPIPE, SIG_IGN);

    char * rbuf = malloc(req);
    char * sbuf = calloc(1, rsp);

    for (;;) {
        int fd = open("/dev/usb_accessory", O_RDWR);
        if (fd < 0) {
            fprintf(stderr, "[aoa] open failed: %s\n", strerror(errno));
            sleep(1);
            continue;
        }
        fprintf(stderr, "[aoa] accessory endpoint open, serving %d->%d\n", req, rsp);
        fflush(stderr);
        for (;;) {
            int got = 0;
            while (got < req) {
                ssize_t r = read(fd, rbuf + got, req - got);
                if (r <= 0) { fprintf(stderr, "[aoa] read: %s\n", strerror(errno)); goto reopen; }
                got += r;
            }
            int put = 0;
            while (put < rsp) {
                ssize_t w = write(fd, sbuf + put, rsp - put);
                if (w <= 0) { fprintf(stderr, "[aoa] write: %s\n", strerror(errno)); goto reopen; }
                put += w;
            }
        }
    reopen:
        close(fd);
        fflush(stderr);
    }
}
