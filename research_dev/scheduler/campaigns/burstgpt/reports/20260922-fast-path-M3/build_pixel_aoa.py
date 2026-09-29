"""Build isolated, finite Pixel AOA/TCP probes from preserved implementations."""

from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[5]
PREVIOUS = BASE / "software/pixel10pro-packed-cpu-v1"
OUTPUT = BASE / "software/pixel10pro-aoa-v1"
TRANSPORT = REPO / "research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1"


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one source anchor: {old[:100]!r}")
    return source.replace(old, new)


def worker_source(source):
    source = '#include <poll.h>\n' + source
    source = replace(source, "bool read_exact(int fd, void * data, size_t size) {", r'''
static bool pixel_accessory = false;
static uint8_t accessory_buffer[65536];
static size_t accessory_begin = 0;
static size_t accessory_end = 0;

static bool wait_readable(int fd) {
    pollfd item = {fd, POLLIN, 0};
    int result;
    do { result = poll(&item, 1, 60000); } while (result < 0 && errno == EINTR);
    return result > 0 && (item.revents & POLLIN);
}

bool read_exact(int fd, void * data, size_t size) {
    if (pixel_accessory) {
        auto * target = static_cast<uint8_t *>(data);
        while (size) {
            if (accessory_begin == accessory_end) {
                if (!wait_readable(fd)) return false;
                const ssize_t count = read(fd, accessory_buffer, sizeof(accessory_buffer));
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) return false;
                accessory_begin = 0;
                accessory_end = static_cast<size_t>(count);
            }
            const size_t count = std::min(size, accessory_end - accessory_begin);
            memcpy(target, accessory_buffer + accessory_begin, count);
            accessory_begin += count;
            target += count;
            size -= count;
        }
        return true;
    }''')
    source = replace(source, "        const ssize_t count = read(fd, ptr, size);",
                     "        if (!wait_readable(fd)) return false;\n"
                     "        const ssize_t count = read(fd, ptr, size);")
    source = replace(source, "    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);",
                     '    pixel_accessory = getenv("S42_PIXEL_AOA") && strcmp(getenv("S42_PIXEL_AOA"), "1") == 0;\n'
                     '    int listen_fd = pixel_accessory ? -1 : socket(AF_INET, SOCK_STREAM, 0);')
    source = replace(source, "    if (listen_fd < 0 ||\n", "    if (!pixel_accessory && (listen_fd < 0 ||\n")
    source = replace(source, "        listen(listen_fd, 4) != 0) {", "        listen(listen_fd, 4) != 0)) {")
    source = replace(source, "        const int client_fd = accept(listen_fd, nullptr, nullptr);\n"
                     "        if (client_fd < 0) {\n            continue;\n        }\n"
                     "        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));", r'''
        int client_fd = -1;
        if (pixel_accessory) {
            for (int attempt = 0; attempt < 600 && client_fd < 0; ++attempt) {
                client_fd = open("/dev/usb_accessory", O_RDWR);
                if (client_fd < 0) usleep(100000);
            }
            accessory_begin = accessory_end = 0;
        } else {
            if (!wait_readable(listen_fd)) return 1;
            client_fd = accept(listen_fd, nullptr, nullptr);
        }
        if (client_fd < 0) return 1;
        if (!pixel_accessory) setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));''')
    source = replace(source, '        fprintf(stderr, "[ffn-worker] client disconnected\\n");',
                     '        fprintf(stderr, "[ffn-worker] client disconnected\\n");\n'
                     '        if (pixel_accessory) return 1;')
    return source


def echo_source(source):
    source = replace(source, '#include <errno.h>', '#include <errno.h>\n#include <arpa/inet.h>\n'
                     '#include <netinet/tcp.h>\n#include <poll.h>\n#include <sys/socket.h>')
    source = replace(source, '        const ssize_t count = read(descriptor, cursor, size);',
                     '        struct pollfd item = {descriptor, POLLIN, 0};\n'
                     '        int ready;\n'
                     '        do { ready = poll(&item, 1, 60000); } while (ready < 0 && errno == EINTR);\n'
                     '        if (ready <= 0 || !(item.revents & POLLIN)) return -1;\n'
                     '        const ssize_t count = read(descriptor, cursor, size);')
    source = replace(source, 'static int open_accessory(void) {', r'''
static int open_accessory(void) {
    const char * port = getenv("S42_PIXEL_ECHO_PORT");
    if (port) {
        const int server = socket(AF_INET, SOCK_STREAM, 0);
        const int one = 1;
        struct sockaddr_in address = {0};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons((uint16_t) atoi(port));
        if (server < 0 || setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) ||
                bind(server, (struct sockaddr *) &address, sizeof(address)) || listen(server, 1)) return -1;
        fprintf(stderr, "[aoa-buffer] TCP listening\n");
        fflush(stderr);
        struct pollfd item = {server, POLLIN, 0};
        const int ready = poll(&item, 1, 60000);
        const int client = ready > 0 ? accept(server, NULL, NULL) : -1;
        close(server);
        if (client >= 0) setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        return client;
    }''')
    return source


def main():
    OUTPUT.mkdir(exist_ok=False)
    for header in PREVIOUS.glob("*.h"):
        shutil.copy2(header, OUTPUT / header.name)
    changes = []
    for original, target, transform in (
        (PREVIOUS / "ffn-split-worker.cpp", "ffn-split-worker.cpp", worker_source),
        (TRANSPORT / "aoa_async_transport_v1/aoa_buffered_daemon.c", "aoa_buffered_daemon.c", echo_source),
    ):
        before = original.read_text()
        after = transform(before)
        (OUTPUT / target).write_text(after)
        changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile=str(original), tofile=target))
    shutil.copy2(TRANSPORT / "aoa_async_transport_v1/aoa_async_protocol.h", OUTPUT)
    shutil.copy2(TRANSPORT / "aoa_bench.py", OUTPUT)
    (OUTPUT / "TRANSPORT.patch").write_text("".join(changes))
    command = json.loads((PREVIOUS / "BUILD_COMMAND.json").read_text())
    command = [value.replace(str(PREVIOUS), str(OUTPUT)) for value in command]
    (OUTPUT / "BUILD_COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
    with (OUTPUT / "BUILD.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        clang = Path.home() / "android/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
        subprocess.run([str(clang), "-O3", "-Wall", "-Wextra", "-Werror", "-pthread",
                        str(OUTPUT / "aoa_buffered_daemon.c"), "-o", str(OUTPUT / "aoa-echo")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    hashes = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
              for item in OUTPUT.iterdir() if item.is_file()}
    (OUTPUT / "BUILD_PROVENANCE.json").write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(), "parent": str(PREVIOUS),
        "private_transport_only": True, "hashes": hashes,
    }, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
