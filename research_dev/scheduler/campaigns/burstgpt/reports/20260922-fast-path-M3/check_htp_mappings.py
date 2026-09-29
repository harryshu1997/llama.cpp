"""Run the actual DSP mapping functions with host-side mapping stubs."""

import argparse
from pathlib import Path
import re
import resource
import subprocess
import tempfile


def structure(text, name):
    return re.search(r"struct " + name + r" \{.*?\n\};", text, re.S).group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    htp = args.source_root / "ggml/src/ggml-hexagon/htp"
    source = (htp / "main.c").read_text()
    context = (htp / "htp-ctx.h").read_text()
    ops = (htp / "htp-ops.h").read_text()
    functions = source[source.index("static inline bool reuse_buf("):
                       source.index("static void prep_tensor(")]
    defines = "\n".join(re.findall(r"^#define HTP_(?:OP_MAX_BUFS|MMAP_MAX_VMEM).*", ops, re.M))
    defines += "\n" + re.search(r"^#define HTP_MAX_MMAPS.*", context, re.M).group()
    program = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#undef NULL
#define NULL 0
#define __HVX_ARCH__ 81
#define FARF(...) ((void)0)
#define HAP_PROT_READ 1
#define HAP_PROT_WRITE 2
'''
    program += defines + "\n" + structure(context, "htp_mmap") + "\n" + structure(ops, "htp_buf_desc")
    program += r'''
struct htp_context { struct htp_mmap mmap[HTP_MAX_MMAPS]; uint64_t max_vmem; };
static unsigned maps, unmaps;
static void * HAP_mmap2(void * base, uint64_t size, int prot, int flags, uint32_t fd, int off) {
    ++maps;
    return (void *)(uintptr_t)(0x100000 + fd * 4096);
}
static void HAP_munmap2(void * base, uint64_t size) { ++unmaps; }
'''
    program += functions
    program += r'''
int main(void) {
    struct htp_context ctx = { .max_vmem = UINT64_C(1) << 32 };
    struct htp_buf_desc bufs[65] = {0};
    for (unsigned i = 0; i < 64; ++i) { bufs[i].fd = i + 1; bufs[i].size = 4096; }
    prep_op_bufs(&ctx, bufs, 64);
    for (unsigned i = 0; i < 64; ++i) { assert(bufs[i].base == 0x100000 + (i + 1) * 4096); }
    assert(maps == 64 && unmaps == 0);
    prep_op_bufs(&ctx, bufs, 64);
    assert(maps == 64 && unmaps == 0);

    // Capacity pressure must evict only the old fd, including when slot 63 is reused.
    bufs[0].fd = 1000;
    prep_op_bufs(&ctx, bufs, 64);
    assert(maps == 65 && unmaps == 1);
    assert(bufs[0].base == 0x100000 + 1000 * 4096);
    assert(bufs[63].base == 0x100000 + 64 * 4096);

    // Byte pressure independently evicts unused mappings.
    ctx.max_vmem = 64 * 4096;
    bufs[0].fd = 2000;
    bufs[0].size = 8192;
    prep_op_bufs(&ctx, bufs, 1);
    assert(maps == 66 && unmaps == 65);
    assert(bufs[0].base == 0x100000 + 2000 * 4096);

    // Malformed oversized batches and exhausted slots fail before tensor execution.
    for (unsigned mode = 0; mode < 2; ++mode) {
        pid_t pid = fork();
        assert(pid >= 0);
        if (!pid) {
            if (mode == 0) { prep_op_bufs(&ctx, bufs, 65); }
            else {
                for (unsigned i = 0; i < HTP_MAX_MMAPS; ++i) { ctx.mmap[i].size = 1; }
                bufs[0].base = 0;
                mmap_buf(&ctx, &bufs[0]);
            }
            _exit(0);
        }
        int status;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFSIGNALED(status) && WTERMSIG(status) == 6);
    }
    puts("PASS: 64 mappings, full reuse, bit 63, slot and byte eviction, fail-fast guards");
}
'''
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with tempfile.TemporaryDirectory(prefix="s42-htp-mappings-") as directory:
        path = Path(directory)
        (path / "check.c").write_text(program)
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
                        "-fsanitize=undefined", "-o", str(path / "check"), str(path / "check.c")], check=True)
        subprocess.run([str(path / "check")], check=True)


if __name__ == "__main__":
    main()
