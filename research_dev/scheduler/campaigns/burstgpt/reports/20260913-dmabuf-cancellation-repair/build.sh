#!/usr/bin/env bash
set -euo pipefail
cd /home/myid/zs89458/Documents/op15-cancel-kernel-20260913-QEC2Bo/source/kernel_platform
test ! -e /home/myid/zs89458/Documents/op15-cancel-kernel-20260913-QEC2Bo/BUILD.log
env -u OPLUS_USE_JFROG_CACHE -u OPLUS_USE_BUILDBUDDY_REMOTE_BUILD \
    nice -n 10 ionice -c 3 tools/bazel build --make_jobs=4 --jobs=2 \
    //common:kernel_aarch64_gki_artifacts \
    2>&1 | tee /home/myid/zs89458/Documents/op15-cancel-kernel-20260913-QEC2Bo/BUILD.log
