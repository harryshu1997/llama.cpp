# Isolated OP15 cancellation-kernel candidate

Requested work: repair DMA-BUF cancellation safety before scheduler phone gates.
This directory is an independent copy of the existing vendor build inputs.
The original source tree, distribution, boot images, crash records, and
transport candidates are not modified.

Base: kernel common 227664cbe007bbad49aa74259179ac99608a2113, including the
previously applied DMA direction, request serialization, and fence-reference
fixes. Base f_fs.c SHA-256:
229d79e869516834161cd8e4e5cc6b619ecd0e8930c096acc33fde5aa2cf4f63.

Reuse only the existing additional candidate from
moe-resident-routing-a6000/artifacts/op15_kernel_fix_20260913/fence_lifetime.patch:
move the fence lock into the fence and free raw allocations on pre-init errors.
No kernel configuration, module policy, or USB wire-format changes are intended.

Compile with bounded parallelism (four make jobs, two Bazel jobs), reduced CPU
and I/O priority, and isolated outputs. Preserve build errors and input hashes.
No automatic boot or flash. A compiled candidate still requires configuration,
image and ABI review before a separately coordinated temporary boot and test.
The current forced-cancellation quarantine remains in place.
