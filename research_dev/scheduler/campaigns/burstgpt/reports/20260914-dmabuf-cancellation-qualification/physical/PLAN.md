# Isolated cancellation qualification

The user authorized a temporary RAM boot of the newly built kernel and a
bounded cancellation prerequisite before real FFN relocation on 2026-09-14.
No partition flash, wipe, desktop reboot, GDM stop, or unrelated process signal.

This directory copies the sibling transport sources solely for a kernel test.
Its active sources, previous qualification, kernel guards, and forced-abort
quarantine are unchanged. The candidate worker and controller accept ONLY the
new image's notes/BTF/config hashes, derived from the verified build.
No prior measured qualification is transferred to this image.

The explicit --cancellation-probe flag permits only four idle 4096-byte receives,
zero transmit requests, no weights and no HTP work. It requires a stop signal
and proves all receive fences are pending before detaching. Exported fences
survive detach on the candidate fence-owned-lock kernel, and every receive must
finish with cancellation status. Buffers remain owned through completion and
acknowledged USB restoration. Other errors retain resources, never force reset.

Tests, in order: one 4096-byte full-payload normal cycle; one cooperative STOP;
one four-receive cancellation. Every test has its own fresh session, shared
device lock, 60-second same-boot check and strict terminal proof. A failure
stops the sequence. Track the eight actual DMA inode IDs and prove their kernel
allocations disappear after the worker exits. Capture kernel diagnostics.
This does not qualify cancellation while HTP compute is active.

The only native build is the isolated USB test worker. Existing HTP libraries
and host transport are reused byte-identically. No FFN worker or shard generator
is rebuilt. Boot uses the existing journalled RAM-boot recorder, with explicit
candidate inputs and separate post-boot binary identity verification.

The boot and each test acquire the existing shared execution lock separately;
each test checks live identity and conflicting work again. No lock is bypassed.
