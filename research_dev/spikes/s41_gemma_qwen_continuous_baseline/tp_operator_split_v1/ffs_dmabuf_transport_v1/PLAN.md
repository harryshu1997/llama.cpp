# FunctionFS DMA-BUF transport spike

This bounded S41 experiment compares four stages of the same exact USB
request/response protocol on the real OP15 and RTX 4060 Ti host:

1. FunctionFS read/write with ordinary host memory.
2. FunctionFS DMA-BUF endpoints with ordinary host memory.
3. FunctionFS DMA-BUF endpoints with libusb persistent DMA memory.
4. The same persistent buffers with multiple independent requests in flight.

The phone uses the stock FunctionFS DMA-BUF UAPI and the Qualcomm system DMA
heap. The host uses `libusb_dev_mem_alloc`. Every response is checked for its
sequence, sizes, and sentinels. A temporary second configfs gadget is bounded
by a watchdog and the original `ptp,adb` gadget is restored after each run.

Completed gates:

1. The ordinary FunctionFS control passes on real hardware.
2. The stock FunctionFS DMA-BUF attempt failed with a captured DWC3 SMMU write
   fault and kernel panic.
3. The exact upstream direction, request-lifetime, and fence-reference fixes
   were backported into a temporary, non-flashed kernel.
4. Copy, phone DMA-BUF, host persistent memory, and queued variants passed
   14,400 paid requests with exact protocol validation.
5. USB wrote directly into an HTP `rpcmem` DMA-BUF, HTP executed on it, and USB
   returned the HTP output DMA-BUF for 5,400 exact paid requests.
6. The phone was rebooted to its stock partition and its original SuperSpeed
   `ptp,adb` gadget was verified.

The next bounded gate is one real model-shaped operator on these same buffers,
then a complete-layer CUDA overlap test. Full-model, BurstGPT, and energy work
remain outside this transport result.

See `RESULTS.md` for measurements and evidence hashes.
