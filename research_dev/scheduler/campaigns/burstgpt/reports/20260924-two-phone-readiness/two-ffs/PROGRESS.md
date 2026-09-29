# PROGRESS: two FunctionFS DMA-BUF USB helpers, host side (2026-09-24)

Scratch: this directory. `base/` = copy of the main working tree (research_dev/scheduler minus reports/ and
baselines/cuda_graph_v1, examples/layersplit, tools/server) at HEAD 5f89a2d9d + uncommitted; `root/` = edited copy.
Main tree is never edited from here. No adb, no rig run, no commit.

## Checkpoint 1 (setup + reading)
- Read AGENTS.md, two-phone readiness README + GAPS_README.
- Copied base/root (15M each, diff -rq empty).
- C++ read: ffn-split-usb-client.{cpp,h}: one libusb_context per usb_client (libusb_init(&impl_->context) in
  connect()), no static state; the only shared-state hazard is libusb_open_device_with_vid_pid (first
  18d1:2d00 match). server.cpp refuses >1 functionfs-usb helper ("at most one server FFN helper can use the
  functionfs-usb transport") for exactly that reason.
- USB client is driven synchronously from the eval-callback thread of its owning ffn_split::client
  (start_usb_exchange / publish_usb_output), not from the client's worker thread; each client has its own
  usb_ pointer and slots.

## Checkpoint 2 (design fixed, 2026-09-24)
Facts that drive the design:
- OP15's FunctionFS gadget exposes iSerialNumber `SCHEDFFN0001` (direct_phone_ffn_session.sh writes it), NOT the
  ADB serial 3C15AU002CL00000. Host sysfs `serial` of the 18d1:2d00 device therefore reads the GADGET serial.
  => a second FunctionFS phone MUST expose a distinct gadget serial; the port path (2-2 / 2-9.2) is stable
  across the android<->functionfs switch and is the durable pin.
- Host-side first-match hazards: C++ `libusb_open_device_with_vid_pid` (client), `ffn-split-usb-close.cpp`
  (same VID:PID open), Python `bridge.probe_functionfs_usb_device` (requires exactly ONE 18d1:2d00 in sysfs ->
  with two gadgets live OP15's own launch verification would fail "not unique").
- Two usb_client instances already have separate libusb contexts/handles/slots; the USB client is driven
  synchronously from the eval-callback thread of its own ffn_split::client; layers run in order so the two
  clients never overlap. Shared kernel resource: usbfs memory pool (usbfs_memory_mb) -> per-helper
  usbfs_available_bytes must be explicit and the sum budgeted (new preflight row).
- Scheduler identity is single-phone (TransportQualificationIdentity keyed by one serial/sysfs); the helper
  identity (PhoneHelperTransportIdentity) already exists per device; extend its functionfs-usb requirements
  with the gadget serial + gadget-switch/scheduler-session receipts and add a serial-keyed lookup.

Decisions:
C++  - usb_client_config += serial_number, port_path (both optional). Empty => legacy loop byte-identical.
       Non-empty => enumerate (libusb_get_device_list/descriptor/bus+port numbers/open/string descriptor),
       filter VID:PID [+port path], match serial; 0 matches => retry within the legacy 30 s budget then
       clear error listing candidates; >1 => immediate error; required serial but iSerialNumber==0 =>
       immediate error. usb_client::selected_device() reports bus-port path + serial.
     - client_config += usb_serial, usb_port_path; server env <prefix>USB_SERIAL / <prefix>USB_PORT_PATH;
       >1 functionfs helpers allowed iff every one has USB_SERIAL, all distinct (port paths distinct if given).
       S41SERVERFFNHELPER line gains usb_serial=/usb_port_path= only for selected helpers (multi-helper only).
     - ffn-split-usb-close gains optional argv[11] serial and argv[12] port path (10-arg form unchanged).
Py   - rig helper transport kinds: adb-tcp | functionfs-dmabuf (+ gadget paths, ffs root, UDC, restore/session
       scripts, session_root, gadget_serial, functionfs_identity, transport_generation, functionfs_transport).
     - phone_transport: functionfs contract += usb_serial/usb_port_path (env only when set).
     - phone_helpers: multi-functionfs rule mirrors C++; identity requirements per transport; serial-keyed
       lookup + loader; per-device preflight rows (gadget state, identity present/qualified, usbfs budget).
     - bridge.probe_functionfs_usb_device(+sysfs_device=None, serial=None) filters; OP15 launch pins to its
       identity's sysfs device when an identity is configured (narrowing only).
     - new adapters/phone_ffs_session.py: FunctionFsPhoneWorkerSession (preflight/start/stop) driving a
       per-phone session script with per-device gadget lock; FunctionFsStopPolicy for CoHelperLifecycle.
     - co_helpers plan contract: RuntimeCoHelperPhone accepts functionfs-usb transport parameters.
     - launch/preflight: --helper-phone-identity device=path; identity row PASS only when qualified; the
       two-phone-dispatch row and launch.py refusal stay BLOCKED/refused.
