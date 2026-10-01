#!/usr/bin/env bash
# Build the Pixel AOA relay (WS10).
#
#   build_aoa_relay.sh android OUT_DIR   # arm64 Android (NDK r28b clang, API 28), the Pixel worker's toolchain image
#   build_aoa_relay.sh host OUT_DIR      # host build for the loopback tests (cc)
#
# Writes OUT_DIR/s43-aoa-relay (android) or OUT_DIR/s43-aoa-relay-host, plus BUILD_<target>.json with the
# exact command, the source and header sha256 and the binary sha256 (pin the binary in the helper evidence).
set -euo pipefail
target=${1:?android|host}
out=$(mkdir -p "${2:?OUT_DIR}" && cd "$2" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
flags=(-std=c11 -O2 -Wall -Wextra -Werror -pthread)
case "$target" in
  android)
    image=${S43_NDK_IMAGE:-snapdragon-toolchain-hostgcc:v0.3}
    cc=/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang
    binary=s43-aoa-relay
    command=(docker run --rm --user "$(id -u):$(id -g)" -v "$here:/src:ro" -v "$out:/out" "$image"
             "$cc" "${flags[@]}" -o "/out/$binary" /src/s43_aoa_relay.c)
    ;;
  host)
    binary=s43-aoa-relay-host
    command=("${CC:-cc}" "${flags[@]}" -o "$out/$binary" "$here/s43_aoa_relay.c")
    ;;
  *) echo "target must be android or host" >&2; exit 2 ;;
esac
"${command[@]}"
python3 - "$out" "$binary" "$target" "$here" "${command[@]}" <<'EOF'
import hashlib, json, sys
from pathlib import Path
out, binary, target, here, *command = sys.argv[1:]
digest = lambda path: "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
record = {"target": target, "binary": binary, "binary_sha256": digest(Path(out) / binary),
          "sources": {name: digest(Path(here) / name) for name in ("s43_aoa_relay.c", "s43a_protocol.h")},
          "command": command}
(Path(out) / f"BUILD_{target}.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
EOF
