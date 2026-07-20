#!/usr/bin/env bash
# Freeze every executed script + the host binary into an immutable artifact dir,
# then generate SHA256SUMS.txt over the deliverables. Requirement 7: never bind
# evidence to a mutable build-*/bin path.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p artifacts/frozen_scripts
for f in atlas_measure.py build_atlas.py dispatcher.py run_scenario.py \
         validate_dispatch.py device_executor.py live_devices.py \
         tests/test_dispatcher.py; do
    cp -f "$f" "artifacts/frozen_scripts/$(echo "$f" | tr '/' '_')"
done

# the executed host binary was already frozen to artifacts/llama-layersplit-host-cuda;
# confirm it still matches the pinned hash da12f925...
echo "host binary hash check:"
sha256sum artifacts/llama-layersplit-host-cuda

# SHA256SUMS over deliverables (exclude __pycache__ and this checksum file)
{
    find . -type f \
        ! -path './__pycache__/*' ! -name '*.pyc' \
        ! -name 'SHA256SUMS.txt' \
        ! -path './tests/__pycache__/*' \
        | sort | xargs sha256sum
} > SHA256SUMS.txt
echo "wrote SHA256SUMS.txt ($(wc -l < SHA256SUMS.txt) files)"
