#!/usr/bin/env bash
# Regenerate TWO_PHONE.diff (scheduler) and TWO_PHONE_SERVER.diff (C++) from root/ against the live main tree.
set -u
SP=$(cd "$(dirname "$0")" && pwd)
MAIN=/home/myid/zs89458/Documents/llama.cpp-release
emit() {  # emit <relative path>
    local rel=$1
    if [ -f "$MAIN/$rel" ]; then
        diff -u --label "a/$rel" --label "b/$rel" "$MAIN/$rel" "$SP/root/$rel"
    else
        diff -u --label /dev/null --label "b/$rel" /dev/null "$SP/root/$rel"
    fi
    return 0
}
{
for rel in \
    research_dev/scheduler/config.py \
    research_dev/scheduler/configuration/rig.py \
    research_dev/scheduler/configuration/models.py \
    research_dev/scheduler/adapters/phone_transport.py \
    research_dev/scheduler/adapters/llama_server_contracts.py \
    research_dev/scheduler/campaigns/burstgpt/launch.py \
    research_dev/scheduler/campaigns/burstgpt/preflight.py \
    research_dev/scheduler/adapters/phone_helpers.py \
    research_dev/scheduler/adapters/phone_tcp_session.py \
    research_dev/scheduler/campaigns/burstgpt/two_phone_gate.py \
    research_dev/scheduler/tests/test_two_phone_helpers.py \
    research_dev/scheduler/tests/test_two_phone_server_native.py; do
    emit "$rel"
done
} > "$SP/TWO_PHONE.diff"
{
for rel in examples/layersplit/ffn-split-client.h examples/layersplit/ffn-split-client.cpp tools/server/server.cpp; do
    emit "$rel"
done
} > "$SP/TWO_PHONE_SERVER.diff"
wc -l "$SP/TWO_PHONE.diff" "$SP/TWO_PHONE_SERVER.diff"
cd "$MAIN" && git apply --check "$SP/TWO_PHONE.diff" < /dev/null && echo "TWO_PHONE.diff applies" ; git apply --check "$SP/TWO_PHONE_SERVER.diff" < /dev/null && echo "TWO_PHONE_SERVER.diff applies"
