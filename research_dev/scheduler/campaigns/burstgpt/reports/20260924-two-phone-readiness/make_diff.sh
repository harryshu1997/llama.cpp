#!/usr/bin/env bash
# Regenerate TWO_PHONE_GAPS.diff from root2/ against the live main tree and check that it applies.
# Refuses when main changed one of the patched files since base2/ was copied (rebase first).
set -u
SP=$(cd "$(dirname "$0")" && pwd)
MAIN=/home/myid/zs89458/Documents/llama.cpp-release
FILES=(
    research_dev/scheduler/_internal/plan_contracts/co_helpers.py
    research_dev/scheduler/_internal/capability_contracts/catalog.py
    research_dev/scheduler/_internal/route_generation/patterns.py
    research_dev/scheduler/_internal/route_generation/envelopes.py
    research_dev/scheduler/_internal/route_generation/costing_parameters.py
    research_dev/scheduler/_internal/adaptive_decode_contracts.py
    research_dev/scheduler/_internal/adaptive_decode_planning.py
    research_dev/scheduler/_unified/common.py
    research_dev/scheduler/_unified/automated_selection_ops/attachment.py
    research_dev/scheduler/_unified/automated_selection_ops/dormant.py
    research_dev/scheduler/adapters/__init__.py
    research_dev/scheduler/adapters/catalog_materialization.py
    research_dev/scheduler/adapters/contracts.py
    research_dev/scheduler/adapters/residency.py
    research_dev/scheduler/adapters/ticket.py
    research_dev/scheduler/adapters/llama_server_contracts.py
    research_dev/scheduler/adapters/llama_server_ops/proofs.py
    research_dev/scheduler/adapters/co_helper_lifecycle.py
    research_dev/scheduler/campaigns/burstgpt/catalog.py
    research_dev/scheduler/campaigns/burstgpt/preflight.py
    research_dev/scheduler/tests/two_phone_harness.py
    research_dev/scheduler/tests/test_two_phone_gaps.py
    research_dev/scheduler/tests/test_two_phone_helpers.py
)
stale=0
for rel in "${FILES[@]}"; do
    if [ -f "$MAIN/$rel" ] && ! cmp -s "$MAIN/$rel" "$SP/base2/$rel"; then
        echo "main changed $rel since base2 was copied: rebase first" >&2
        stale=1
    fi
    if [ ! -f "$MAIN/$rel" ] && [ -f "$SP/base2/$rel" ]; then
        echo "main removed $rel: rebase first" >&2
        stale=1
    fi
done
[ "$stale" = 0 ] || exit 1
{
for rel in "${FILES[@]}"; do
    if [ -f "$MAIN/$rel" ]; then
        diff -u --label "a/$rel" --label "b/$rel" "$MAIN/$rel" "$SP/root2/$rel"
    else
        diff -u --label /dev/null --label "b/$rel" /dev/null "$SP/root2/$rel"
    fi
done
} > "$SP/TWO_PHONE_GAPS.diff"
wc -l "$SP/TWO_PHONE_GAPS.diff"
cd "$MAIN" && git apply --check "$SP/TWO_PHONE_GAPS.diff" < /dev/null && echo "TWO_PHONE_GAPS.diff applies to the live tree"
