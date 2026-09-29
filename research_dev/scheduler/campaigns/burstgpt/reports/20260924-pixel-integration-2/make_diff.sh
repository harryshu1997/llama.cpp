#!/bin/bash
# Usage: make_diff.sh <base_root> <new_root> <out.diff> ; diffs research_dev/scheduler (no reports/__pycache__)
set -e
B=$1; R=$2; OUT=$3
: > $OUT
cd $R
files=$( (cd $B && find research_dev/scheduler -path research_dev/scheduler/campaigns/burstgpt/reports -prune -o -name __pycache__ -prune -o -type f -print); find research_dev/scheduler -path research_dev/scheduler/campaigns/burstgpt/reports -prune -o -name __pycache__ -prune -o -type f -print )
for f in $(echo "$files" | sort -u); do
  if ! cmp -s "$B/$f" "$R/$f" 2>/dev/null; then
    if [ -e "$B/$f" ] && [ -e "$R/$f" ]; then diff -u --label "a/$f" --label "b/$f" "$B/$f" "$R/$f" | sed "1i diff --git a/$f b/$f" >> $OUT || true
    elif [ -e "$R/$f" ]; then (echo "diff --git a/$f b/$f"; echo "new file mode 100644"; diff -u --label /dev/null --label "b/$f" /dev/null "$R/$f") >> $OUT || true
    else (echo "diff --git a/$f b/$f"; echo "deleted file mode 100644"; diff -u --label "a/$f" --label /dev/null "$B/$f" /dev/null) >> $OUT || true; fi
  fi
done
grep -c '^diff --git' $OUT
