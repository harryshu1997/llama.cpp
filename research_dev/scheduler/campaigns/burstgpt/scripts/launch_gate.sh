#!/usr/bin/env bash
# Deploy the local scheduler tree to the desktop and drive one physical gate.
#
# usage: launch_gate.sh <config> <tag> <resolve|calibrate|preflight|offline|run>
#
# Environment overrides:
#   S42_GATE_HOST         ssh target (default zhihao@172.20.74.85)
#   S42_GATE_BASE_DEPLOY  desktop repo copy cloned for a new tag
#                         (default /home/zhihao/s42-online-learning-deploy-20260902-v5)
#
# A per-gate deploy dir /home/zhihao/s42-<tag>-deploy is created from the base
# deploy on first use, research_dev/scheduler is rsynced into it, and launch.py
# writes to /home/zhihao/s42-<tag>-<mode>. See README.md "Physical gates".
set -euo pipefail

config=${1:?config}
tag=${2:?tag}
mode=${3:?mode}
host=${S42_GATE_HOST:-zhihao@172.20.74.85}
base_deploy=${S42_GATE_BASE_DEPLOY:-/home/zhihao/s42-online-learning-deploy-20260902-v5}
local_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
deploy=/home/zhihao/s42-${tag}-deploy
output=/home/zhihao/s42-${tag}-${mode}

test -d "$local_repo/research_dev/scheduler"

case $mode in
    resolve) flag=--resolve-only ;;
    calibrate) flag=--desktop-parent-calibration-only ;;
    preflight) flag=--preflight-only ;;
    offline) flag=--offline-residency-only ;;
    run) flag= ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
esac

ssh -o BatchMode=yes "$host" "test -d $base_deploy && if [ ! -d $deploy ]; then cp -a --reflink=auto $base_deploy $deploy && rm -rf $deploy/inputs; fi"
rsync -rlptc --delete --exclude __pycache__ --exclude '*.pyc' \
    "$local_repo/research_dev/scheduler/" "$host:$deploy/research_dev/scheduler/"
ssh -o BatchMode=yes "$host" "cd $deploy && find research_dev/scheduler -name __pycache__ -prune -exec rm -rf {} + ; \
    sha256sum research_dev/scheduler/scheduler.py research_dev/scheduler/adapters/runtime.py \
        research_dev/scheduler/adapters/phone_session.py research_dev/scheduler/adapters/heterogeneous_rig.py"

ssh -o BatchMode=yes "$host" "test ! -e $output && cd $deploy && \
    S42_UNIFIED_REPO_ROOT=$deploy \
    python3 research_dev/scheduler/campaigns/burstgpt/launch.py \
        research_dev/scheduler/campaigns/burstgpt/configs/$config/campaign.json \
        $output $flag"
