#!/bin/bash
cd /mnt/storage/s42-trace-v2-20260921-prep/source
export LANG=C.UTF-8 S42_UNIFIED_REPO_ROOT=/mnt/storage/s42-trace-v2-20260921-prep/source LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64
echo "preflight start $(date -u +%H:%M:%SZ)" > /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/CHAIN.log
python3 research_dev/scheduler/campaigns/burstgpt/launch.py /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/campaign.json /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/preflight-1 --preflight-only > /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/PREFLIGHT_1.log 2>&1
grep -q "\"status\": \"PASS\"" /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/PREFLIGHT_1.log && echo PREFLIGHT_PASS >> /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/CHAIN.log || { echo PREFLIGHT_NOT_PASS >> /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/CHAIN.log; exit 1; }
flock -w 900 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock python3 research_dev/scheduler/campaigns/burstgpt/launch.py /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/campaign.json /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/run-treatment-1 > /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/RUN_1.log 2>&1
echo "RUN_EXIT=$? $(date -u +%H:%M:%SZ)" >> /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs/CHAIN.log
