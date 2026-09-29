#!/bin/bash
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -w 3600 9
A="/usr/bin/adb -P 5037 -s 3C15AU002CL00000"
mkdir -p /mnt/storage/s43-dual-prep/smoke
$A pull /data/local/tmp/s43-dual-smoke-20260923/worker.log /mnt/storage/s43-dual-prep/smoke/worker.log </dev/null >/dev/null
$A shell "ps -A | grep -c llama-ffn" </dev/null
