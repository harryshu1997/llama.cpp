#!/bin/bash
# Re-run the two CPU legs with package energy once RAPL is readable on each host:
#   on the desktop and on FCHLLX01:   sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
# then, from FCHLLX01:   bash rerun_cpu_energy.sh server|desktop
# (about 45 minutes per host; nothing else should load the host; results land in $B/<host>-cpu)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
B=/mnt/data_h16t/zs89458/xband-runs/stream-energy-20261110
SH=/mnt/data_h16t/zs89458/xband-data/models/stream-energy-shards
MODELS="qwen25-05b qwen3-06b llama32-1b qwen3-17b gemma4-e2b qwen3-4b qwen3-8b gemma4-12b qwen3-14b qwen3-32b"
items=""; for m in $MODELS; do for n in 2 4 8; do items="$items $m:$n"; done; done
case "${1:-}" in
  server)
    [ -r /sys/class/powercap/intel-rapl:0/energy_uj ] || { echo "RAPL is not readable on $(hostname)"; exit 1; }
    IDLE=10 WARM=3 SECS=20 COOL_C=76000 MIN_COOL_S=30 MAX_COOL_S=300 MAX_LOAD=3 \
      nohup "$HERE/run_host.sh" $B/server-cpu /mnt/data_h16t/zs89458/claude-scratch/stream/ffn-stream-bench.server $SH 64 $items \
      > $B/server-cpu.log 2>&1 &
    echo "server leg started: tail -f $B/server-cpu/campaign.log" ;;
  desktop)
    ssh -o HostKeyAlias=172.20.74.85 zhihao@172.20.125.58 "[ -r /sys/class/powercap/intel-rapl:0/energy_uj ] || { echo 'RAPL is not readable on the desktop'; exit 1; }; COOL_C=35000 bash /mnt/storage/s43-stream-energy-20261110/tools/launch_desktop.sh desktop-cpu $items"
    echo "desktop leg started on the rig: /mnt/storage/s43-stream-energy-20261110/runs/desktop-cpu/campaign.log" ;;
  *) echo "usage: $0 server|desktop"; exit 2 ;;
esac
