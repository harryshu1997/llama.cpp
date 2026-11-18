#!/bin/bash
# Accounting for the g14 pair against the g8 pair (4x trace): fleet energy, latency, service window, per-request queue table.
set -u
S=/mnt/data_h16t/zs89458/xband-runs/second-desktop
T=/home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/tools
A=$S/analysis-g16ab; mkdir -p $A
cd /home/myid/zs89458/Documents/llama.cpp-release
RUNS="--run g62nph=$S/inputs-desktop-g62nph --run g62new=$S/inputs-derived-mpaoa-g62new --run g161nph=$S/inputs-desktop-g161nph --run g16r1new=$S/inputs-derived-mpaoa-g16r1new --run g16c1new=$S/inputs-derived-mpaoa-g16c1new --run g16o1new=$S/inputs-derived-mpaoa-g16o1new"
METERS="--meter g62nph=$S/g62nph-desktop/METER.json --meter g62new=$S/g62new-derived-mpaoa/METER.json --meter g161nph=$S/g161nph-desktop/METER.json --meter g16r1new=$S/g16r1new-derived-mpaoa/METER.json --meter g16c1new=$S/g16c1new-derived-mpaoa/METER.json --meter g16o1new=$S/g16o1new-derived-mpaoa/METER.json"
python3 $T/fleet_energy.py $RUNS --baseline g161nph $METERS --json $A/fleet-g16ab.json --md $A/fleet-g16ab.md > $A/fleet-g16ab.log 2>&1; echo "fleet exit=$?"
python3 $T/latency_report.py $RUNS --json $A/latency-g16ab.json --md $A/latency-g16ab.md > $A/latency-g16ab.log 2>&1; echo "latency exit=$?"
python3 $T/service_energy.py $RUNS $METERS --baseline g161nph --json $A/service-g16ab.json --md $A/service-g16ab.md > $A/service-g16ab.log 2>&1; echo "service exit=$?"
python3 /mnt/data_h16t/zs89458/claude-scratch/lane/compare_qwen_queue.py g62nph=$S/inputs-desktop-g62nph g62new=$S/inputs-derived-mpaoa-g62new g161nph=$S/inputs-desktop-g161nph g16r1new=$S/inputs-derived-mpaoa-g16r1new g16c1new=$S/inputs-derived-mpaoa-g16c1new g16o1new=$S/inputs-derived-mpaoa-g16o1new > $A/queue-g16ab.md 2>&1; echo "queue exit=$?"
cp /mnt/data_h16t/zs89458/claude-scratch/lane/compare_qwen_queue.py $A/
