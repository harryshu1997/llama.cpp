#!/bin/bash
# block3_pilot_generality.sh  (desktop, setsid nohup) - waits for block 2 (QUEUE-p2m "queue done"), then:
#  1. quality pilot (protocol ws4-quality-gsm8k-ni-v1 section 3: GSM8K TRAIN split, legacy arm only, 512-token budget rule;
#     pilot questions are never reused by the confirmatory suite), scored with `quality.score pilot`;
#  2. generality: tuned dispatcher-only vs full system (paper_config_v1) on windows w2-w4 and densities d2x/d4x/d0p5x
#     (prefix p3g, 1 repeat, shuffled, both phones metered).
# A failed pilot is logged and does not block step 2. Stops if block 2 aborted.
set -u
R=/mnt/storage/s43-two-phone-eval-20260925
D=/mnt/storage/s42-trace-v2-20260921-prep/source
Q=/mnt/storage/quality-gsm8k-v1
LOCK=/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
H=$R/rig; LOG=$R/chains/QUEUE-p3.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
log "block 3 waiting for block 2"
for i in $(seq 1 5760); do
  grep -q "queue done" $R/chains/QUEUE-p2m.log 2>/dev/null && break
  if grep -q "ABORT" $R/chains/QUEUE-p2m.log 2>/dev/null || tail -n +172 $R/chains/QUEUE-p0m.log | grep -q "ABORT"; then log "block 2 aborted: block 3 not started"; exit 3; fi
  sleep 30
done
grep -q "queue done" $R/chains/QUEUE-p2m.log || { log "block 2 not done after 48 h: block 3 not started"; exit 4; }
pilot() {
  CODEC=/home/zhihao/s41-dynamic-ffn-v1/host/llama-token-codec; LIB=$(dirname $CODEC)
  QTOK=/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf; GTOK=/home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf
  LTOK=/home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf
  for pair in "$QTOK 500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0" "$GTOK 494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c" "$LTOK 4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad"; do
    set -- $pair; got=$(sha256sum "$1" | cut -d' ' -f1); [ "$got" = "$2" ] || { log "pilot: tokenizer sha mismatch $1 ($got)"; return 1; }
  done
  (cd $Q/data && sha256sum -c --quiet SHA256SUMS) || { log "pilot: dataset sha mismatch"; return 1; }
  COMMON="--codec $CODEC --library-dir $LIB --qwen-tokenizer-model $QTOK --gemma-tokenizer-model $GTOK --llama-tokenizer-model $LTOK --inventory-from /mnt/storage/burstgpt-source/longtail_eval_v2/TRACE_MANIFEST.json"
  mkdir -p $Q/inputs $Q/chains $Q/reports
  (cd $D && python3 -m research_dev.scheduler.campaigns.burstgpt.quality.build_trace --pilot --split train --gsm8k $Q/data/train.jsonl --output-dir $Q/pilot $COMMON < /dev/null) >> "$LOG" 2>&1 || { log "pilot: build_trace failed"; return 1; }
  (cd $D && python3 -m research_dev.scheduler.campaigns.burstgpt.quality.derive_inputs $R/inputs-desktop-legacy-p0m1 $Q/pilot/shard-00 $Q/inputs/legacy-pilot < /dev/null) >> "$LOG" 2>&1 || { log "pilot: derive_inputs failed"; return 1; }
  until bash $R/rig_ready.sh < /dev/null | grep -q READY; do bash $R/rig_ready.sh < /dev/null | grep -q LATCHED && { log "pilot: charger latched"; return 1; }; sleep 60; done
  log "pilot run start"
  (cd $R && flock -w 7200 $LOCK python3 $R/run_chain_eval.py --status $Q/chains/CHAIN-legacy-pilot.jsonl --stage $R/stage --arm $Q/inputs/legacy-pilot < /dev/null) >> "$LOG" 2>&1 || { log "pilot: run failed"; return 1; }
  (cd $D && python3 -m research_dev.scheduler.campaigns.burstgpt.quality.score pilot --suite $Q/pilot/SUITE.json --key $Q/pilot/QUALITY_KEY.jsonl --arm legacy=$Q/inputs/legacy-pilot/run-eval/run --out $Q/reports/PILOT.json < /dev/null) >> "$LOG" 2>&1 || { log "pilot: scoring failed"; return 1; }
  log "pilot done: $Q/reports/PILOT.json"
}
if [ "${RUN_PILOT:-1}" = 1 ]; then pilot || log "pilot FAILED (see above); continuing with generality"; else log "pilot skipped (RUN_PILOT=0): streamed token ids are dropped for multi-byte-split tokens (62 % of Qwen GSM8K outputs) -> fix streaming first; budget 512 set from WS9 calibration"; fi
log "generality start (QUEUE-p3g)"
T=template-eval2-s2-longtail_eval_v2
exec bash $H/matrix_queue2.sh p3g 1 "w2d=desktop:${T}_w2 w2p=two-phone:${T}_w2 w3d=desktop:${T}_w3 w3p=two-phone:${T}_w3 w4d=desktop:${T}_w4 w4p=two-phone:${T}_w4 d2d=desktop:${T}_d2x d2p=two-phone:${T}_d2x d4d=desktop:${T}_d4x d4p=two-phone:${T}_d4x dhd=desktop:${T}_d0p5x dhp=two-phone:${T}_d0p5x" 20260930
