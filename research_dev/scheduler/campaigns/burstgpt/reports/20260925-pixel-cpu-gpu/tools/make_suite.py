"""Generate a phone-local Pixel FFN bench suite (RUN_PHONE.sh + SUITE.json).

Each arm starts one worker with a finite call budget equal to the number of
calls the replay client sends, so every worker exits normally on its own.
"""

import argparse
import json
from pathlib import Path

PH = '/data/local/tmp/s43-pixel-cpugpu-20260925-v1'
LAYERS = [18, 19, 20, 21, 22, 23]
ARTIFACT = 'sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718'
PROD_ENV = {
    'S42_PIXEL_CPU_POOL': '1', 'S42_PIXEL_PACKED_WEIGHTS': '1', 'S42_PIXEL_CPU_QUANT_RESIDUAL': '1',
    'S42_PIXEL_FUSED_RESIDUAL': '1', 'S42_PIXEL_CPU_PAIR_DOT': '1', 'S42_PIXEL_CPU_THREADS': '6',
    'S42_PIXEL_CPU_MASK': 'fc', 'S42_PIXEL_CPU_ROW_CHUNK': '64', 'S42_PIXEL_CPU_ROW_PROFILE': '0',
}
GPU_ONLY_ENV = {'S42_PIXEL_PACKED_WEIGHTS': '1'}
SAMPLE_FILES = ','.join([
    '/sys/devices/system/cpu/cpufreq/policy2/scaling_cur_freq',
    '/sys/devices/system/cpu/cpufreq/policy5/scaling_cur_freq',
    '/sys/devices/system/cpu/cpufreq/policy7/scaling_cur_freq',
    '/sys/class/devfreq/34f00000.gpu0/cur_freq',
    '/sys/class/devfreq/gmc_freq/cur_freq',
    '/sys/class/devfreq/200c0780.dsufreq/cur_freq',
    '/sys/class/devfreq/memss_freq/cur_freq',
    '/sys/class/devfreq/fabhbw_freq/cur_freq',
])
GPU_F16_ENV = {'S42_PIXEL_PACKED_WEIGHTS': '1', 'S42_PIXEL_GPU_EXPAND_F16': '1', 'S42_PIXEL_F16_SHADER': 'vec4_u1',
               'S42_PIXEL_F16_WG': '128', 'S42_PIXEL_F16_ROWS': '8', 'S42_PIXEL_F16_SUBGROUP': '128'}
FREQ_FILES = ','.join([
    '/sys/devices/system/cpu/cpufreq/policy2/scaling_cur_freq',
    '/sys/devices/system/cpu/cpufreq/policy5/scaling_cur_freq',
    '/sys/devices/system/cpu/cpufreq/policy7/scaling_cur_freq',
    '/sys/class/devfreq/34f00000.gpu0/cur_freq',
])


def segments_for(rows_list, burst_steps, burst_warmup, prod_steps, prod_warmup, gap_call_us, gap_token_us):
    out = []
    for rows in rows_list:
        if burst_steps:
            out.append(dict(name=f'burst-m{rows}', rows=rows, steps=burst_steps, warmup=burst_warmup,
                            gap_call_us=0, gap_token_us=0))
        if prod_steps:
            out.append(dict(name=f'prod-m{rows}', rows=rows, steps=prod_steps, warmup=prod_warmup,
                            gap_call_us=gap_call_us, gap_token_us=gap_token_us))
    return out


def arm_script(arm, suite):
    segs = arm['segments']
    calls = sum((s['warmup'] + s['steps']) * len(LAYERS) for s in segs)
    backend = arm.get('backend', 'CPU')
    binary = f'{PH}/prod/llama-ffn-split-worker' if arm.get('prod') else f'{PH}/llama-ffn-split-worker'
    env = ' '.join(f'{k}={v}' for k, v in sorted(arm['env'].items()))
    worker = (f'env LD_LIBRARY_PATH={PH} {env} {binary} -m {PH}/QWEN_PACKED.ffn.gguf --artifact-sha256 {ARTIFACT} '
              f'--layers {",".join(map(str, LAYERS))} --columns 17408 --column-quantum 4352 --backend {backend} '
              f'--port {suite["port"]} --bind 127.0.0.1 --f16-io --max-tokens 4 --max-requests {calls}')
    seg_args = ' '.join('--segment {name}:{rows}:{steps}:{warmup}:{gap_call_us}:{gap_token_us}'.format(**s) for s in segs)
    client = (f'{PH}/pixel-ffn-replay --port {suite["port"]} --artifact {ARTIFACT} --layers {",".join(map(str, LAYERS))} '
              f'--columns 17408 --max-tokens 4 --input-dir {PH}/inputs --out {PH}/runs/{suite["name"]}/{arm["name"]}/replay '
              f'--freq-files {FREQ_FILES} --sample-files {SAMPLE_FILES} --sample-us 1000 {seg_args}')
    arm['calls'] = calls
    arm['worker_command'] = worker
    arm['client_command'] = client
    return f'''run_arm {arm["name"]} {calls} '{worker}' '{client}' || exit $?
sleep {suite["cooldown_s"]}
'''


HEADER = r'''#!/system/bin/sh
PH=__PH__
RUN=$PH/runs/__SUITE__
cd $PH || exit 2
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || { echo 'FAIL phone lock busy'; exit 3; }
mkdir $RUN || exit 4
ps -A -o PID,ARGS > $RUN/PROCESSES_BEFORE.txt
if grep -q '[l]lama-ffn' $RUN/PROCESSES_BEFORE.txt; then echo 'FAIL existing worker'; exit 5; fi
cat /proc/sys/kernel/random/boot_id > $RUN/BOOT_BEFORE.txt
snapshot() {
    dumpsys battery > "$1/BATTERY_$2.txt" 2>&1
    for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done > "$1/THERMAL_$2.txt" 2>&1
    for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq /sys/class/devfreq/34f00000.gpu0/cur_freq; do echo "$f $(cat $f)"; done > "$1/FREQ_$2.txt" 2>&1
}
run_arm() {
    name=$1; calls=$2; worker=$3; client=$4
    mkdir $RUN/$name || return 7
    snapshot $RUN/$name BEFORE
    date +%s.%N > $RUN/$name/START.txt
    sh -c "exec $worker" > $RUN/$name/worker.log 2>&1 &
    pid=$!
    echo $pid > $RUN/$name/PID.txt
    ready=0
    for attempt in $(seq 1 600); do
        if grep -q '\[ffn-worker\] ready backend=' $RUN/$name/worker.log; then ready=1; break; fi
        if ! kill -0 $pid 2>/dev/null; then echo "FAIL $name startup"; wait $pid; echo $? > $RUN/$name/WORKER_EXIT.txt; return 8; fi
        sleep 0.5
    done
    if [ "$ready" != 1 ]; then echo "FAIL $name readiness"; kill -TERM $pid; wait $pid; return 9; fi
    for t in /proc/$pid/task/*; do echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status) $(grep -E 'uclamp' $t/sched 2>/dev/null | tr -s ' ' | tr '\n' ' ')"; done > $RUN/$name/THREADS.txt 2>&1
    sh -c "exec $client" > $RUN/$name/client.log 2>&1
    cstatus=$?
    echo $cstatus > $RUN/$name/CLIENT_EXIT.txt
    if [ "$cstatus" != 0 ]; then
        sleep 2
        # client gone: the worker is idle in accept(), not in a call
        if kill -0 $pid 2>/dev/null; then kill -TERM $pid; fi
        wait $pid; echo $? > $RUN/$name/WORKER_EXIT.txt
        echo "FAIL $name client $cstatus"; return 10
    fi
    wait $pid
    wstatus=$?
    echo $wstatus > $RUN/$name/WORKER_EXIT.txt
    date +%s.%N > $RUN/$name/END.txt
    snapshot $RUN/$name AFTER
    if [ "$wstatus" != 0 ]; then echo "FAIL $name worker $wstatus"; return 11; fi
    echo "PASS $name"
}
snapshot $RUN SUITE_BEFORE
'''

FOOTER = r'''snapshot $RUN SUITE_AFTER
cat /proc/sys/kernel/random/boot_id > $RUN/BOOT_AFTER.txt
ps -A -o PID,ARGS > $RUN/PROCESSES_AFTER.txt
if grep -q '[l]lama-ffn' $RUN/PROCESSES_AFTER.txt; then echo 'FAIL remaining worker'; exit 12; fi
cmp $RUN/BOOT_BEFORE.txt $RUN/BOOT_AFTER.txt || exit 13
echo PASS > $RUN/DONE.txt
echo 'PASS all arms'
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('outdir', type=Path)
    args = parser.parse_args()
    suite = json.loads(args.config.read_text())
    body = HEADER.replace('__PH__', PH).replace('__SUITE__', suite['name'])
    for arm in suite['arms']:
        base = {'cpu': PROD_ENV, 'gpu': GPU_ONLY_ENV, 'gpu_f16': GPU_F16_ENV}[arm.get('base', 'gpu' if arm.get('backend') == 'Vulkan0' else 'cpu')]
        env = dict(base)
        env.update(arm.get('extra_env', {}))
        arm['env'] = env
        if 'segments' not in arm:
            arm['segments'] = list(suite['segments']) if isinstance(suite['segments'], list) else segments_for(**suite['segments'])
        body += arm_script(arm, suite)
    body += FOOTER
    args.outdir.mkdir(parents=True, exist_ok=False)
    (args.outdir / 'RUN_PHONE.sh').write_text(body)
    (args.outdir / 'SUITE.json').write_text(json.dumps(suite, indent=2) + '\n')
    print(args.outdir / 'RUN_PHONE.sh', sum(a['calls'] for a in suite['arms']), 'calls')


if __name__ == '__main__':
    main()
