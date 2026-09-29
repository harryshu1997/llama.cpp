#!/system/bin/sh
cd /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1 || exit 2
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || exit 3
mkdir raw || exit 4
ps -A -o PID,ARGS > raw/PROCESSES_BEFORE.txt
if grep -Eq '[l]lama-ffn|/[p]ixel-bandwidth ' raw/PROCESSES_BEFORE.txt; then echo 'FAIL existing worker'; exit 5; fi
cat /proc/sys/kernel/random/boot_id > raw/BOOT_BEFORE.txt
cat /proc/meminfo > raw/MEMORY_BEFORE.txt
sha256sum /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v1_u1.spv /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u4.spv /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u8.spv > raw/HASHES.txt || exit 6
cmp raw/HASHES.txt EXPECTED_HASHES.sha256 || exit 6
snapshot() {
    date -u
    dumpsys battery
    for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq /sys/class/devfreq/*/cur_freq; do
        echo "$f"
        cat "$f"
    done
}
run_arm() {
    name="$1"
    shift
    mkdir "raw/$name" || return 7
    snapshot > "raw/$name/BEFORE.txt" 2>&1
    "$@" > "raw/$name/RESULTS.jsonl" 2> "raw/$name/STDERR.txt"
    code=$?
    echo "$code" > "raw/$name/EXIT.txt"
    snapshot > "raw/$name/AFTER.txt" 2>&1
    if [ "$code" != 0 ]; then echo "FAIL $name exit=$code"; return "$code"; fi
    echo "PASS $name"
}
run_arm 00-cpu4-free /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth cpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 65536 4 0 0 3 512 || exit $?
run_arm 01-cpu1-pin /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth cpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 65536 1 80 0 3 512 || exit $?
run_arm 02-cpu1-prefetch1024 /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth cpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 65536 1 80 1024 3 512 || exit $?
run_arm 03-gpu-scalar /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth gpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v1_u1.spv 1 128 65536 4 0 0 3 512 || exit $?
run_arm 04-gpu-v4-16k /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth gpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 16384 4 0 0 3 512 || exit $?
run_arm 05-gpu-v4-4k /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth gpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 4096 4 0 0 3 512 || exit $?
run_arm 06-gpu-v4-16k-u4 /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth gpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u4.spv 4 128 16384 4 0 0 3 512 || exit $?
run_arm 07-joint-cpu1 /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth both /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 16384 1 80 0 3 512 || exit $?
run_arm 08-joint-cpu2 /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth both /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 16384 2 c0 0 3 512 || exit $?
run_arm 09-gpu-v4-16k /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth gpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 16384 4 0 0 3 512 || exit $?
run_arm 10-cpu1-pin /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth cpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 65536 1 80 0 3 512 || exit $?
run_arm 11-cpu4-free /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth cpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 65536 4 0 0 3 512 || exit $?
run_arm 12-joint-cpu2 /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth both /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 16384 2 c0 0 3 512 || exit $?
run_arm 13-joint-cpu1 /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth both /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 16384 1 80 0 3 512 || exit $?
run_arm 14-gpu-v4-16k /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth gpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 16384 4 0 0 3 512 || exit $?
run_arm 15-cpu1-pin /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth cpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v4_u1.spv 4 128 65536 1 80 0 3 512 || exit $?
run_arm 16-gpu-scalar /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/pixel-bandwidth gpu /data/local/tmp/s42-pixel10pro-bandwidth-20260923-confirm-v1/read_v1_u1.spv 1 128 65536 4 0 0 3 512 || exit $?
cat /proc/sys/kernel/random/boot_id > raw/BOOT_AFTER.txt
ps -A -o PID,ARGS > raw/PROCESSES_AFTER.txt
cat /proc/meminfo > raw/MEMORY_AFTER.txt
if grep -Eq '[l]lama-ffn|/[p]ixel-bandwidth ' raw/PROCESSES_AFTER.txt; then echo 'FAIL remaining worker'; exit 12; fi
cmp raw/BOOT_BEFORE.txt raw/BOOT_AFTER.txt || exit 13
echo PASS > raw/DONE.txt
echo 'PASS all arms'
