# runs on the phone as root
D=/data/local/tmp/s43-pixel-cl; DF=/sys/class/devfreq; Q=/sys/kernel/debug/google_pm_qos
echo "=== OPP tables"
for d in gmc_freq irm_gmc_freq memss_freq fabhbw_freq; do echo "-- $d: $(ls /sys/kernel/debug/opp/platform-$d 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; done
echo "=== PM QoS at idle"
for d in gmc_freq irm_gmc_freq memss_freq fabhbw_freq; do echo "-- $d:"; cat $Q/$d 2>/dev/null | head -24; done
echo "=== PM QoS during a GPU stream"
(cd $D && taskset 80 ./clgemv clgemv.cl stream_read16 17408 5120 4000 64 512 >/dev/null) &
sleep 2.5
for d in gmc_freq irm_gmc_freq memss_freq fabhbw_freq; do echo "-- $d:"; cat $Q/$d 2>/dev/null | head -24; done
wait
echo "=== PM QoS during a CPU stream (6 threads)"
(cd $D && ./cpustream 512 6 4 >/dev/null) &
sleep 2.5
for d in gmc_freq irm_gmc_freq memss_freq; do echo "-- $d:"; cat $Q/$d 2>/dev/null | head -24; done
wait
echo "=== CPU stream vs memss cap (does the CPU path use memss?)"
cd $D && ./cpustream 512 6 3
echo 466000000 > $DF/memss_freq/max_freq; echo 466000000 > $DF/memss_freq/min_freq
echo "memss 466: $(./cpustream 512 6 3)"
echo 932000000 > $DF/memss_freq/max_freq; echo 178000000 > $DF/memss_freq/min_freq
echo "=== GPU stream vs memss steps"
for f in 666000000 800000000 932000000; do echo $f > $DF/memss_freq/max_freq; echo $f > $DF/memss_freq/min_freq; echo "memss $f: $(taskset 80 ./clgemv clgemv.cl stream_read16 17408 5120 30 64 512 | tail -1 | sed 's/.*"GBps_med":\([0-9.]*\).*/\1 GBps/')"; done
echo 932000000 > $DF/memss_freq/max_freq; echo 178000000 > $DF/memss_freq/min_freq; echo "restored memss $(cat $DF/memss_freq/min_freq)-$(cat $DF/memss_freq/max_freq)"
