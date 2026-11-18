# One metered run on the phone (root). No output until the end; nothing else is started or changed.
#   sh phone_run.sh TAG ART NEMBD NFF GEGLU IDLE WARM SECS SHARD GROUP [GROUP]   GROUP = PORT:LAYERSPEC:LAYERLIST
T=/data/local/tmp/stream-energy-20261110
B=/data/local/tmp/s42-ffn-shards-20260904-v1-bin
BACKEND=HTP0
MAXTOK=1
# another phone sets B (worker + library directory), BACKEND, MAXTOK and WENV (worker environment) here
[ -f $T/device.conf ] && . $T/device.conf
[ -n "$WENV" ] || WENV="LD_LIBRARY_PATH=$B ADSP_LIBRARY_PATH=$B GGML_HEXAGON_MBUF=4192 GGML_HEXAGON_NDEV=1 GGML_HEXAGON_NHVX=4 GGML_HEXAGON_VMEM=3328 S41_DISABLE_GRAPH_CACHE=1"
TAG=$1; ART=$2; D=$3; NFF=$4; GEGLU=$5; IDLE=$6; WARM=$7; SECS=$8; SHARD=$9; shift 9
O=$T/runs/$TAG
[ -e "$O" ] && { echo "ABORT $O exists"; exit 2; }
if pgrep -f "llama-ffn-split-[w]orker" > /dev/null; then echo "ABORT another FFN worker is running"; exit 3; fi
mkdir -p $O
sh $T/phone_sampler.sh $O/power.txt $O/stop 0.25 &
SPID=$!
sleep 5
read up rest < /proc/uptime; echo "launch_uptime $up" > $O/marks.txt
WPIDS=""; GARGS=""; n=0
for G in "$@"; do
  port=${G%%:*}; rest=${G#*:}; spec=${rest%%:*}; list=${rest#*:}
  env $WENV \
      $B/llama-ffn-split-worker -m $SHARD --artifact-sha256 $ART --layers $spec --columns $NFF --column-quantum ${QUANTUM:-$NFF} \
      --backend $BACKEND --bind 127.0.0.1 --f16-io --max-tokens $MAXTOK --max-requests 0 --port $port > $O/worker$n.log 2>&1 &
  WPIDS="$WPIDS $!"; GARGS="$GARGS --group $port:$list"
  i=0
  while [ $i -lt 600 ]; do
    grep -q "ready backend=" $O/worker$n.log && break
    kill -0 $! 2> /dev/null || break
    sleep 0.2; i=$((i+1))
  done
  if ! grep -q "ready backend=" $O/worker$n.log; then
    echo "ABORT worker $n not ready"; tail -n 6 $O/worker$n.log
    kill $WPIDS 2> /dev/null; touch $O/stop; wait $SPID; exit 4
  fi
  n=$((n+1))
done
read up rest < /proc/uptime; echo "ready_uptime $up" >> $O/marks.txt
GE=""; [ "$GEGLU" = 1 ] && GE="--geglu"
$T/bin/ffn-stream-client $GARGS --artifact $ART --n-embd $D --columns $NFF $GE --idle-s $IDLE --warmup-s $WARM \
    --seconds $SECS --out $O/calls.csv > $O/client.log 2>&1
echo "client_rc $?" >> $O/marks.txt
read up rest < /proc/uptime; echo "client_done_uptime $up" >> $O/marks.txt
kill $WPIDS 2> /dev/null
sleep 3
touch $O/stop; wait $SPID
echo "status $(dumpsys thermalservice 2> /dev/null | grep -m1 'Thermal Status')" >> $O/marks.txt
cat $O/marks.txt; cat $O/client.log; grep -c measure $O/calls.csv
