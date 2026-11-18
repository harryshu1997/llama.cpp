# start the session-long 1 s sampler (calibration of battery current against the coulomb counter); detached
T=/data/local/tmp/stream-energy-20261110
rm -f $T/runs/session.stop
nohup setsid sh $T/phone_sampler.sh $T/runs/session-power.txt $T/runs/session.stop 1 > /dev/null 2>&1 < /dev/null &
echo started
