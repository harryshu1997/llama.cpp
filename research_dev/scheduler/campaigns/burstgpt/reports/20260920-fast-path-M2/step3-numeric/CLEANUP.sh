#!/usr/bin/env bash
set -euo pipefail
deploy=/mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7
exec 9>/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock
flock -n 9
python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
p=Path('/mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7/physical')
rows=[]
for run in ('run1','run2'):
    identity=json.loads((p/run/'SERVER_IDENTITY.json').read_text())
    pid=identity['pid']; proc=Path(f'/proc/{pid}/cmdline')
    argv=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
    owned_alive=bool(argv and '/mnt/storage/s42-fast-path-M2-numeric-20260920-2d61e7/cuda-build/bin/llama-server' in argv)
    scope=f'fast-path-m2-{run}-2d61e7.scope'
    status=subprocess.run(['systemctl','--user','is-active',scope],text=True,capture_output=True).stdout.strip()
    close=json.loads((p/run/'PHONE_CLOSE.json').read_text())
    row={'run':run,'server_pid':pid,'current_argv':argv,'owned_server_alive':owned_alive,
         'scope':scope,'scope_state':status,'phone_close':close}
    rows.append(row)
    if owned_alive or status!='inactive':raise RuntimeError(f'run remains active: {row}')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
if gpu:raise RuntimeError('GPU is occupied: '+gpu)
adb=['/usr/bin/adb','-P','5037','-s','3C15AU002CL00000']
state=subprocess.check_output([*adb,'get-state'],text=True).strip()
kernel=subprocess.check_output([*adb,'shell','uname','-r'],text=True).strip()
if state!='device' or kernel!='6.12.23-android16-5-o-g227664cbe007-4k':raise RuntimeError('OP15 identity differs')
record={'status':'PASS','at_utc':datetime.now(timezone.utc).isoformat(),'arms':rows,
        'gpu_processes':gpu,'rig_lock_acquired_nonblocking':True,'adb_port':5037,
        'phone_serial':'3C15AU002CL00000','phone_state':state,'phone_kernel':kernel}
with (p/'CLEANUP.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps({key:value for key,value in record.items() if key!='arms'},indent=2))
PY
