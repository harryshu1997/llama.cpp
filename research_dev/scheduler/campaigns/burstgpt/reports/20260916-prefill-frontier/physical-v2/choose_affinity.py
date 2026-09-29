"""Record an explicit hardware-capacity-based affinity for this physical experiment."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
ADB = ['adb', '-P', '5037', '-s', '3C15AU002CL00000', 'shell']


def read(path):
    return subprocess.check_output(ADB + ['cat', path], text=True).strip()


online = read('/sys/devices/system/cpu/online')
cpus = []
for group in online.split(','):
    ends = [int(value) for value in group.split('-')]
    cpus.extend(range(ends[0], ends[-1] + 1))
capacities = {cpu: int(read(f'/sys/devices/system/cpu/cpu{cpu}/cpu_capacity')) for cpu in cpus}
selected = [cpu for cpu in cpus if capacities[cpu] == max(capacities.values())]
mask = format(sum(1 << cpu for cpu in selected), 'x')
with (ROOT / 'CPU_AFFINITY.json').open('x') as stream:
    json.dump({'schema': 's42-phone-affinity-experiment-v1', 'online': online,
        'capacities': capacities, 'selected_cpus': selected, 'mask': mask,
        'scope': 'Only the gate-owned session subprocess tree; no global CPU settings changed.',
        'qualification': 'Same-boot USB qualification; model-path performance is pending this gate.'}, stream, indent=2)
    stream.write('\n')
print(mask)
