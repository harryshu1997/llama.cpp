"""Read-only live capacity estimates; never claim an unexecuted context passes."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from research_dev.scheduler.campaigns.burstgpt import remote_resident_gate as gate
from research_dev.scheduler import DeviceMemoryCapacity, RuntimePlacementSnapshot
from research_dev.scheduler.adapters import nvidia_gpu_snapshot

command = json.loads((ROOT / 'GATE_COMMAND.json').read_text())
sys.argv = command[1:]
args = gate.parse_args()
models = gate.runner._load_trace_models(args)
scheduler, manifests, _ = gate.runner._build_scheduler(args, models, load_adaptive_observations=False)
manifest = manifests[models.expected_gemma.model_id]
control = models.catalog.desktop_control_by_artifact[manifest.artifact_sha256]
source = models.catalog.composite_executor_by_id[control.executor_id]
gpu_id = source.adapter_parameters['gpu_device_id']
pool = models.catalog.placement_profile.devices[gpu_id].memory_pool_id
gpu = nvidia_gpu_snapshot()
host = gate._host_memory()
memory = RuntimePlacementSnapshot(snapshot_id='context-frontier-live-nvml', captured_at_us=0,
    valid_until_us=60_000_000, capacities={pool: DeviceMemoryCapacity(pool,
        gpu['memory_total_bytes'], gpu['memory_total_bytes'] - gpu['memory_free_bytes'], 512 * 1024**2)})
rows = []
for context in (8192, 16384, 32768, 65536, 131072, 262144):
    try:
        selection = scheduler.select_live_vram_desktop_parent(manifest.model_id, memory,
            cuda_graph_mode='default', preserve_placement=True, maximum_gpu_layers=23,
            launch_overrides={'context_size': context})
        selected = selection.selected
        placement = {row.operator_id: row.primary_device_id for row in selected.operator_placements}
        cpu_kv = sum(manifest.preallocated_kv_cache_bytes(row.operator_id, context_size=context,
            parallel=1, sliding_window_padding_tokens=512) for row in manifest.operators
            if row.kind == 'kv_cache' and placement[row.operator_id] == source.adapter_parameters['cpu_device_id'])
        host_required = (manifest.artifact_bytes * 1_050_000 + 999_999) // 1_000_000 + cpu_kv + 768 * 1024**2
        rows.append({'context_size': context, 'status': 'PREDICTED_ONLY',
            'gpu_required_bytes': selected.required_with_reserve_bytes,
            'gpu_kv_bytes': selected.gpu_kv_bytes, 'cpu_kv_bytes': cpu_kv,
            'host_full_load_required_bytes': host_required,
            'predicted_host_feasible': host_required <= host['MemAvailable'],
            'same_gpu_placement': selected.placement_sha256 == control.placement_sha256})
    except Exception as error:
        rows.append({'context_size': context, 'status': 'NOT_ADMITTED', 'error': str(error)})
result = {'schema': 's42-context-frontier-capacity-screen-v1', 'host': host, 'gpu': gpu,
    'rows': rows, 'note': 'Read-only planning, not a physical context capacity result; no omission credit before proof.'}
with (ROOT / 'CAPACITY_SCREEN.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(rows))
