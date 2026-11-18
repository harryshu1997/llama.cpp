# f32 reference for Pixel FFN layers 18/19 from the origin Qwen3-14B Q4_K_M model; compare with cpu and cl worker dumps.
import sys, numpy as np
sys.path.insert(0, '/mnt/storage/s42-trace-v2-20260921-prep/source/gguf-py')
import gguf
from gguf.quants import dequantize
r = gguf.GGUFReader('/mnt/repository/zhihao/home-zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf')
T = {t.name: t for t in r.tensors}
def W(name):
    t = T[name]; a = np.asarray(t.data).astype(np.float32) if t.tensor_type.name in ('F16','F32') else dequantize(t.data, t.tensor_type)
    return a.reshape(int(t.shape[1]), int(t.shape[0])) if a.ndim == 1 else a  # (rows=ne1, cols=ne0)
layers = [18, 19, 20, 21, 22, 23]; rows = 4
inputs = {(l, k): np.fromfile(f'/tmp/pxcl/inputs/input-layer{l}-row{k}.f16', dtype=np.float16).astype(np.float32) for l in layers for k in range(rows)}
CL = sys.argv[1] if len(sys.argv) > 1 else 'cl'
dumps = {n: np.fromfile(f'/tmp/pxcl/ab/runs/{n}/replay.burst4.f16', dtype=np.float16).astype(np.float32) for n in ('cpu', CL)}
dumps['cl'] = dumps[CL]
print('layer  type(down)   rel_L2(cpu vs f32)  rel_L2(cl vs f32)  rel_L2(cl vs cpu)')
for li, l in enumerate(layers):
    g = W(f'blk.{l}.ffn_gate.weight'); u = W(f'blk.{l}.ffn_up.weight'); d = W(f'blk.{l}.ffn_down.weight')
    dt = T[f'blk.{l}.ffn_down.weight'].tensor_type.name
    rc = []; rl = []; rx = []
    for k in range(rows):
        x = inputs[(l, k)]
        gg = g @ x; uu = u @ x; h = gg / (1 + np.exp(-gg)) * uu; y = d @ h
        yc = dumps['cpu'][(li * rows + k) * 5120:(li * rows + k + 1) * 5120]; yl = dumps['cl'][(li * rows + k) * 5120:(li * rows + k + 1) * 5120]
        n = np.linalg.norm(y); rc.append(np.linalg.norm(yc - y) / n); rl.append(np.linalg.norm(yl - y) / n); rx.append(np.linalg.norm(yl - yc) / n)
    print(f'{l:5d}  {dt:10s}  ' + ' '.join(f'{v:.4f}' for v in rc) + '   ' + ' '.join(f'{v:.4f}' for v in rl) + '   ' + ' '.join(f'{v:.4f}' for v in rx))
    del g, u, d
