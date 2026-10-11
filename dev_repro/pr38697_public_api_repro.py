import argparse
import json
import numpy as np
import openvino as ov
from openvino import opset13 as op

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=256)
p.add_argument('--k', type=int, default=128)
p.add_argument('--orientation', choices=['kn', 'nk'], default='kn')
p.add_argument('--gs', type=int, default=32)
p.add_argument('--device', default='GPU.1')
p.add_argument('--dump')
p.add_argument('--broadcast-n', action='store_true')
p.add_argument('--plugin')
a = p.parse_args()
g, s = 2, a.k // 2
x = op.parameter([1, 2, a.k], ov.Type.f16, name='input')
qn = 1 if a.broadcast_n else a.n
if a.orientation == 'kn':
    w_shape, q_shape, flat = (g, s, a.n), (g, 1, qn), [a.k, a.n]
else:
    w_shape, q_shape, flat = (a.n, g, s), (qn, g, 1), [a.n, a.k]
w_data = np.full(w_shape, 3, np.uint8)
scale_data = np.arange(g * qn, dtype=np.float16).reshape((g, qn)) % 3 + 2
zp_data = np.arange(g * qn, dtype=np.uint8).reshape((g, qn)) % 2 + 1
if a.orientation == 'nk':
    scale_data, zp_data = scale_data.T, zp_data.T
w = op.convert(op.constant(w_data), ov.Type.f16)
zp = op.convert(op.constant(zp_data.reshape(q_shape)), ov.Type.f16)
scale = op.constant(np.ascontiguousarray(scale_data.reshape(q_shape), dtype=np.float16))
weights = op.reshape(op.multiply(op.subtract(w, zp), scale), op.constant(flat, np.int64), False)
fc = op.matmul(x, weights, False, a.orientation == 'nk', name='fc')
m = ov.Model([fc], [x])
if a.plugin:
    from pathlib import Path
    path = Path('/tmp/PR38697_baseline/plugins.xml')
    path.write_text('<ie><plugins><plugin name="GPU" location="' + a.plugin + '"/></plugins></ie>')
    c = ov.Core(str(path))
else:
    c = ov.Core()
config = {'DYNAMIC_QUANTIZATION_GROUP_SIZE': a.gs,
          'INFERENCE_PRECISION_HINT': ov.Type.f16}
if a.dump:
    import os
    os.environ['OV_GPU_DUMP_GRAPHS_PATH'] = a.dump
print(json.dumps(vars(a)), flush=True)
compiled = c.compile_model(m, a.device, config)
print('COMPILE PASS', flush=True)
result = list(compiled([np.ones([1, 2, a.k], np.float16)]).values())[0]
if a.orientation == 'kn':
    ref_w = (w_data.astype(np.float16) - zp_data.reshape(q_shape)) * scale_data.reshape(q_shape)
    expected = ref_w.reshape(flat).astype(np.float32).sum(axis=0)
else:
    ref_w = (w_data.astype(np.float16) - zp_data.reshape(q_shape)) * scale_data.reshape(q_shape)
    expected = ref_w.reshape(flat).astype(np.float32).sum(axis=1)
print('max_error:', np.max(np.abs(result - expected)), 'result:', result.reshape(-1)[:8].tolist(), flush=True)
for node in compiled.get_runtime_model().get_ordered_ops():
    rt = node.get_rt_info()
    print(node.get_friendly_name(), node.get_type_name(), {k: str(rt[k].value) for k in rt}, flush=True)
