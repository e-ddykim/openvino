# Reproducers for compressed-FC orientation regressions (review of PR #38697)

Branch base: PR #38697 head `391d540f834e4b3f411c3369e844ebd09ede62b3`.

These tests are **reproducers**: they assert the *intended correct* behavior and
therefore **FAIL on this revision by design**. They are not proposed as final
product tests.

## Finding 1 — `DynamicQuantizeFullyConnected` computes weight group size on the wrong axis for non-transposed scale/ZP

`ConvertFCToCompressed` (has_transpose path) now emits weights `[N,K]`
(`transpose_b=true`) together with scale/ZP `[G,N]`
(`transpose_b_scale=false`, `transpose_b_zp=false`).
`dynamic_quantize_fully_connected.cpp` still derives `k_axis` from the weights'
`transpose_b` and applies it to the scale/ZP shapes, so the weight group size
becomes `K/N` instead of `K/G`:

- `N > K` (e.g. K=128, N=256, G=2): group size `128/256 = 0` →
  `required_group_size % adj_group_size` evaluates `0 % 0` → **SIGFPE kills the
  compiling process** (`compile_model()` does not return an exception).
- `N < K` (e.g. K=128, N=64, G=2): group size `128/64 = 2` → the pass bails out
  at the `adj_group_size < 32` check → **DynamicQuantize is silently dropped**
  (FC runtime precision falls back to f16 instead of i8 with GS32).

Reproducer (transformation unit test, no GPU needed):
`src/plugins/intel_gpu/tests/unit/transformations/dynamic_quantize_fully_connected_test.cpp`

```bash
cmake --build build --target ov_gpu_unit_tests -j8

# control (old orientation): PASS
bin/intel64/Debug/ov_gpu_unit_tests \
  --gtest_filter='*DynamicQuantizeFullyConnectedTest*/*_TransposedParameters'

# new orientation, N<K: FAIL (DQ dropped: 5 inputs vs 8)
bin/intel64/Debug/ov_gpu_unit_tests \
  --gtest_filter='*DynamicQuantizeFullyConnectedTest*/*N64_NonTransposedParameters'

# new orientation, N>K: SIGFPE (run in a separate process; exit code 136)
bin/intel64/Debug/ov_gpu_unit_tests \
  --gtest_filter='*DynamicQuantizeFullyConnectedTest*/*N256_NonTransposedParameters'
```

## Finding 2 — OCL compressed FC misreads `[G,1]` scale/ZP broadcast across `N>1` output channels

`prepare_quantization.cpp` skips the scale/ZP reorder whenever
`transpose_b_*=false`, while the OCL `convert_decompression_tensor` reinterpret
only fires when `Feature() == OFM()`. For a valid `[G,1]` scale/ZP broadcast
across `N>1` output channels, neither path applies: the kernel receives
`Batch=G, Feature=1` and treats `G` groups as a single group → wrong values.
Before the PR the transformation transposed such scale/ZP (`[G,1] -> [1,G]`),
so the case worked.

Reproducer (primitive execution, forces OCL kernels):
`src/plugins/intel_gpu/tests/unit/test_cases/fully_connected_gpu_test.cpp`
(`smoke/fully_connected_gpu_compressed_broadcast`)

```bash
# 4 FAIL (NonTransposed) / 4 PASS (Transposed control) on both GPU.0 and GPU.1
bin/intel64/Debug/ov_gpu_unit_tests --device_suffix=0 \
  --gtest_filter='*fully_connected_gpu_compressed_broadcast*'
```

Observed vs expected (expected `112` for all outputs):
`bfyx_ref`: `[128, 96, 128]`, `bf_tiled`: `[128, 96, 0]`.

## Public-API reproducer

`pr38697_public_api_repro.py` builds a plain ONNX-style decompression graph
(Convert/Subtract/Multiply/Reshape + MatMul) that `ConvertFCToCompressed`
converts to the new orientation, then compiles on GPU:

```bash
export PYTHONPATH="$PWD/bin/intel64/Debug/python"
export LD_LIBRARY_PATH="$PWD/bin/intel64/Debug"

# SIGFPE during compile_model (exit code 136)
python3 dev_repro/pr38697_public_api_repro.py --device GPU.1 --k 128 --n 256 --orientation kn --gs 32

# numerically correct but DynamicQuantize missing from the runtime graph
python3 dev_repro/pr38697_public_api_repro.py --device GPU.1 --k 128 --n 64 --orientation kn --gs 32

# wrong outputs [128,96,0,...] instead of [112,112,112,...]
python3 dev_repro/pr38697_public_api_repro.py --device GPU.0 --k 32 --n 3 --orientation kn --gs 0 --broadcast-n
```

## Scope note

The full (non-broadcast) `[G,N]` scale/ZP path was verified working on both
backends (OCL `bfyx_ref`/`bf_tiled`/`gemv`(u4) and oneDNN, caching on/off).
The two findings above are limited to the consumers listed in each section.
