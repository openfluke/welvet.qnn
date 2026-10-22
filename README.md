# Welvet × Qualcomm QNN (Hexagon NPU)

**QNN** = Qualcomm AI Engine Direct (API). **QAIRT** = the SDK that ships it.  
**HTP** = Hexagon Tensor Processor backend (`QnnHtp.dll`) — the Snapdragon **NPU**.

Authority for this matrix: QAIRT **2.42** `SupportedOps.html` HTP columns + `w2a/suites/go_v_qnn` / `accel.QualcommHTPCaps()`.  
Pass: **forward / inference only** — no NormalBP or TrainMode on NPU.

---

## Compatible with Welvet today (wired)

These Welvet layers have a C++ QNN graph path (`welvet.cpp` + `SyncToHTP` / C ABI `backend=2`).  
**Run dtype today:** `float32` (often executed as **FP16 math** on HTP).

| Welvet layer | QNN op(s) | HTP dtypes (SDK) | Notes |
|--------------|-----------|------------------|--------|
| **Dense** | FullyConnected, MatMul | bf16, fp16, int16, int8 | Primary path |
| **Softmax** | Softmax, LogSoftmax | bf16, fp16, int16, int8 | Standard / Temperature@1 only |
| **RMSNorm** | RmsNorm | fp16, int16, int8 | **No BF16** on HTP in 2.42 |
| **LayerNorm** | LayerNorm | bf16, fp16, int16, int8 | |

**Host dtypes Welvet will schedule on HTP (SDK ceiling):**  
`float32` (via fp16), `float16`, `bfloat16` (Windows **V81+**), `int8`, `uint8`, `int16`, `uint16`.

**Not on HTP:** `float64`, FP8/FP6/FP4, sub-byte ints, nf4, ternary/binary, complex, native int/uint/uintptr, etc.

---

## NPU can do it — Welvet not wired yet

SDK has a direct HTP op (or a clear compose path), but Welvet has **no** `SyncToHTP` / QNN graph for these yet.

### Native QNN ops (direct)

| Welvet layer | QNN op(s) | HTP dtypes | Gap |
|--------------|-----------|------------|-----|
| CNN2 | Conv2d, DepthWiseConv2d | bf16, fp16, int16, int8 | No HTP forward in Welvet |
| CNN3 | Conv3d | fp16, int16, int8 | No BF16; not wired |
| ConvT2 | TransposeConv2d | bf16, fp16, int16, int8 | Not wired |
| ConvT3 | TransposeConv3d | fp16, int16, int8 | No BF16; not wired |
| LSTM | Lstm | fp16, int8 | No BF16/INT16; not wired |
| Embedding | Gather | bf16, fp16, int16, int8, others | Indices often INT32; not wired |

### Compose (build from HTP primitives)

| Welvet layer | Would use | Gap |
|--------------|-----------|-----|
| MHA | MatMul, Softmax, Mul, Transpose, Reshape | No MultiHead op; MaskedSoftmax HTP=NO in 2.42 |
| SwiGLU | FC/MatMul, Mul, Sigmoid | SiLU ≈ x·σ(x); not wired as one graph |
| Sequential | Child graph | Only if every child is HTP-capable |
| Residual | ElementWiseAdd | Not wired |

### Extra dtypes on wired layers

NPU columns exist for **fp16 / bf16 / int8 / int16** on Dense/Softmax/norms, but Welvet’s **run allowlist is still float32 only**. Quantized and native fp16/bf16 SyncToHTP paths are not exposed yet.

---

## Welvet layers the NPU cannot carry (as-is)

| Welvet layer | Why |
|--------------|-----|
| CNN1 | Conv1d **HTP=NO** all dtypes |
| ConvT1 | TransposeConv1d **HTP=NO** |
| RNN | No QNN RNN on HTP (only Gru/Lstm) |
| KMeans | No clustering op |
| Mamba | No SSM / selective-scan op in 2.42 |
| GDN / Metacognition | Welvet-specific; no QNN equivalent |
| Parallel / Stack | Hosts — not single HTP leaves (children may map individually) |

---

## What else QNN/HTP can do (no Welvet layer yet)

Useful SDK capability that Welvet does **not** model as a first-class layer (non-exhaustive; see QAIRT OpDef):

| Area | Examples (QNN) | Welvet status |
|------|----------------|---------------|
| Pooling | PoolAvg2d/3d, PoolMax2d/3d, L2 pool | No dedicated pool layers |
| Activations | Relu, Gelu, Tanh, PRelu, HardSwish, ElementWiseNeuron | Mostly fused in Dense act; no standalone NPU acts |
| Elementwise | Add/Mul/Sub/Div, unary ops | Only as compose pieces (e.g. Residual) |
| Reduce / Softmax variants | Reduce*, LogSoftmax, ArgMax | Softmax wired; LogSoftmax / reduces not |
| Normalization | BatchNorm, InstanceNorm, GroupNorm | Only LayerNorm / RMSNorm |
| Attention extras | (pattern) multi-head, RoPE-style ops if any | MHA compose only; no RoPE layer |
| RNN family | **Gru** (HTP), Lstm | LSTM not wired; no GRU layer |
| Image / CV | Pad, Crop, Resize, ChannelShuffle, SpaceToDepth | No Welvet CV preprocess layers |
| Quant tooling | Encoding, per-channel scales, UFIXED graphs | No Welvet quant-deploy path on HTP |
| Backends | QnnCpu, QnnGpu, QnnSaver/Ir | Welvet targets HTP (optional Cpu for bring-up); not Adreno as a Welvet backend yet |
| Graph packaging | Context binary / DLC / qnn-net-run models | Welvet builds small online graphs; no DLC import |

---

## Status legend

| Tag | Meaning |
|-----|---------|
| **Wired** | C++ QNN path + run allowlist (`dense`, `softmax`, `rmsnorm`, `layernorm` × f32) |
| **Native (gap)** | HTP op exists; Welvet SyncToHTP not implemented |
| **Compose (gap)** | Doable from HTP ops; not a single Welvet→QNN mapping yet |
| **None** | Do not schedule on NPU |

Live probe: `welvet_qnn_probe.exe` / `SyncToHTP`. On some Windows Snapdragon hosts `backendCreate` still faults (`0xC00000FD`) even when the Hexagon NPU device is present — see `welvet.cpp/docs/QNN_WINDOWS_ARM64.md`.

---

## Related

- `accel` — `CheckHTP` / `QualcommHTPCaps`
- `w2a/suites/go_v_qnn` — printable matrix + Go/Native/HTP triad
- `native/cpp/windows_arm64` — shipped `libwelvet.dll` + probe
- `welvet.cpp` — `src/backends/qnn_htp.cpp`
