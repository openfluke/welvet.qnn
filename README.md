# Welvet × Qualcomm QNN (Hexagon NPU)

**QNN** = Qualcomm AI Engine Direct (API). **QAIRT** = the SDK that ships it.  
**HTP** = Hexagon Tensor Processor backend (`QnnHtp`) — the Snapdragon **NPU**.

This repo (`welvet.qnn`) is a **standalone C++17** library that talks to QNN/HTP only.  
Target: **NPU forward**. Saver is a graph-compose sandbox when HTP create is broken.  
CPU/GPU probes exist only to diagnose host faults — they are **not** the product.

Authority for this matrix: QAIRT **2.42** `SupportedOps.html` HTP columns.  
Pass: **forward / inference only** — no backprop / TrainMode on NPU.

**Library version:** `0.1.0` (CMake `project(welvet_qnn VERSION 0.1.0)`)  
**Verified against SDK:** QAIRT **2.42.0.251225** · QNN API **2.32** · HTP backendId **6**  
**Host smoke (2026-10-23):** Windows ARM64 · Snapdragon X126100 · `pass=7 fail=0 skip=2`

---

## Implementation checklist (welvet.qnn)

Tick = code path exists in this tree. **NPU numeric** = HTP `backendCreate` + real execute on Hexagon.

### Infra

| Item | Ver | Status | Notes |
|------|-----|--------|-------|
| CMake portable C++17 (Win/Linux/mac stub) | 0.1.0 | ✅ | Optional `QAIRT_SDK_ROOT` |
| Dynamic load `QnnHtp` (+ Cpu/Gpu/Saver/Ir for diagnose) | 0.1.0 | ✅ | No static link to QNN |
| **Detect NPU stack** (SDK dir, DLL, `getProviders`, buildId) | 0.1.0 | ✅ **works on host** | Does **not** call `backendCreate` |
| Session open **HTP first** | 0.1.0 | ⬜ blocked | Probe `0xC00000FD` STACK_OVERFLOW |
| Windows child-process `backendCreate` probe | 0.1.0 | ✅ | Soft-skip so smoke survives |
| Saver fallback (graph compose only) | 0.1.0 | ✅ | **Not NPU** — bring-up only |
| CPU open as product path | — | ❌ won't do | Diagnose only; same crash as HTP here |
| Live **HTP execute** / numeric golden | — | ⬜ | Blocked on create |
| Context binary / DLC load | — | ⬜ | Later |
| Quant / fp16 / bf16 / int8 host allowlist | — | ⬜ | v0.1 host dtype = float32 only |

### Layers on NPU (HTP)

| Layer | QNN op | Blank API | Graph on Saver | Graph on **HTP** | Numeric NPU | Ver |
|-------|--------|-----------|----------------|------------------|-------------|-----|
| Dense | FullyConnected | ✅ | ✅ | ⬜ create blocked | ⬜ | 0.1.0 |
| Softmax | Softmax | ✅ | ✅ | ⬜ | ⬜ | 0.1.0 |
| RMSNorm | RmsNorm | ✅ | ✅ | ⬜ | ⬜ | 0.1.0 |
| LayerNorm | LayerNorm | ✅ | ✅ | ⬜ | ⬜ | 0.1.0 |
| CNN2 / CNN3 / ConvT* | Conv* | ⬜ | ⬜ | ⬜ | ⬜ | — |
| LSTM / Embedding | Lstm / Gather | ⬜ | ⬜ | ⬜ | ⬜ | — |
| MHA / SwiGLU / Residual | compose | ⬜ | ⬜ | ⬜ | ⬜ | — |

### Host smoke result (this machine)

| Check | Result |
|-------|--------|
| Detect SDK + `QnnHtp` + `getProviders` | **PASS** (NPU software stack visible) |
| `open_htp` / `backendCreate` | **SKIP** `0xC00000FD` — **this is the NPU blocker** |
| `open_cpu` | SKIP same fault (ignore — not the goal) |
| `open_saver` + blank Dense/Softmax/RMSNorm/LayerNorm | **PASS** (graphs only, no Hexagon math) |
| Numeric NPU forward | **NOT YET** |

Until HTP create works: we have **detect + blank layer API + Saver graphs**. We do **not** yet have “NPU ran the op.”

---

## NPU can do it — wired in welvet.qnn (blank + graph)

These have a C++ QNN graph path in this repo.  
**Run dtype today (v0.1):** `float32` (HTP often executes as **FP16 math** once create works).

| Layer | QNN op(s) | HTP dtypes (SDK 2.42) | Notes |
|-------|-----------|------------------------|-------|
| **Dense** | FullyConnected, MatMul | bf16, fp16, int16, int8 | Primary path |
| **Softmax** | Softmax, LogSoftmax | bf16, fp16, int16, int8 | Standard / Temperature@1 only |
| **RMSNorm** | RmsNorm | fp16, int16, int8 | **No BF16** on HTP in 2.42 |
| **LayerNorm** | LayerNorm | bf16, fp16, int16, int8 | |

**Host dtypes we will schedule on HTP (SDK ceiling):**  
`float32` (via fp16), `float16`, `bfloat16` (Windows **V81+**), `int8`, `uint8`, `int16`, `uint16`.

**Not on HTP:** `float64`, FP8/FP6/FP4, sub-byte ints, nf4, ternary/binary, complex, native int/uint/uintptr, etc.

---

## NPU can do it — not implemented in welvet.qnn yet

SDK has a direct HTP op (or a clear compose path), but this library has **no** blank/forward graph yet.

### Native QNN ops (direct)

| Layer | QNN op(s) | HTP dtypes | Gap |
|-------|-----------|------------|-----|
| CNN2 | Conv2d, DepthWiseConv2d | bf16, fp16, int16, int8 | No HTP forward yet |
| CNN3 | Conv3d | fp16, int16, int8 | No BF16; not wired |
| ConvT2 | TransposeConv2d | bf16, fp16, int16, int8 | Not wired |
| ConvT3 | TransposeConv3d | fp16, int16, int8 | No BF16; not wired |
| LSTM | Lstm | fp16, int8 | No BF16/INT16; not wired |
| Embedding | Gather | bf16, fp16, int16, int8, others | Indices often INT32; not wired |

### Compose (build from HTP primitives)

| Layer | Would use | Gap |
|-------|-----------|-----|
| MHA | MatMul, Softmax, Mul, Transpose, Reshape | No MultiHead op; MaskedSoftmax HTP=NO in 2.42 |
| SwiGLU | FC/MatMul, Mul, Sigmoid | SiLU ≈ x·σ(x); not wired as one graph |
| Sequential | Child graph | Only if every child is HTP-capable |
| Residual | ElementWiseAdd | Not wired |

### Extra dtypes on wired layers

NPU columns exist for **fp16 / bf16 / int8 / int16** on Dense/Softmax/norms, but the **run allowlist is still float32 only**. Quantized and native fp16/bf16 paths are not exposed yet.

---

## Layers the NPU cannot carry (as-is)

| Layer | Why |
|-------|-----|
| CNN1 | Conv1d **HTP=NO** all dtypes |
| ConvT1 | TransposeConv1d **HTP=NO** |
| RNN | No QNN RNN on HTP (only Gru/Lstm) |
| KMeans | No clustering op |
| Mamba | No SSM / selective-scan op in 2.42 |
| GDN / Metacognition | App-specific; no QNN equivalent |
| Parallel / Stack | Hosts — not single HTP leaves (children may map individually) |

---

## What else QNN/HTP can do (no first-class layer yet)

Non-exhaustive; see QAIRT OpDef / `SupportedOps.html`:

| Area | Examples (QNN) | welvet.qnn status |
|------|----------------|-------------------|
| Pooling | PoolAvg2d/3d, PoolMax2d/3d, L2 pool | Not modeled |
| Activations | Relu, Gelu, Tanh, PRelu, HardSwish, ElementWiseNeuron | Not standalone |
| Elementwise | Add/Mul/Sub/Div, unary | Compose pieces only |
| Reduce / Softmax variants | Reduce*, LogSoftmax, ArgMax | Softmax only |
| Normalization | BatchNorm, InstanceNorm, GroupNorm | Only LayerNorm / RMSNorm |
| Attention extras | multi-head pattern, RoPE-style | MHA compose later |
| RNN family | **Gru** (HTP), Lstm | Neither wired |
| Image / CV | Pad, Crop, Resize, ChannelShuffle, SpaceToDepth | Not modeled |
| Quant tooling | Encoding, per-channel scales, UFIXED graphs | No quant-deploy path |
| Backends | QnnCpu, QnnGpu, QnnSaver/Ir | **Product = HTP only**; others diagnose/bring-up |
| Graph packaging | Context binary / DLC / qnn-net-run | Online small graphs only |

---

## Status legend

| Tag | Meaning |
|-----|---------|
| **Wired (blank)** | C++ blank + QNN graph in this repo (`dense` / `softmax` / `rmsnorm` / `layernorm` × f32) |
| **NPU numeric** | Same graph executed on live HTP (needs `backendCreate`) |
| **Native (gap)** | HTP op exists; blank/forward not implemented yet |
| **Compose (gap)** | Doable from HTP ops; not mapped yet |
| **None** | Do not schedule on NPU |

---

## Build

Needs CMake ≥ 3.16 and a C++17 compiler (Clang, GCC, MSVC).  
Optional: QAIRT SDK (`QAIRT_SDK_ROOT` or `-DQAIRT_SDK_ROOT=`).

```bash
# Windows ARM64 (llvm-mingw example)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=aarch64-w64-mingw32-clang++ \
  -DQAIRT_SDK_ROOT="$env:QAIRT_SDK_ROOT"
cmake --build build -j
```

```bash
# Linux aarch64
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQAIRT_SDK_ROOT=$QAIRT_SDK_ROOT
cmake --build build -j
```

Without the SDK you still get a **stub** library: detect can see filesystem hints; session/layers return Unavailable.

| Target | Role |
|--------|------|
| `libwelvet_qnn` | Static library |
| `welvet_qnn_smoke` | Detect NPU + open HTP + blank layer smoke |
| `welvet_qnn_probe` | Child-process HTP `backendCreate` gate (Windows) |

---

## Run smoke

```powershell
$env:QAIRT_SDK_ROOT = "C:\Qualcomm\AIStack\QAIRT\qairt\<ver>"
$env:PATH = ".\build;$env:QAIRT_SDK_ROOT\lib\aarch64-windows-msvc;$env:PATH"
# Force NPU only (default preference is already htp):
$env:WELVET_QNN_BACKEND = "htp"
.\build\welvet_qnn_smoke.exe
```

| Variable | Meaning |
|----------|---------|
| `QAIRT_SDK_ROOT` / `QNN_SDK_ROOT` | SDK root |
| `WELVET_QNN_BACKEND` | Prefer `htp` (NPU). `saver` = graph-only. `cpu`/`gpu` = diagnose only |
| `WELVET_QNN_INPROC=1` | Skip subprocess probe (unsafe if create crashes) |

---

## API sketch

```cpp
#include "welvet_qnn/detect.hpp"
#include "welvet_qnn/session.hpp"
#include "welvet_qnn/layers.hpp"

auto d = welvet_qnn::detect();          // NPU stack visible?
welvet_qnn::Session sess;
welvet_qnn::Session::open(Backend::Htp, &sess);  // real NPU — currently blocked on this host
auto dense = welvet_qnn::Dense::blank(4, 4);
dense.forward(sess, 1, x, y);
```

---

## Layout

```
include/welvet_qnn/   public headers
src/                  detect, session, layers
tools/smoke/          end-to-end verifier
tools/probe/          backendCreate subprocess
```

---

## Next (to get real NPU, not Saver)

1. Unblock HTP `backendCreate` on this Snapdragon host (driver / QAIRT / stack size — same `0xC00000FD` hits official tools).
2. Re-run smoke with `WELVET_QNN_BACKEND=htp` until Dense/Softmax/RMSNorm/LayerNorm show **numeric PASS on HTP**.
3. Then expand checklist: fp16/bf16/int8 allowlist, then CNN2 / LSTM / compose (MHA, SwiGLU, Residual).
