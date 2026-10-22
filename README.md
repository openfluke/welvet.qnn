# Welvet × Qualcomm QNN (Hexagon NPU)

**QNN** = Qualcomm AI Engine Direct (API). **QAIRT** = the SDK that ships it.  
**HTP** = Hexagon Tensor Processor (`QnnHtp`) — the Snapdragon **NPU**.

Standalone **C++17** library (`welvet.qnn`). Product target = **NPU forward only**.  
Saver = graph-compose sandbox when HTP create is broken. CPU/GPU = diagnose only.

Authority: QAIRT **2.42** `SupportedOps.html` HTP columns.  
Pass: **forward / inference only** — no backprop on NPU.

| | |
|--|--|
| **Library** | **0.2.0** |
| **SDK verified** | QAIRT **2.42.0.251225** · QNN API **2.32** · HTP id **6** |
| **Host** | Windows ARM64 · Snapdragon X · Hexagon **V73** |
| **Smoke** | `pass=42 fail=0 skip=2` — **NPU execute = YES** (`open_htp` PASS, `numeric=1`) |

---

## NPU status (this host)

| Fact | Detail |
|------|--------|
| Memory Integrity (HVCI) | **OFF** (required — was the `0xC00000FD` blocker) |
| `open_htp` / live Hexagon execute | ✅ |
| Skel / traditional DSP path | warns `1002` then **HNRD user-driver** — still runs |
| HTP APP tensors | `clientBuf` must be **nullptr at create**, bind at `graphExecute` |
| bf16 | ❌ V73 rejects (`V81+` only) — skipped |
| LSTM on HTP | ⬜ needs full 14-tensor QNN pack (Saver-only blank for now) |
| Native `RmsNorm` on V73/HNRD | disabled in driver → **falls back to LayerNorm(beta=0)** on HTP |

---

## Implementation checklist (welvet.qnn **0.2.0**)

### Infra

| Item | Ver | Status |
|------|-----|--------|
| Portable CMake C++17 | 0.1.0 | ✅ |
| Dynamic load QnnHtp (+ diagnose backends) | 0.1.0 | ✅ |
| Detect NPU stack (`getProviders`, buildId) | 0.1.0 | ✅ on host |
| Auto `ADSP_LIBRARY_PATH` (hexagon-v73… unsigned skels) | 0.2.0 | ✅ |
| Child-process + large-stack `backendCreate` probe | 0.2.0 | ✅ (still SKIP here) |
| Live HTP `backendCreate` / execute | — | ⬜ **host blocked** |
| DType pack/unpack f32↔f16/bf16/i8/u8/i16/u16 | 0.2.0 | ✅ |
| HTP caps matrix API (`htp_supports`) | 0.2.0 | ✅ |

### Layers × HTP dtypes (SDK 2.42)

Legend: **api** = blank+forward in this tree · **—** = HTP=NO · **NPU** = numeric on live HTP (needs create)

| Layer | QNN | f32 | f16 | bf16 | i8 | u8 | i16 | u16 | Saver graph | NPU numeric |
|-------|-----|-----|-----|------|----|----|-----|-----|-------------|-------------|
| Dense | FullyConnected | api | api | —¹ | api | api | api | api | ✅ | ✅ |
| Softmax | Softmax | api | api | —¹ | api | api | api | api | ✅ | ✅ |
| RMSNorm | RmsNorm→LN² | api | api | — | api | api | api | api | ✅ | ✅² |
| LayerNorm | LayerNorm | api | api | —¹ | api | api | api | api | ✅ | ✅ |
| CNN2 | Conv2d | api | api | —¹ | api | api | api | api | ✅ | ✅ |
| CNN3 | Conv3d | api | api | — | api | api | api | api | ✅ | ✅ |
| ConvT2 | TransposeConv2d | api | api | —¹ | api | api | api | api | ✅ | ✅ |
| ConvT3 | TransposeConv3d | api | api | — | api | api | api | api | ✅ | ✅ |
| LSTM | Lstm | api | api | — | api | api | — | — | ✅ | ⬜ pack |
| Embedding | Gather | api | api | —¹ | api | api | api | api | ✅ | ✅ |
| Residual | ElementWiseAdd | api | api | —¹ | api | api | api | api | ✅ | ✅ |
| SwiGLU | FC+Sigmoid+Mul | api | api | —¹ | api | api | api | api | ✅ | ✅ |
| MHA | FC+Softmax+Mul | api | api | —¹ | api | api | api | api | ✅ | ✅ |

¹ bf16 = V81+ only (V73 skip). ² HTP uses LayerNorm(β=0); native RmsNorm disabled on HNRD.

Host API remains **float32**; `DType` selects the QNN tensor datatype (HTP often runs fp16 math once create works).

### Not on HTP (do not schedule)

| Layer | Why |
|-------|-----|
| CNN1 / ConvT1 | Conv1d / TransposeConv1d **HTP=NO** |
| RNN | no QNN RNN (only Gru/Lstm) |
| KMeans / Mamba / GDN | no op |
| Parallel / Stack | hosts, not leaves |

---

## NPU can do it — notes (keep)

### Native vs compose

| Kind | Layers |
|------|--------|
| Native QNN | Dense, Softmax, RMSNorm, LayerNorm, CNN2/3, ConvT2/3, LSTM, Embedding |
| Compose | Residual, SwiGLU, MHA (no MultiHead op; MaskedSoftmax HTP=NO in 2.42) |

### Extra

Pooling, standalone activations, BatchNorm/InstanceNorm/GroupNorm, Gru, CV pad/resize, quant-deploy / DLC import — SDK can; not first-class layers here yet.

---

## Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=aarch64-w64-mingw32-clang++ \
  -DQAIRT_SDK_ROOT="$env:QAIRT_SDK_ROOT"
cmake --build build -j
```

| Target | Role |
|--------|------|
| `libwelvet_qnn` | Static library |
| `welvet_qnn_smoke` | Detect + **HTP first** + all layers/dtypes (Saver fallback labeled NOT NPU) |
| `welvet_qnn_probe` | Child-process HTP create gate (64 MiB stack thread) |

---

## Run smoke

```powershell
$env:QAIRT_SDK_ROOT = "C:\Qualcomm\AIStack\QAIRT\qairt\2.42.0.251225"
$env:PATH = ".\build;$env:QAIRT_SDK_ROOT\lib\aarch64-windows-msvc;$env:PATH"
$env:WELVET_QNN_BACKEND = "htp"
.\build\welvet_qnn_smoke.exe
```

| Env | Meaning |
|-----|---------|
| `WELVET_QNN_BACKEND=htp` | Prefer NPU |
| `WELVET_QNN_INPROC=1` | Skip subprocess probe (unsafe) |
| `WELVET_QNN_ADSP_FORCE=1` | Overwrite existing `ADSP_LIBRARY_PATH` |

Success criteria for “NPU works”: `open_htp` **PASS** and layer lines without `(graph)` / with `numeric=1`.

---

## API sketch

```cpp
#include "welvet_qnn/detect.hpp"
#include "welvet_qnn/dtype.hpp"
#include "welvet_qnn/layers.hpp"
#include "welvet_qnn/session.hpp"

auto d = welvet_qnn::detect();
welvet_qnn::Session sess;
welvet_qnn::Session::open(welvet_qnn::BackendKind::Htp, &sess); // NPU

auto dense = welvet_qnn::Dense::blank(4, 4, welvet_qnn::DType::Float16);
dense.forward(sess, 1, x, y);

auto cnn = welvet_qnn::Cnn2::blank(3, 16);
auto mha = welvet_qnn::Mha::blank(64, 8);
```

---

## Layout

```
include/welvet_qnn/   detect, session, layers, dtype, status
src/                  detect, dtype, session, layers, graph_util
tools/smoke/          verifier (HTP first; Saver ≠ NPU)
tools/probe/          backendCreate subprocess
```

---

## Next

1. Full QNN LSTM 14-tensor pack on HTP.
2. True RmsNorm compose (ElementWise) — drop LN fallback.
3. Real MHA (Q·Kᵀ / reshape / transpose), not Softmax⊙V placeholder.
4. Quant encodings beyond scale=1 blank; context binary / DLC later.
