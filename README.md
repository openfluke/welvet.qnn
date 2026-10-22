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
| **Smoke (2026-10-23)** | `pass=23 fail=0 skip=1` — **NPU execute = NO** |

---

## STOP — NPU is not running on this host

```
SKIP  open_htp — backendCreate probe exit=0xC00000FD (STACK_OVERFLOW)
!! NPU NOT RUNNING
```

| Fact | Detail |
|------|--------|
| Detect `QnnHtp` + `getProviders` | ✅ works |
| Official `qnn-platform-validator --backend dsp` | ❌ same `0xC00000FD` |
| Large-stack create thread (64 MiB) | ❌ still overflows inside QNN |
| `ADSP_LIBRARY_PATH` → `hexagon-v73/unsigned` | set automatically; does not fix create |
| **Windows Memory Integrity (HVCI)** | **Enabled = 1** on this machine |
| Saver blank graphs (all layers × dtypes) | ✅ compose only — **not Hexagon** |

**To unblock real NPU:** turn off Windows Security → Device Security → Core Isolation → **Memory Integrity**, reboot, re-run smoke until `open_htp` **PASS** and layers show `numeric=1`.  
Even then some Snapdragon X Plus units still fault in Qualcomm stacks (GenieX #1471) — driver/QAIRT updates may be required.

Until `open_htp` PASSes: **no layer “works on NPU.”** Saver PASS ≠ NPU.

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
| Dense | FullyConnected | api | api | api | api | api | api | api | ✅ 0.2.0 | ⬜ |
| Softmax | Softmax | api | api | api | api | api | api | api | ✅ | ⬜ |
| RMSNorm | RmsNorm | api | api | — | api | api | api | api | ✅ | ⬜ |
| LayerNorm | LayerNorm | api | api | api | api | api | api | api | ✅ | ⬜ |
| CNN2 | Conv2d | api | api | api | api | api | api | api | ✅ 0.2.0 | ⬜ |
| CNN3 | Conv3d | api | api | — | api | api | api | api | ✅ 0.2.0 | ⬜ |
| ConvT2 | TransposeConv2d | api | api | api | api | api | api | api | ✅ 0.2.0 | ⬜ |
| ConvT3 | TransposeConv3d | api | api | — | api | api | api | api | ✅ 0.2.0 | ⬜ |
| LSTM | Lstm | api | api | — | api | api | — | — | ✅ 0.2.0 | ⬜ |
| Embedding | Gather | api | api | api | api | api | api | api | ✅ 0.2.0 | ⬜ |
| Residual | ElementWiseAdd | api | api | api | api | api | api | api | ✅ 0.2.0 | ⬜ |
| SwiGLU | FC+Sigmoid+Mul | api | api | api | api | api | api | api | ✅ 0.2.0 | ⬜ |
| MHA | FC+Softmax+Mul compose | api | api | api | api | api | api | api | ✅ 0.2.0 | ⬜ |

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

## Next (real NPU)

1. Disable Memory Integrity → reboot.
2. Smoke until `open_htp` PASS + Dense…MHA numeric on HTP.
3. Harden MHA (real Q·Kᵀ / reshape / transpose) and LSTM weight packing against HTP finalize.
4. Quant encodings beyond scale=1 blank; context binary / DLC later.
