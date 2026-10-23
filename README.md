# welvet.qnn — Qualcomm Hexagon NPU (QNN HTP)

Standalone **C++17** library. Target = **NPU forward**.  
Public surface is **Welvet-shaped** (layer blanks + `DType`) + **C ABI** (`welvet_qnn_c.h`) for Welvet Go `nativeqnn` — not an ONNX importer.

| | |
|--|--|
| **Library** | **0.2.1** |
| **SDK** | QAIRT **2.42.0.251225** · QNN API **2.32** |
| **Host** | Windows ARM64 · Snapdragon X · Hexagon **V73** · HNRD |
| **Smoke** | `pass=165 fail=0 skip=21` — **live HTP** |

**Requires:** Memory Integrity (HVCI) **OFF** + reboot, or `backendCreate` dies with `0xC00000FD`.

---

## HTP layer × dtype (this V73 host)

Legend: ✅ live NPU · — skipped (caps) · bf16 = V81+ only

| Layer | f32 | f16 | i8/u8 | i16/u16 | bf16 | Notes |
|-------|-----|-----|-------|---------|------|-------|
| Dense | ✅ | ✅ | ✅ | ✅ | — | |
| Softmax | ✅ | ✅ | ✅ | ✅ | — | |
| RMSNorm | ✅ | ✅ | ✅ | ✅ | — | ElementWise compose (native RmsNorm disabled on HNRD) |
| LayerNorm | ✅ | ✅ | ✅ | ✅ | — | |
| CNN2 | ✅ | ✅ | ✅ | ✅ | — | |
| CNN3 | ✅ | ✅ | ✅ | — | — | i16 finalize fails on V73 |
| ConvT2 | ✅ | ✅ | ✅ | ✅ | — | |
| ConvT3 | ✅ | ✅ | ✅ | — | — | i16 finalize fails on V73 |
| LSTM | ✅ | ✅ | — | — | — | **Welvet** gate pack `[i\|f\|g\|o]×(IH\|HH\|bias)` compose |
| Embedding | ✅ | ✅ | ✅ | ✅ | — | |
| Residual | ✅ | ✅ | ✅ | ✅ | — | |
| SwiGLU | ✅ | ✅ | ✅ | ✅ | — | |
| MHA | ✅ | ✅ | ✅ | ✅ | — | Dense Q/K/V/O + MatMul attention |

Skips are intentional caps in `htp_supports()` — not silent fails.

---

## Sequential networks (Welvet Go style)

Chain layers dynamically — output of A feeds B feeds C:

```cpp
#include "welvet_qnn/sequential.hpp"

welvet_qnn::Sequential net;
net.push(welvet_qnn::Dense::blank(4, 8));
net.push(welvet_qnn::RmsNorm::blank(8));
net.push(welvet_qnn::Dense::blank(8, 4));
net.push(welvet_qnn::Softmax::blank(4));

net.forward(sess, /*batch=*/1, x, y, /*seq=*/1);
// or: Sequential::dense_tower(dim, depth)  // like sequential.New
```

`push` rejects shape mismatches (`child in != prev out`).  
Today each child is still its own HTP graph (host ping-pong buffers). Fused single-graph Sequential = later (with C ABI).

Children: Dense, Softmax, RmsNorm, LayerNorm, SwiGlu, Mha, Lstm.

---

## C ABI → Welvet Go

| Artifact | Location |
|----------|----------|
| Header | `include/welvet_qnn_c.h` (also `welvet/native/qnn/include/`) |
| DLL | `libwelvet_qnn.dll` → `welvet/native/qnn/windows_arm64/` |
| Go | `github.com/openfluke/welvet/nativeqnn` |

Caps: `wqnn_layer_supported` / `wqnn_supported_layers` — Go rejects unsupported mounts on `SyncToQNN`.

User-owned sync in Welvet:

```text
SyncToQNN → BackendQNN Forward → SyncFromQNN (optional)
```

## Packing

| Layer | Host pack |
|-------|-----------|
| Dense | `W[out,in]`, `bias[out]`, host f32 API, `DType` selects QNN tensor type |
| LSTM | Welvet `[i\|f\|g\|o]` each with IH / HH / bias — **same idea as Welvet Go**, not ONNX W/R/B |
| MHA | Four Dense projections + MatMul scores |
| RMSNorm | `gamma[dim]`; HTP = Mul/ReduceMean/Add/Rsqrt compose |

Binary / context-cache compilation = later. This tree is online graphs today.

---

## Not on HTP (as-is)

CNN1, ConvT1, RNN, KMeans, Mamba, GDN — no HTP op / Welvet host-only.

---

## Build / smoke

```powershell
$env:QAIRT_SDK_ROOT = "C:\Qualcomm\AIStack\QAIRT\qairt\2.42.0.251225"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=aarch64-w64-mingw32-clang++ `
  -DQAIRT_SDK_ROOT="$env:QAIRT_SDK_ROOT"
cmake --build build -j
$env:PATH = ".\build;$env:QAIRT_SDK_ROOT\lib\aarch64-windows-msvc;$env:PATH"
.\build\welvet_qnn_smoke.exe
```

| Target | Role |
|--------|------|
| `libwelvet_qnn` | Static lib (C ABI wrapper later) |
| `welvet_qnn_smoke` | Full layer×dtype on HTP + Saver |
| `welvet_qnn_probe` | Child-process `backendCreate` gate |

---

## Next (after this)

1. C ABI (`welvet_qnn.h`) + Go FFI into Welvet.
2. Context binary / compiled graph cache.
3. MHA head-split reshape; LSTM int8 compose if HNRD allows.
4. Real quant encodings (not scale=1 blank).
