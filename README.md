# welvet.qnn — Qualcomm Hexagon NPU (QNN HTP)

Standalone **C++17** library. Target = **NPU forward**.  
Public surface is **Welvet-shaped** (layer blanks + `DType`) + **C ABI** (`welvet_qnn_c.h`) for Welvet Go `nativeqnn` — not an ONNX importer.

| | |
|--|--|
| **Library** | **1.1.0** |
| **SDK** | QAIRT **2.42.0.251225** · QNN API **2.32** |
| **Hosts** | Windows ARM64 · Snapdragon X · Hexagon **V73** · HNRD · **Android arm64-v8a** |
| **Go** | `nativeqnn` · `BackendQNN` · user `SyncToQNN` / `SyncFromQNN` |
| **Bench** | `w2a` `go_v_qnn` large matrix — NPU wins dense/cnn/mha/swiglu/… vs Go/native |

**Requires (Windows):** Memory Integrity (HVCI) **OFF** + reboot, or `backendCreate` dies with `0xC00000FD`.

See [VERSIONING.md](VERSIONING.md) for 1.x policy and sibling vendor repos.

---

## HTP layer × dtype (this V73 host)

Legend: ✅ live NPU · — skipped (caps) · bf16 = V81+ only

| Layer | f32 | f16 | i8/u8 | i16/u16 | bf16 | Notes |
|-------|-----|-----|-------|---------|------|-------|
| Dense | ✅ | ✅ | ✅ | ✅ | — | Prepare-once + warmed exec |
| Softmax | ✅ | ✅ | ✅ | ✅ | — | Online compose |
| RMSNorm | ✅ | ✅ | ✅ | ✅ | — | ElementWise compose (native RmsNorm disabled on HNRD) |
| LayerNorm | ✅ | ✅ | ✅ | ✅ | — | |
| CNN2 | ✅ | ✅ | ✅ | ✅ | — | |
| CNN3 | ✅ | ✅ | ✅ | — | — | i16 finalize fails on V73 |
| ConvT2 | ✅ | ✅ | ✅ | ✅ | — | |
| ConvT3 | ✅ | ✅ | ✅ | — | — | i16 finalize fails on V73 |
| LSTM | ✅* | ✅* | — | — | — | *Smoke OK; larger matrix shapes can `execute` fail |
| Embedding | ✅ | ✅ | ✅ | ✅ | — | |
| Residual | ✅ | ✅ | ✅ | ✅ | — | |
| SwiGLU | ✅ | ✅ | ✅ | ✅ | — | |
| MHA | ✅ | ✅ | ✅ | ✅ | — | Dense Q/K/V/O + MatMul attention |

Skips are intentional caps in `htp_supports()` — not silent fails.

**1.0.0 known bits:** LSTM flake at large shapes; online layers rebuild graphs each Go→NPU call (DLL `graphExecute` is still the headline); blank scale=1 quants; fused Sequential graph = later.

---

## Sequential networks (Welvet Go style)

```cpp
#include "welvet_qnn/sequential.hpp"

welvet_qnn::Sequential net;
net.push(welvet_qnn::Dense::blank(4, 8));
net.push(welvet_qnn::RmsNorm::blank(8));
net.push(welvet_qnn::Dense::blank(8, 4));
net.push(welvet_qnn::Softmax::blank(4));

net.forward(sess, /*batch=*/1, x, y, /*seq=*/1);
```

Today each child is its own HTP graph (host ping-pong). Fused single-graph Sequential = later.

---

## C ABI → Welvet Go

| Artifact | Location |
|----------|----------|
| Header | `include/welvet_qnn_c.h` (also `welvet/native/qnn/include/`) |
| Windows DLL | `libwelvet_qnn.dll` → `welvet/native/qnn/windows_arm64/` |
| Android `.so` | `libwelvet_qnn.so` → `welvet/native/qnn/android_arm64/` |
| Go | `github.com/openfluke/welvet/nativeqnn` |

Caps: `wqnn_layer_supported` / `wqnn_supported_layers` — Go rejects unsupported mounts on `SyncToQNN`.

```text
SyncToQNN → BackendQNN Forward → SyncFromQNN (optional)
```

Timing: **NPU DLL exec** = `mono_ns` around `graphExecute` inside this library. **Go→NPU call** = full cgo + pack/unpack (+ graph create on online layers).

---

## Not on HTP (as-is)

CNN1, ConvT1, RNN, KMeans, Mamba, GDN — no HTP op / Welvet host-only.

---

## Build / smoke (Windows ARM64)

```powershell
$env:QAIRT_SDK_ROOT = "C:\Qualcomm\AIStack\QAIRT\qairt\2.42.0.251225"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=aarch64-w64-mingw32-clang++ `
  -DQAIRT_SDK_ROOT="$env:QAIRT_SDK_ROOT"
cmake --build build -j
$env:PATH = ".\build;$env:QAIRT_SDK_ROOT\lib\aarch64-windows-msvc;$env:PATH"
.\build\welvet_qnn_smoke.exe
```

## Build / install (Android arm64)

```bash
export ANDROID_NDK="$HOME/Android/Sdk/ndk/28.2.13676358"   # or any NDK 26+
# optional for full HTP (not stub):
# export QAIRT_SDK_ROOT=/path/to/qairt/...
./scripts/android_install_go.sh   # → ../welvet/native/qnn/android_arm64/libwelvet_qnn.so
```

## Release

```bash
./release.sh --plan                 # version + actions
./release.sh --dry-run              # notes only (no git/publish)
./release.sh                        # source tag + GitHub Release
./release.sh --package-android      # also attach android-arm64-v8a .so tarball
```

CMake / README `**Library**` / `wqnn_version()` must agree. Tags are immutable.

| Target | Role |
|--------|------|
| `libwelvet_qnn` | Shared lib + C ABI |
| `welvet_qnn_smoke` | Full layer×dtype on HTP (host builds) |
| `welvet_qnn_probe` | Child-process `backendCreate` gate (Windows) |

---

## Next (1.x)

1. LSTM large-shape execute harden
2. Prepare-once for norms/softmax/cnn (cut Go→NPU call tax)
3. Context binary / compiled graph cache
4. Real quant encodings
5. Linux ARM64 HTP
