# welvet.qnn versioning

| Tag | Meaning |
|-----|---------|
| **1.0.0** | First stable HTP ship — C ABI + Go `nativeqnn`, layer×dtype matrix on Snapdragon X / Hexagon V73 (Windows ARM64) |
| **1.x.y** | Qualcomm Hexagon / QNN HTP only (this repo). Patch = bugfix; minor = new layers/dtypes/OS |

## Known 1.0.0 gaps (honest)

- LSTM can fail execute at larger matrix shapes (small smoke ✅)
- Most layers are **online compose** (graph rebuild each call) — Dense is prepare-once
- CNN3 / ConvT3 skip i16/u16 on V73 (caps)
- bf16 = V81+ only
- Real quant encodings (not blank scale=1) = later
- Linux ARM64 HTP = later 1.x

## Sibling vendor repos (do not merge into this tree)

Vendor accel stays isolated to prevent SDK / symbol / license contamination:

| Repo (planned) | Vendor | OS targets |
|----------------|--------|------------|
| **welvet.qnn** (this) | Qualcomm Hexagon NPU | Windows ARM64 → Linux ARM64 |
| welvet.cuda / welvet.nvidia | NVIDIA GPU | Windows + Linux |
| welvet.rocm / welvet.amd | AMD GPU | Windows + Linux |
| welvet.xpu / welvet.intel_gpu | Intel GPU | Windows + Linux |
| welvet.intel_npu | Intel NPU | Windows + Linux |
| welvet.cpp | **CPU core + C ABI only** (0.2+) | all |

Go Welvet binds each via a thin `native*` package (`nativeqnn`, future `nativecuda`, …).
