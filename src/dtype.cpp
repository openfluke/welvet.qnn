#include "welvet_qnn/dtype.hpp"

namespace welvet_qnn {

bool htp_supports(LayerId layer, DType dtype) {
  // From QAIRT 2.42 SupportedOps.html HTP columns (Welvet.qnn README matrix).
  const bool f16 = dtype == DType::Float16;
  const bool bf16 = dtype == DType::BFloat16;
  const bool i8 = dtype == DType::Int8 || dtype == DType::UInt8;
  const bool i16 = dtype == DType::Int16 || dtype == DType::UInt16;
  const bool f32 = dtype == DType::Float32; // host f32 usually runs as fp16 math on HTP

  auto fp_family = [&](bool allow_bf16) {
    if (f32 || f16 || i8 || i16) return true;
    if (bf16 && allow_bf16) return true;
    return false;
  };

  switch (layer) {
    case LayerId::Dense:
    case LayerId::Softmax:
    case LayerId::LayerNorm:
    case LayerId::Cnn2:
    case LayerId::ConvT2:
    case LayerId::Embedding:
    case LayerId::Residual:
    case LayerId::SwiGlu:
    case LayerId::Mha:
      return fp_family(/*allow_bf16=*/true);
    case LayerId::RmsNorm:
      // No BF16 on HTP in 2.42
      return f32 || f16 || i8 || i16;
    case LayerId::Cnn3:
    case LayerId::ConvT3:
      // No BF16
      return f32 || f16 || i8 || i16;
    case LayerId::Lstm:
      // fp16 + int8 only (no BF16 / INT16)
      return f32 || f16 || dtype == DType::Int8 || dtype == DType::UInt8;
    default:
      return false;
  }
}

bool layer_implemented(LayerId layer) {
  switch (layer) {
    case LayerId::Dense:
    case LayerId::Softmax:
    case LayerId::RmsNorm:
    case LayerId::LayerNorm:
    case LayerId::Cnn2:
    case LayerId::Cnn3:
    case LayerId::ConvT2:
    case LayerId::ConvT3:
    case LayerId::Lstm:
    case LayerId::Embedding:
    case LayerId::Residual:
    case LayerId::SwiGlu:
    case LayerId::Mha:
      return true;
    default:
      return false;
  }
}

} // namespace welvet_qnn
