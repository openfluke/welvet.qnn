#include "welvet_qnn/dtype.hpp"

namespace welvet_qnn {

bool htp_supports(LayerId layer, DType dtype) {
  // QAIRT 2.42 HTP columns. bf16 needs Hexagon V81+ — Snapdragon X (V73) rejects it.
  const bool f16 = dtype == DType::Float16;
  const bool i8 = dtype == DType::Int8 || dtype == DType::UInt8;
  const bool i16 = dtype == DType::Int16 || dtype == DType::UInt16;
  const bool f32 = dtype == DType::Float32;
  if (dtype == DType::BFloat16) return false; // V81+ only

  switch (layer) {
    case LayerId::Dense:
    case LayerId::Softmax:
    case LayerId::LayerNorm:
    case LayerId::RmsNorm:
    case LayerId::Cnn2:
    case LayerId::ConvT2:
    case LayerId::Embedding:
    case LayerId::Residual:
    case LayerId::SwiGlu:
    case LayerId::Mha:
      return f32 || f16 || i8 || i16;
    case LayerId::Cnn3:
    case LayerId::ConvT3:
      // V73/HNRD: int16/uint16 Conv3d/TransposeConv3d finalize fails (prepare -1).
      return f32 || f16 || i8;
    case LayerId::Lstm:
      // Welvet gate-compose on HTP: f32/f16 solid; int8 execute flaky on HNRD.
      return f32 || f16;
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
