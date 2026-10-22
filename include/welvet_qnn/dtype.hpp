#pragma once

#include <cstddef>
#include <cstdint>

namespace welvet_qnn {

// Host / QNN tensor dtypes we schedule on HTP (SDK 2.42 ceiling).
enum class DType : uint8_t {
  Float32 = 0,
  Float16,
  BFloat16,
  Int8,
  UInt8,
  Int16,
  UInt16,
};

inline const char* dtype_name(DType d) {
  switch (d) {
    case DType::Float32:
      return "float32";
    case DType::Float16:
      return "float16";
    case DType::BFloat16:
      return "bfloat16";
    case DType::Int8:
      return "int8";
    case DType::UInt8:
      return "uint8";
    case DType::Int16:
      return "int16";
    case DType::UInt16:
      return "uint16";
    default:
      return "?";
  }
}

inline size_t dtype_nbytes(DType d) {
  switch (d) {
    case DType::Float32:
      return 4;
    case DType::Float16:
    case DType::BFloat16:
    case DType::Int16:
    case DType::UInt16:
      return 2;
    case DType::Int8:
    case DType::UInt8:
      return 1;
    default:
      return 0;
  }
}

// QNN_DATATYPE_* numeric values from QnnTypes.h (keep in sync with QAIRT 2.42).
inline uint32_t dtype_to_qnn(DType d) {
  switch (d) {
    case DType::Float32:
      return 0x0232; // QNN_DATATYPE_FLOAT_32
    case DType::Float16:
      return 0x0216; // QNN_DATATYPE_FLOAT_16
    case DType::BFloat16:
      return 0x0226; // QNN_DATATYPE_BFLOAT_16
    case DType::Int8:
      return 0x0008; // QNN_DATATYPE_INT_8
    case DType::Int16:
      return 0x0016; // QNN_DATATYPE_INT_16
    case DType::UInt8:
      return 0x0108; // QNN_DATATYPE_UINT_8
    case DType::UInt16:
      return 0x0116; // QNN_DATATYPE_UINT_16
    default:
      return 0;
  }
}

inline constexpr DType kAllDtypes[] = {
    DType::Float32, DType::Float16, DType::BFloat16, DType::Int8,
    DType::UInt8,   DType::Int16,   DType::UInt16,
};

// Layer ids for capability matrix.
enum class LayerId : uint8_t {
  Dense = 0,
  Softmax,
  RmsNorm,
  LayerNorm,
  Cnn2,
  Cnn3,
  ConvT2,
  ConvT3,
  Lstm,
  Embedding,
  Residual,
  SwiGlu,
  Mha,
  Count
};

inline const char* layer_id_name(LayerId id) {
  switch (id) {
    case LayerId::Dense:
      return "dense";
    case LayerId::Softmax:
      return "softmax";
    case LayerId::RmsNorm:
      return "rmsnorm";
    case LayerId::LayerNorm:
      return "layernorm";
    case LayerId::Cnn2:
      return "cnn2";
    case LayerId::Cnn3:
      return "cnn3";
    case LayerId::ConvT2:
      return "convt2";
    case LayerId::ConvT3:
      return "convt3";
    case LayerId::Lstm:
      return "lstm";
    case LayerId::Embedding:
      return "embedding";
    case LayerId::Residual:
      return "residual";
    case LayerId::SwiGlu:
      return "swiglu";
    case LayerId::Mha:
      return "mha";
    default:
      return "?";
  }
}

// HTP support for (layer, dtype) per QAIRT 2.42 SupportedOps HTP columns.
// Returns: 1 = native/compose OK on HTP, 0 = HTP=NO / not applicable.
bool htp_supports(LayerId layer, DType dtype);

// True if this library has a blank+forward graph for the layer (any dtype).
bool layer_implemented(LayerId layer);

} // namespace welvet_qnn
