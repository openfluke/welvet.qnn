#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "welvet_qnn/dtype.hpp"
#include "welvet_qnn/session.hpp"
#include "welvet_qnn/status.hpp"

namespace welvet_qnn {

// Blank layers + forward graphs for every HTP-capable Welvet.qnn leaf/compose.
// Host API is float32; `dtype` selects the QNN tensor datatype (HTP may run fp16 math).

struct Dense {
  int32_t in = 0, out = 0;
  DType dtype = DType::Float32;
  std::vector<float> weight; // [out, in]
  std::vector<float> bias;   // [out]

  static Dense blank(int32_t in, int32_t out, DType dt = DType::Float32);
  Status forward(Session& sess, int32_t batch, const float* x, float* y) const;
};

struct Softmax {
  int32_t cols = 0;
  DType dtype = DType::Float32;
  static Softmax blank(int32_t cols, DType dt = DType::Float32);
  Status forward(Session& sess, int32_t rows, const float* x, float* y) const;
};

struct RmsNorm {
  int32_t dim = 0;
  float eps = 1e-5f;
  DType dtype = DType::Float32;
  std::vector<float> gamma;
  static RmsNorm blank(int32_t dim, float eps = 1e-5f, DType dt = DType::Float32);
  Status forward(Session& sess, int32_t batch, int32_t seq, const float* x, float* y) const;
};

struct LayerNorm {
  int32_t dim = 0;
  float eps = 1e-5f;
  DType dtype = DType::Float32;
  std::vector<float> gamma, beta;
  static LayerNorm blank(int32_t dim, float eps = 1e-5f, DType dt = DType::Float32);
  Status forward(Session& sess, int32_t batch, int32_t seq, const float* x, float* y) const;
};

// NHWC Conv2d — blank 1×1 identity-ish when cin==cout.
struct Cnn2 {
  int32_t n = 1, h = 1, w = 1, cin = 1, cout = 1, kh = 1, kw = 1;
  int32_t stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0;
  DType dtype = DType::Float32;
  std::vector<float> weight; // [cout, kh, kw, cin] QNN Conv2d layout
  std::vector<float> bias;
  static Cnn2 blank(int32_t cin, int32_t cout, int32_t h = 4, int32_t w = 4, DType dt = DType::Float32);
  Status forward(Session& sess, const float* x, float* y) const; // x[n,h,w,cin]
};

struct Cnn3 {
  int32_t n = 1, d = 1, h = 1, w = 1, cin = 1, cout = 1, kd = 1, kh = 1, kw = 1;
  DType dtype = DType::Float32;
  std::vector<float> weight; // [cout, kd, kh, kw, cin]
  std::vector<float> bias;
  static Cnn3 blank(int32_t cin, int32_t cout, int32_t d = 2, int32_t h = 2, int32_t w = 2,
                    DType dt = DType::Float32);
  Status forward(Session& sess, const float* x, float* y) const;
};

struct ConvT2 {
  int32_t n = 1, h = 1, w = 1, cin = 1, cout = 1, kh = 1, kw = 1;
  DType dtype = DType::Float32;
  std::vector<float> weight;
  std::vector<float> bias;
  static ConvT2 blank(int32_t cin, int32_t cout, int32_t h = 2, int32_t w = 2,
                      DType dt = DType::Float32);
  Status forward(Session& sess, const float* x, float* y) const;
};

struct ConvT3 {
  int32_t n = 1, d = 1, h = 1, w = 1, cin = 1, cout = 1, kd = 1, kh = 1, kw = 1;
  DType dtype = DType::Float32;
  std::vector<float> weight;
  std::vector<float> bias;
  static ConvT3 blank(int32_t cin, int32_t cout, int32_t d = 2, int32_t h = 2, int32_t w = 2,
                      DType dt = DType::Float32);
  Status forward(Session& sess, const float* x, float* y) const;
};

struct Lstm {
  int32_t input_size = 0, hidden_size = 0;
  DType dtype = DType::Float32;
  // Blank: zero weights (smoke = graph compose only).
  std::vector<float> w_in, w_rec, b;
  static Lstm blank(int32_t input_size, int32_t hidden_size, DType dt = DType::Float32);
  // x: [seq, batch, input], y: [seq, batch, hidden]
  Status forward(Session& sess, int32_t seq, int32_t batch, const float* x, float* y) const;
};

struct Embedding {
  int32_t vocab = 0, dim = 0;
  DType dtype = DType::Float32;
  std::vector<float> table; // [vocab, dim]
  static Embedding blank(int32_t vocab, int32_t dim, DType dt = DType::Float32);
  // ids: int32 [n], y: float [n, dim]
  Status forward(Session& sess, int32_t n, const int32_t* ids, float* y) const;
};

struct Residual {
  DType dtype = DType::Float32;
  static Residual blank(DType dt = DType::Float32);
  Status forward(Session& sess, int32_t n, const float* a, const float* b, float* y) const;
};

struct SwiGlu {
  int32_t in = 0, hidden = 0;
  DType dtype = DType::Float32;
  std::vector<float> w_gate, w_up, w_down;
  static SwiGlu blank(int32_t in, int32_t hidden, DType dt = DType::Float32);
  Status forward(Session& sess, int32_t batch, const float* x, float* y) const;
};

struct Mha {
  int32_t dim = 0, heads = 0;
  DType dtype = DType::Float32;
  std::vector<float> w_q, w_k, w_v, w_o;
  static Mha blank(int32_t dim, int32_t heads, DType dt = DType::Float32);
  // x/y: [batch, seq, dim]
  Status forward(Session& sess, int32_t batch, int32_t seq, const float* x, float* y) const;
};

// All layers this library implements (HTP-capable set).
inline const char* supported_layers[] = {
    "dense",     "softmax", "rmsnorm", "layernorm", "cnn2",    "cnn3", "convt2",
    "convt3",    "lstm",    "embedding", "residual", "swiglu", "mha",  nullptr};

} // namespace welvet_qnn
