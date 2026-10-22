#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "welvet_qnn/session.hpp"
#include "welvet_qnn/status.hpp"

namespace welvet_qnn {

// Blank (zero / identity-scale) float32 layers for smoke tests.
// Forward-only on the Session's QNN backend.

struct Dense {
  int32_t in = 0;
  int32_t out = 0;
  std::vector<float> weight; // [out, in] row-major, zeros = blank
  std::vector<float> bias;   // [out], zeros

  static Dense blank(int32_t in, int32_t out);
  Status forward(Session& sess, int32_t batch, const float* x, float* y) const;
};

struct Softmax {
  int32_t cols = 0; // last-dim size

  static Softmax blank(int32_t cols);
  Status forward(Session& sess, int32_t rows, const float* x, float* y) const;
};

struct RmsNorm {
  int32_t dim = 0;
  float eps = 1e-5f;
  std::vector<float> gamma; // ones = blank

  static RmsNorm blank(int32_t dim, float eps = 1e-5f);
  Status forward(Session& sess, int32_t batch, int32_t seq, const float* x, float* y) const;
};

struct LayerNorm {
  int32_t dim = 0;
  float eps = 1e-5f;
  std::vector<float> gamma; // ones
  std::vector<float> beta;  // zeros

  static LayerNorm blank(int32_t dim, float eps = 1e-5f);
  Status forward(Session& sess, int32_t batch, int32_t seq, const float* x, float* y) const;
};

// Layers this library aims to support (QNN HTP allowlist for Welvet.qnn).
inline const char* supported_layers[] = {"dense", "softmax", "rmsnorm", "layernorm", nullptr};

} // namespace welvet_qnn
