#include "welvet_qnn/layers.hpp"

#include <cmath>

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
#include "QnnOpDef.h"
#include "graph_util.hpp"
#endif

namespace welvet_qnn {

Dense Dense::blank(int32_t in, int32_t out) {
  Dense d;
  d.in = in;
  d.out = out;
  d.weight.assign((size_t)out * (size_t)in, 0.f);
  d.bias.assign((size_t)out, 0.f);
  // Identity-ish smoke: put 1 on diagonal so output isn't all zeros when in==out
  const int n = in < out ? in : out;
  for (int i = 0; i < n; ++i) d.weight[(size_t)i * (size_t)in + (size_t)i] = 1.f;
  return d;
}

Softmax Softmax::blank(int32_t cols) {
  Softmax s;
  s.cols = cols;
  return s;
}

RmsNorm RmsNorm::blank(int32_t dim, float eps) {
  RmsNorm r;
  r.dim = dim;
  r.eps = eps;
  r.gamma.assign((size_t)dim, 1.f);
  return r;
}

LayerNorm LayerNorm::blank(int32_t dim, float eps) {
  LayerNorm l;
  l.dim = dim;
  l.eps = eps;
  l.gamma.assign((size_t)dim, 1.f);
  l.beta.assign((size_t)dim, 0.f);
  return l;
}

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK

Status Dense::forward(Session& sess, int32_t batch, const float* x, float* y) const {
  if (!sess.ready() || !x || !y || batch <= 0 || in <= 0 || out <= 0) {
    return Status::Failed("dense: bad args / session");
  }
  auto st = graph::run_fc(sess, batch, in, out, weight.data(), x, y);
  if (!st) return st;
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t o = 0; o < out; ++o) {
      y[b * out + o] += bias[(size_t)o];
    }
  }
  return Status::Ok();
}

Status Softmax::forward(Session& sess, int32_t rows, const float* x, float* y) const {
  if (!sess.ready() || !x || !y || rows <= 0 || cols <= 0) {
    return Status::Failed("softmax: bad args / session");
  }
  uint32_t dims[2] = {(uint32_t)rows, (uint32_t)cols};
  const size_t n = (size_t)rows * (size_t)cols;
  return graph::run_unary(sess, "wq_softmax", QNN_OP_SOFTMAX, 2, dims, x, y, n, nullptr, 0,
                          nullptr, nullptr, 0, nullptr);
}

Status RmsNorm::forward(Session& sess, int32_t batch, int32_t seq, const float* x, float* y) const {
  if (!sess.ready() || !x || !y || batch <= 0 || seq <= 0 || dim <= 0) {
    return Status::Failed("rmsnorm: bad args / session");
  }
  uint32_t dims[3] = {(uint32_t)batch, (uint32_t)seq, (uint32_t)dim};
  const size_t n = (size_t)batch * (size_t)seq * (size_t)dim;
  return graph::run_unary(sess, "wq_rmsnorm", QNN_OP_RMS_NORM, 3, dims, x, y, n, gamma.data(),
                          gamma.size(), "gamma", nullptr, 0, nullptr);
}

Status LayerNorm::forward(Session& sess, int32_t batch, int32_t seq, const float* x,
                          float* y) const {
  if (!sess.ready() || !x || !y || batch <= 0 || seq <= 0 || dim <= 0) {
    return Status::Failed("layernorm: bad args / session");
  }
  uint32_t dims[3] = {(uint32_t)batch, (uint32_t)seq, (uint32_t)dim};
  const size_t n = (size_t)batch * (size_t)seq * (size_t)dim;
  return graph::run_unary(sess, "wq_layernorm", QNN_OP_LAYER_NORM, 3, dims, x, y, n, gamma.data(),
                          gamma.size(), "gamma", beta.data(), beta.size(), "beta");
}

#else

Status Dense::forward(Session&, int32_t, const float*, float*) const {
  return Status::Unavailable("built without QAIRT SDK");
}
Status Softmax::forward(Session&, int32_t, const float*, float*) const {
  return Status::Unavailable("built without QAIRT SDK");
}
Status RmsNorm::forward(Session&, int32_t, int32_t, const float*, float*) const {
  return Status::Unavailable("built without QAIRT SDK");
}
Status LayerNorm::forward(Session&, int32_t, int32_t, const float*, float*) const {
  return Status::Unavailable("built without QAIRT SDK");
}

#endif

} // namespace welvet_qnn
