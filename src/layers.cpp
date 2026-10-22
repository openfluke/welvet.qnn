#include "welvet_qnn/layers.hpp"

#include <cmath>
#include <cstring>

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
#include "QnnOpDef.h"
#include "graph_util.hpp"
#endif

namespace welvet_qnn {

Dense Dense::blank(int32_t in, int32_t out, DType dt) {
  Dense d;
  d.in = in;
  d.out = out;
  d.dtype = dt;
  d.weight.assign((size_t)out * (size_t)in, 0.f);
  d.bias.assign((size_t)out, 0.f);
  const int n = in < out ? in : out;
  for (int i = 0; i < n; ++i) d.weight[(size_t)i * (size_t)in + (size_t)i] = 1.f;
  return d;
}

Softmax Softmax::blank(int32_t cols, DType dt) {
  Softmax s;
  s.cols = cols;
  s.dtype = dt;
  return s;
}

RmsNorm RmsNorm::blank(int32_t dim, float eps, DType dt) {
  RmsNorm r;
  r.dim = dim;
  r.eps = eps;
  r.dtype = dt;
  r.gamma.assign((size_t)dim, 1.f);
  return r;
}

LayerNorm LayerNorm::blank(int32_t dim, float eps, DType dt) {
  LayerNorm l;
  l.dim = dim;
  l.eps = eps;
  l.dtype = dt;
  l.gamma.assign((size_t)dim, 1.f);
  l.beta.assign((size_t)dim, 0.f);
  return l;
}

Cnn2 Cnn2::blank(int32_t cin, int32_t cout, int32_t h, int32_t w, DType dt) {
  Cnn2 c;
  c.cin = cin;
  c.cout = cout;
  c.h = h;
  c.w = w;
  c.dtype = dt;
  c.weight.assign((size_t)cout * 1 * 1 * (size_t)cin, 0.f);
  c.bias.assign((size_t)cout, 0.f);
  const int n = cin < cout ? cin : cout;
  for (int i = 0; i < n; ++i) c.weight[(size_t)i * (size_t)cin + (size_t)i] = 1.f;
  return c;
}

Cnn3 Cnn3::blank(int32_t cin, int32_t cout, int32_t d, int32_t h, int32_t w, DType dt) {
  Cnn3 c;
  c.cin = cin;
  c.cout = cout;
  c.d = d;
  c.h = h;
  c.w = w;
  c.dtype = dt;
  c.weight.assign((size_t)cout * 1 * 1 * 1 * (size_t)cin, 0.f);
  c.bias.assign((size_t)cout, 0.f);
  const int n = cin < cout ? cin : cout;
  for (int i = 0; i < n; ++i) c.weight[(size_t)i * (size_t)cin + (size_t)i] = 1.f;
  return c;
}

ConvT2 ConvT2::blank(int32_t cin, int32_t cout, int32_t h, int32_t w, DType dt) {
  ConvT2 c;
  c.cin = cin;
  c.cout = cout;
  c.h = h;
  c.w = w;
  c.dtype = dt;
  c.weight.assign((size_t)cout * 1 * 1 * (size_t)cin, 0.f);
  c.bias.assign((size_t)cout, 0.f);
  const int n = cin < cout ? cin : cout;
  for (int i = 0; i < n; ++i) c.weight[(size_t)i * (size_t)cin + (size_t)i] = 1.f;
  return c;
}

ConvT3 ConvT3::blank(int32_t cin, int32_t cout, int32_t d, int32_t h, int32_t w, DType dt) {
  ConvT3 c;
  c.cin = cin;
  c.cout = cout;
  c.d = d;
  c.h = h;
  c.w = w;
  c.dtype = dt;
  c.weight.assign((size_t)cout * 1 * 1 * 1 * (size_t)cin, 0.f);
  c.bias.assign((size_t)cout, 0.f);
  const int n = cin < cout ? cin : cout;
  for (int i = 0; i < n; ++i) c.weight[(size_t)i * (size_t)cin + (size_t)i] = 1.f;
  return c;
}

Lstm Lstm::blank(int32_t input_size, int32_t hidden_size, DType dt) {
  Lstm l;
  l.input_size = input_size;
  l.hidden_size = hidden_size;
  l.dtype = dt;
  l.w_in.assign((size_t)4 * hidden_size * input_size, 0.f);
  l.w_rec.assign((size_t)4 * hidden_size * hidden_size, 0.f);
  l.b.assign((size_t)4 * hidden_size, 0.f);
  return l;
}

Embedding Embedding::blank(int32_t vocab, int32_t dim, DType dt) {
  Embedding e;
  e.vocab = vocab;
  e.dim = dim;
  e.dtype = dt;
  e.table.assign((size_t)vocab * dim, 0.f);
  for (int i = 0; i < vocab && i < dim; ++i) e.table[(size_t)i * dim + (size_t)i] = 1.f;
  return e;
}

Residual Residual::blank(DType dt) {
  Residual r;
  r.dtype = dt;
  return r;
}

SwiGlu SwiGlu::blank(int32_t in, int32_t hidden, DType dt) {
  SwiGlu s;
  s.in = in;
  s.hidden = hidden;
  s.dtype = dt;
  s.w_gate.assign((size_t)hidden * in, 0.f);
  s.w_up.assign((size_t)hidden * in, 0.f);
  s.w_down.assign((size_t)in * hidden, 0.f);
  const int n = in < hidden ? in : hidden;
  for (int i = 0; i < n; ++i) {
    s.w_gate[(size_t)i * in + i] = 1.f;
    s.w_up[(size_t)i * in + i] = 1.f;
    s.w_down[(size_t)i * hidden + i] = 1.f;
  }
  return s;
}

Mha Mha::blank(int32_t dim, int32_t heads, DType dt) {
  Mha m;
  m.dim = dim;
  m.heads = heads;
  m.dtype = dt;
  m.w_q.assign((size_t)dim * dim, 0.f);
  m.w_k.assign((size_t)dim * dim, 0.f);
  m.w_v.assign((size_t)dim * dim, 0.f);
  m.w_o.assign((size_t)dim * dim, 0.f);
  for (int i = 0; i < dim; ++i) {
    m.w_q[(size_t)i * dim + i] = 1.f;
    m.w_k[(size_t)i * dim + i] = 1.f;
    m.w_v[(size_t)i * dim + i] = 1.f;
    m.w_o[(size_t)i * dim + i] = 1.f;
  }
  return m;
}

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK

Status Dense::forward(Session& sess, int32_t batch, const float* x, float* y) const {
  if (!sess.ready() || !x || !y || batch <= 0 || in <= 0 || out <= 0)
    return Status::Failed("dense: bad args / session");
  if (!htp_supports(LayerId::Dense, dtype) && sess.backend() == BackendKind::Htp)
    return Status::Unsupported(std::string("dense dtype ") + dtype_name(dtype));
  auto st = graph::run_fc(sess, batch, in, out, weight.data(), x, y, dtype);
  if (!st) return st;
  for (int32_t b = 0; b < batch; ++b)
    for (int32_t o = 0; o < out; ++o) y[b * out + o] += bias[(size_t)o];
  return Status::Ok();
}

Status Softmax::forward(Session& sess, int32_t rows, const float* x, float* y) const {
  if (!sess.ready() || !x || !y || rows <= 0 || cols <= 0)
    return Status::Failed("softmax: bad args / session");
  uint32_t dims[2] = {(uint32_t)rows, (uint32_t)cols};
  return graph::run_unary(sess, "wq_softmax", QNN_OP_SOFTMAX, 2, dims, x, y,
                          (size_t)rows * cols, dtype, nullptr, 0, nullptr, nullptr, 0, nullptr);
}

Status RmsNorm::forward(Session& sess, int32_t batch, int32_t seq, const float* x,
                        float* y) const {
  if (!sess.ready() || !x || !y || batch <= 0 || seq <= 0 || dim <= 0)
    return Status::Failed("rmsnorm: bad args / session");
  if (!htp_supports(LayerId::RmsNorm, dtype) && sess.backend() == BackendKind::Htp)
    return Status::Unsupported("rmsnorm: dtype not on HTP");
  // Snapdragon X (V73) + HNRD path: native q::QNN_RmsNorm is registered as *disabled*
  // ("no properties registered for q::QNN_RmsNorm"). Use LayerNorm-equivalent blank
  // (beta=0) on HTP until an ElementWise compose lands.
  if (sess.backend() == BackendKind::Htp) {
    LayerNorm ln;
    ln.dim = dim;
    ln.eps = eps;
    ln.dtype = dtype;
    ln.gamma = gamma;
    ln.beta.assign((size_t)dim, 0.f);
    return ln.forward(sess, batch, seq, x, y);
  }
  const int32_t tokens = batch * seq;
  uint32_t dims[2] = {(uint32_t)tokens, (uint32_t)dim};
  return graph::run_unary(sess, "wq_rmsnorm", QNN_OP_RMS_NORM, 2, dims, x, y,
                          (size_t)tokens * dim, dtype, gamma.data(), gamma.size(), "gamma",
                          nullptr, 0, nullptr, eps, /*with_axes=*/true);
}

Status LayerNorm::forward(Session& sess, int32_t batch, int32_t seq, const float* x,
                          float* y) const {
  if (!sess.ready() || !x || !y || batch <= 0 || seq <= 0 || dim <= 0)
    return Status::Failed("layernorm: bad args / session");
  uint32_t dims[3] = {(uint32_t)batch, (uint32_t)seq, (uint32_t)dim};
  return graph::run_unary(sess, "wq_layernorm", QNN_OP_LAYER_NORM, 3, dims, x, y,
                          (size_t)batch * seq * dim, dtype, gamma.data(), gamma.size(), "gamma",
                          beta.data(), beta.size(), "beta", eps, /*with_axes=*/true);
}

Status Cnn2::forward(Session& sess, const float* x, float* y) const {
  if (!sess.ready() || !x || !y) return Status::Failed("cnn2: bad args");
  return graph::run_conv2d(sess, n, h, w, cin, cout, kh, kw, stride_h, stride_w, pad_h, pad_w,
                           weight.data(), x, y, dtype, QNN_OP_CONV_2D);
}

Status Cnn3::forward(Session& sess, const float* x, float* y) const {
  if (!sess.ready() || !x || !y) return Status::Failed("cnn3: bad args");
  return graph::run_conv3d(sess, n, d, h, w, cin, cout, kd, kh, kw, weight.data(), x, y, dtype,
                           QNN_OP_CONV_3D);
}

Status ConvT2::forward(Session& sess, const float* x, float* y) const {
  if (!sess.ready() || !x || !y) return Status::Failed("convt2: bad args");
  // TransposeConv2d weight layout often [cin, kh, kw, cout]; blank uses same buffer — graph compose.
  return graph::run_conv2d(sess, n, h, w, cin, cout, kh, kw, 1, 1, 0, 0, weight.data(), x, y, dtype,
                           QNN_OP_TRANSPOSE_CONV_2D);
}

Status ConvT3::forward(Session& sess, const float* x, float* y) const {
  if (!sess.ready() || !x || !y) return Status::Failed("convt3: bad args");
  return graph::run_conv3d(sess, n, d, h, w, cin, cout, kd, kh, kw, weight.data(), x, y, dtype,
                           QNN_OP_TRANSPOSE_CONV_3D);
}

Status Lstm::forward(Session& sess, int32_t seq, int32_t batch, const float* x, float* y) const {
  if (!sess.ready() || !x || !y || seq <= 0 || batch <= 0)
    return Status::Failed("lstm: bad args");
  if (!htp_supports(LayerId::Lstm, dtype) && sess.backend() == BackendKind::Htp)
    return Status::Unsupported("lstm dtype");
  // HTP Lstm expects the full multi-input weight pack (~14 tensors). Saver accepts a
  // simplified 4-input form for graph compose; live HTP needs the full layout (next).
  if (sess.backend() == BackendKind::Htp) {
    return Status::Unsupported(
        "lstm: HTP needs full QNN Lstm input pack (14 tensors) — blank 4-input form is Saver-only");
  }
  return graph::run_lstm_native(sess, seq, batch, input_size, hidden_size, w_in.data(),
                                w_rec.data(), b.data(), x, y, dtype);
}

Status Embedding::forward(Session& sess, int32_t n, const int32_t* ids, float* y) const {
  if (!sess.ready() || !ids || !y || n <= 0) return Status::Failed("embedding: bad args");
  return graph::run_gather(sess, vocab, dim, table.data(), n, ids, y, dtype);
}

Status Residual::forward(Session& sess, int32_t n, const float* a, const float* b,
                         float* y) const {
  if (!sess.ready() || !a || !b || !y || n <= 0) return Status::Failed("residual: bad args");
  return graph::run_binary(sess, "wq_residual", QNN_OP_ELEMENT_WISE_ADD, (size_t)n, a, b, y, dtype);
}

Status SwiGlu::forward(Session& sess, int32_t batch, const float* x, float* y) const {
  if (!sess.ready() || !x || !y || batch <= 0) return Status::Failed("swiglu: bad args");
  // gate = σ(x @ Wg), up = x @ Wu, h = gate ⊙ up, y = h @ Wd
  std::vector<float> gate((size_t)batch * hidden), up((size_t)batch * hidden),
      h((size_t)batch * hidden), sig((size_t)batch * hidden);
  auto st = graph::run_fc(sess, batch, in, hidden, w_gate.data(), x, gate.data(), dtype);
  if (!st) return st;
  uint32_t dims[2] = {(uint32_t)batch, (uint32_t)hidden};
  st = graph::run_unary(sess, "wq_swiglu_sig", QNN_OP_SIGMOID, 2, dims, gate.data(), sig.data(),
                        gate.size(), dtype, nullptr, 0, nullptr, nullptr, 0, nullptr);
  if (!st) return st;
  st = graph::run_fc(sess, batch, in, hidden, w_up.data(), x, up.data(), dtype);
  if (!st) return st;
  st = graph::run_binary(sess, "wq_swiglu_mul", QNN_OP_ELEMENT_WISE_MULTIPLY, gate.size(),
                         sig.data(), up.data(), h.data(), dtype);
  if (!st) return st;
  return graph::run_fc(sess, batch, hidden, in, w_down.data(), h.data(), y, dtype);
}

Status Mha::forward(Session& sess, int32_t batch, int32_t seq, const float* x, float* y) const {
  if (!sess.ready() || !x || !y || batch <= 0 || seq <= 0 || dim <= 0 || heads <= 0)
    return Status::Failed("mha: bad args");
  if (dim % heads != 0) return Status::Failed("mha: dim % heads != 0");
  // Compose: Q/K/V projections + scaled-dot via MatMul-less blank path:
  // For blank smoke we project Q,K,V then residual-add Q+V (attention body = later).
  // Full attention score Softmax needs reshape/transpose; blank verifies FC+Softmax+Add compose.
  const int32_t tokens = batch * seq;
  std::vector<float> q((size_t)tokens * dim), k((size_t)tokens * dim), v((size_t)tokens * dim),
      attn((size_t)tokens * dim), out((size_t)tokens * dim);
  auto st = graph::run_fc(sess, tokens, dim, dim, w_q.data(), x, q.data(), dtype);
  if (!st) return st;
  st = graph::run_fc(sess, tokens, dim, dim, w_k.data(), x, k.data(), dtype);
  if (!st) return st;
  st = graph::run_fc(sess, tokens, dim, dim, w_v.data(), x, v.data(), dtype);
  if (!st) return st;
  // Softmax over last dim of Q (placeholder attention weights per token)
  uint32_t dims[2] = {(uint32_t)tokens, (uint32_t)dim};
  st = graph::run_unary(sess, "wq_mha_sm", QNN_OP_SOFTMAX, 2, dims, q.data(), attn.data(),
                        (size_t)tokens * dim, dtype, nullptr, 0, nullptr, nullptr, 0, nullptr);
  if (!st) return st;
  st = graph::run_binary(sess, "wq_mha_mul", QNN_OP_ELEMENT_WISE_MULTIPLY, (size_t)tokens * dim,
                         attn.data(), v.data(), out.data(), dtype);
  if (!st) return st;
  return graph::run_fc(sess, tokens, dim, dim, w_o.data(), out.data(), y, dtype);
}

#else

#define STUB_FWD(name) \
  Status name { return Status::Unavailable("built without QAIRT SDK"); }

STUB_FWD(Dense::forward(Session&, int32_t, const float*, float*) const)
STUB_FWD(Softmax::forward(Session&, int32_t, const float*, float*) const)
STUB_FWD(RmsNorm::forward(Session&, int32_t, int32_t, const float*, float*) const)
STUB_FWD(LayerNorm::forward(Session&, int32_t, int32_t, const float*, float*) const)
STUB_FWD(Cnn2::forward(Session&, const float*, float*) const)
STUB_FWD(Cnn3::forward(Session&, const float*, float*) const)
STUB_FWD(ConvT2::forward(Session&, const float*, float*) const)
STUB_FWD(ConvT3::forward(Session&, const float*, float*) const)
STUB_FWD(Lstm::forward(Session&, int32_t, int32_t, const float*, float*) const)
STUB_FWD(Embedding::forward(Session&, int32_t, const int32_t*, float*) const)
STUB_FWD(Residual::forward(Session&, int32_t, const float*, const float*, float*) const)
STUB_FWD(SwiGlu::forward(Session&, int32_t, const float*, float*) const)
STUB_FWD(Mha::forward(Session&, int32_t, int32_t, const float*, float*) const)

#endif

} // namespace welvet_qnn
