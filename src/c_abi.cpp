#include "welvet_qnn_c.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "prepared.hpp"
#include "welvet_qnn/detect.hpp"
#include "welvet_qnn/dtype.hpp"
#include "welvet_qnn/layers.hpp"
#include "welvet_qnn/sequential.hpp"
#include "welvet_qnn/session.hpp"

using namespace welvet_qnn;

struct wqnn_session {
  Session sess;
#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
  DensePrepared dense_prep;
#endif
};

struct wqnn_sequential {
  Sequential net;
};

namespace {

wqnn_status ok() {
  wqnn_status s{};
  s.code = 0;
  s.message[0] = '\0';
  return s;
}

wqnn_status from(const Status& st) {
  wqnn_status s{};
  s.code = (int32_t)st.code;
  std::snprintf(s.message, sizeof(s.message), "%s", st.message.c_str());
  return s;
}

void copy_str(char* dst, int32_t n, const std::string& src) {
  if (!dst || n <= 0) return;
  std::snprintf(dst, (size_t)n, "%s", src.c_str());
}

DType parse_dtype(const char* d) {
  if (!d || !d[0]) return DType::Float32;
  if (std::strcmp(d, "float16") == 0 || std::strcmp(d, "f16") == 0) return DType::Float16;
  if (std::strcmp(d, "bfloat16") == 0 || std::strcmp(d, "bf16") == 0) return DType::BFloat16;
  if (std::strcmp(d, "int8") == 0 || std::strcmp(d, "i8") == 0) return DType::Int8;
  if (std::strcmp(d, "uint8") == 0 || std::strcmp(d, "u8") == 0) return DType::UInt8;
  if (std::strcmp(d, "int16") == 0 || std::strcmp(d, "i16") == 0) return DType::Int16;
  if (std::strcmp(d, "uint16") == 0 || std::strcmp(d, "u16") == 0) return DType::UInt16;
  return DType::Float32;
}

const char* dtype_cstr(DType dt) { return dtype_name(dt); }

wqnn_status need_layer(const char* layer, const char* dtype) {
  DType dt = parse_dtype(dtype);
  if (!wqnn_layer_supported(layer, dtype_cstr(dt)))
    return from(Status::Unsupported(std::string(layer) + "/" + dtype_cstr(dt)));
  return ok();
}

LayerId parse_layer(const char* name) {
  if (!name) return LayerId::Count;
  for (int i = 0; i < (int)LayerId::Count; ++i) {
    if (std::strcmp(name, layer_id_name((LayerId)i)) == 0) return (LayerId)i;
  }
  return LayerId::Count;
}

} // namespace

extern "C" {

const char* wqnn_version(void) { return "0.2.1"; }

int32_t wqnn_has_sdk(void) {
#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
  return 1;
#else
  return 0;
#endif
}

int32_t wqnn_layer_supported(const char* layer, const char* dtype) {
  auto id = parse_layer(layer);
  if (id == LayerId::Count || !layer_implemented(id)) return 0;
  return htp_supports(id, parse_dtype(dtype)) ? 1 : 0;
}

const char* const* wqnn_supported_layers(void) {
  return supported_layers;
}

wqnn_status wqnn_detect(int32_t* sdk_found, int32_t* htp_lib, int32_t* htp_prov, char* sdk_root,
                        int32_t sdk_root_n, char* build_id, int32_t build_id_n) {
  auto d = detect();
  if (sdk_found) *sdk_found = d.sdk_found ? 1 : 0;
  if (htp_lib) *htp_lib = d.htp_library() ? 1 : 0;
  if (htp_prov) *htp_prov = d.htp_providers() ? 1 : 0;
  copy_str(sdk_root, sdk_root_n, d.sdk_root);
  std::string bid;
  for (const auto& b : d.backends) {
    if (b.kind == BackendKind::Htp && !b.build_id.empty()) {
      bid = b.build_id;
      break;
    }
  }
  copy_str(build_id, build_id_n, bid);
  return ok();
}

wqnn_status wqnn_session_open_htp(wqnn_session** out) {
  if (!out) return from(Status::Failed("nil out"));
  auto* s = new wqnn_session();
  auto st = Session::open(BackendKind::Htp, &s->sess);
  if (!st) {
    delete s;
    return from(st);
  }
  *out = s;
  return ok();
}

void wqnn_session_close(wqnn_session* s) { delete s; }

int32_t wqnn_session_ready(const wqnn_session* s) {
  return (s && s->sess.ready()) ? 1 : 0;
}

wqnn_status wqnn_dense_prepare_f32(wqnn_session* s, int32_t batch, int32_t in, int32_t out,
                                   const float* W, const float* bias, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("dense", dtype);
    if (st.code) return st;
  }
  if (!W) return from(Status::Failed("nil W"));
#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
  return from(s->dense_prep.prepare(s->sess, batch, in, out, W, bias, parse_dtype(dtype)));
#else
  (void)batch;
  (void)in;
  (void)out;
  (void)bias;
  return from(Status::Unavailable("no SDK"));
#endif
}

wqnn_status wqnn_dense_execute_f32(wqnn_session* s, const float* x, float* y) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  if (!x || !y) return from(Status::Failed("nil x/y"));
#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
  return from(s->dense_prep.execute(s->sess, x, y));
#else
  return from(Status::Unavailable("no SDK"));
#endif
}

wqnn_status wqnn_dense_forward_f32(wqnn_session* s, int32_t batch, int32_t in, int32_t out,
                                   const float* W, const float* bias, const float* x, float* y,
                                   const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("dense", dtype);
    if (st.code) return st;
  }
#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
  auto st = s->dense_prep.prepare(s->sess, batch, in, out, W, bias, parse_dtype(dtype));
  if (!st) return from(st);
  return from(s->dense_prep.execute(s->sess, x, y));
#else
  Dense d = Dense::blank(in, out, parse_dtype(dtype));
  d.weight.assign(W, W + (size_t)out * (size_t)in);
  if (bias)
    d.bias.assign(bias, bias + out);
  else
    d.bias.assign((size_t)out, 0.f);
  return from(d.forward(s->sess, batch, x, y));
#endif
}

wqnn_status wqnn_softmax_forward_f32(wqnn_session* s, int32_t rows, int32_t cols, const float* x,
                                     float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("softmax", dtype);
    if (st.code) return st;
  }
  return from(Softmax::blank(cols, parse_dtype(dtype)).forward(s->sess, rows, x, y));
}

wqnn_status wqnn_rmsnorm_forward_f32(wqnn_session* s, int32_t batch, int32_t seq, int32_t dim,
                                     float eps, const float* gamma, const float* x, float* y,
                                     const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("rmsnorm", dtype);
    if (st.code) return st;
  }
  RmsNorm r = RmsNorm::blank(dim, eps, parse_dtype(dtype));
  if (gamma) r.gamma.assign(gamma, gamma + dim);
  return from(r.forward(s->sess, batch, seq, x, y));
}

wqnn_status wqnn_layernorm_forward_f32(wqnn_session* s, int32_t batch, int32_t seq, int32_t dim,
                                       float eps, const float* gamma, const float* beta,
                                       const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("layernorm", dtype);
    if (st.code) return st;
  }
  LayerNorm l = LayerNorm::blank(dim, eps, parse_dtype(dtype));
  if (gamma) l.gamma.assign(gamma, gamma + dim);
  if (beta) l.beta.assign(beta, beta + dim);
  return from(l.forward(s->sess, batch, seq, x, y));
}

wqnn_status wqnn_seq_create(wqnn_sequential** out) {
  if (!out) return from(Status::Failed("nil out"));
  *out = new wqnn_sequential();
  return ok();
}

void wqnn_seq_free(wqnn_sequential* seq) { delete seq; }

wqnn_status wqnn_seq_push_dense(wqnn_sequential* seq, int32_t in, int32_t out, const float* W,
                                const float* bias) {
  if (!seq) return from(Status::Failed("nil seq"));
  if (!wqnn_layer_supported("dense", "float32")) return from(Status::Unsupported("dense"));
  Dense d;
  d.in = in;
  d.out = out;
  d.dtype = DType::Float32;
  d.weight.assign(W, W + (size_t)out * (size_t)in);
  if (bias)
    d.bias.assign(bias, bias + out);
  else
    d.bias.assign((size_t)out, 0.f);
  return from(seq->net.push(std::move(d)));
}

wqnn_status wqnn_seq_push_rmsnorm(wqnn_sequential* seq, int32_t dim, float eps, const float* gamma) {
  if (!seq) return from(Status::Failed("nil seq"));
  if (!wqnn_layer_supported("rmsnorm", "float32")) return from(Status::Unsupported("rmsnorm"));
  RmsNorm r = RmsNorm::blank(dim, eps);
  if (gamma) r.gamma.assign(gamma, gamma + dim);
  return from(seq->net.push(std::move(r)));
}

wqnn_status wqnn_seq_push_layernorm(wqnn_sequential* seq, int32_t dim, float eps, const float* gamma,
                                    const float* beta) {
  if (!seq) return from(Status::Failed("nil seq"));
  if (!wqnn_layer_supported("layernorm", "float32")) return from(Status::Unsupported("layernorm"));
  LayerNorm l = LayerNorm::blank(dim, eps);
  if (gamma) l.gamma.assign(gamma, gamma + dim);
  if (beta) l.beta.assign(beta, beta + dim);
  return from(seq->net.push(std::move(l)));
}

wqnn_status wqnn_seq_push_softmax(wqnn_sequential* seq, int32_t cols) {
  if (!seq) return from(Status::Failed("nil seq"));
  if (!wqnn_layer_supported("softmax", "float32")) return from(Status::Unsupported("softmax"));
  return from(seq->net.push(Softmax::blank(cols)));
}

int32_t wqnn_seq_in_features(const wqnn_sequential* seq) {
  return seq ? seq->net.in_features() : 0;
}
int32_t wqnn_seq_out_features(const wqnn_sequential* seq) {
  return seq ? seq->net.out_features() : 0;
}
int32_t wqnn_seq_size(const wqnn_sequential* seq) {
  return seq ? (int32_t)seq->net.size() : 0;
}

wqnn_status wqnn_seq_forward_f32(wqnn_session* s, wqnn_sequential* seq, int32_t batch,
                                 int32_t seq_len, const float* x, float* y) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  if (!seq) return from(Status::Failed("nil seq"));
  return from(seq->net.forward(s->sess, batch, x, y, seq_len));
}

void wqnn_last_phase_ns(const wqnn_session* s, int64_t* create_ns, int64_t* finalize_ns,
                        int64_t* execute_ns) {
  if (!s) {
    if (create_ns) *create_ns = 0;
    if (finalize_ns) *finalize_ns = 0;
    if (execute_ns) *execute_ns = 0;
    return;
  }
  if (create_ns) *create_ns = s->sess.last_phase.create_ns;
  if (finalize_ns) *finalize_ns = s->sess.last_phase.finalize_ns;
  if (execute_ns) *execute_ns = s->sess.last_phase.execute_ns;
}

wqnn_status wqnn_cnn2_forward_f32(wqnn_session* s, int32_t n, int32_t h, int32_t w, int32_t cin,
                                  int32_t cout, const float* W, const float* bias, const float* x,
                                  float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("cnn2", dtype);
    if (st.code) return st;
  }
  Cnn2 c = Cnn2::blank(cin, cout, h, w, parse_dtype(dtype));
  c.n = n;
  if (W) c.weight.assign(W, W + (size_t)cout * (size_t)cin);
  if (bias) c.bias.assign(bias, bias + cout);
  return from(c.forward(s->sess, x, y));
}

wqnn_status wqnn_cnn3_forward_f32(wqnn_session* s, int32_t n, int32_t d, int32_t h, int32_t w,
                                  int32_t cin, int32_t cout, const float* W, const float* bias,
                                  const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("cnn3", dtype);
    if (st.code) return st;
  }
  Cnn3 c = Cnn3::blank(cin, cout, d, h, w, parse_dtype(dtype));
  c.n = n;
  if (W) c.weight.assign(W, W + c.weight.size());
  if (bias) c.bias.assign(bias, bias + cout);
  return from(c.forward(s->sess, x, y));
}

wqnn_status wqnn_convt2_forward_f32(wqnn_session* s, int32_t n, int32_t h, int32_t w, int32_t cin,
                                    int32_t cout, const float* W, const float* bias, const float* x,
                                    float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("convt2", dtype);
    if (st.code) return st;
  }
  ConvT2 c = ConvT2::blank(cin, cout, h, w, parse_dtype(dtype));
  c.n = n;
  if (W) c.weight.assign(W, W + c.weight.size());
  if (bias) c.bias.assign(bias, bias + cout);
  return from(c.forward(s->sess, x, y));
}

wqnn_status wqnn_convt3_forward_f32(wqnn_session* s, int32_t n, int32_t d, int32_t h, int32_t w,
                                    int32_t cin, int32_t cout, const float* W, const float* bias,
                                    const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("convt3", dtype);
    if (st.code) return st;
  }
  ConvT3 c = ConvT3::blank(cin, cout, d, h, w, parse_dtype(dtype));
  c.n = n;
  if (W) c.weight.assign(W, W + c.weight.size());
  if (bias) c.bias.assign(bias, bias + cout);
  return from(c.forward(s->sess, x, y));
}

wqnn_status wqnn_lstm_forward_f32(wqnn_session* s, int32_t seq, int32_t batch, int32_t input_size,
                                  int32_t hidden_size, const float* w_in, const float* w_rec,
                                  const float* b, const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("lstm", dtype);
    if (st.code) return st;
  }
  Lstm l = Lstm::blank(input_size, hidden_size, parse_dtype(dtype));
  if (w_in) l.w_in.assign(w_in, w_in + l.w_in.size());
  if (w_rec) l.w_rec.assign(w_rec, w_rec + l.w_rec.size());
  if (b) l.b.assign(b, b + l.b.size());
  return from(l.forward(s->sess, seq, batch, x, y));
}

wqnn_status wqnn_embedding_forward_f32(wqnn_session* s, int32_t vocab, int32_t dim, int32_t n,
                                       const float* table, const int32_t* ids, float* y,
                                       const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("embedding", dtype);
    if (st.code) return st;
  }
  Embedding e = Embedding::blank(vocab, dim, parse_dtype(dtype));
  if (table) e.table.assign(table, table + (size_t)vocab * (size_t)dim);
  return from(e.forward(s->sess, n, ids, y));
}

wqnn_status wqnn_residual_forward_f32(wqnn_session* s, int32_t n, const float* a, const float* b,
                                      float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("residual", dtype);
    if (st.code) return st;
  }
  return from(Residual::blank(parse_dtype(dtype)).forward(s->sess, n, a, b, y));
}

wqnn_status wqnn_swiglu_forward_f32(wqnn_session* s, int32_t batch, int32_t in, int32_t hidden,
                                    const float* w_gate, const float* w_up, const float* w_down,
                                    const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("swiglu", dtype);
    if (st.code) return st;
  }
  SwiGlu g = SwiGlu::blank(in, hidden, parse_dtype(dtype));
  if (w_gate) g.w_gate.assign(w_gate, w_gate + g.w_gate.size());
  if (w_up) g.w_up.assign(w_up, w_up + g.w_up.size());
  if (w_down) g.w_down.assign(w_down, w_down + g.w_down.size());
  return from(g.forward(s->sess, batch, x, y));
}

wqnn_status wqnn_mha_forward_f32(wqnn_session* s, int32_t batch, int32_t seq, int32_t dim,
                                 int32_t heads, const float* w_q, const float* w_k, const float* w_v,
                                 const float* w_o, const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no HTP session"));
  {
    auto st = need_layer("mha", dtype);
    if (st.code) return st;
  }
  Mha m = Mha::blank(dim, heads, parse_dtype(dtype));
  if (w_q) m.w_q.assign(w_q, w_q + m.w_q.size());
  if (w_k) m.w_k.assign(w_k, w_k + m.w_k.size());
  if (w_v) m.w_v.assign(w_v, w_v + m.w_v.size());
  if (w_o) m.w_o.assign(w_o, w_o + m.w_o.size());
  return from(m.forward(s->sess, batch, seq, x, y));
}

} // extern "C"
