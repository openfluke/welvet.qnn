#pragma once

// Multi-op HTP graphs (Welvet-shaped packs). No ONNX types in the public API —
// these helpers exist so C ABI / FFI can call the same Session+Layer surface later.

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK

#include <cstring>

#include "graph_util.hpp"

namespace welvet_qnn::graph {

inline Status add_node(QNN_INTERFACE_VER_TYPE* api, Qnn_GraphHandle_t graph, const char* name,
                       const char* type, Qnn_Tensor_t* ins, uint32_t n_in, Qnn_Tensor_t* outs,
                       uint32_t n_out, Qnn_Param_t* params = nullptr, uint32_t n_params = 0) {
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = name;
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = type;
  op.v1.numOfParams = n_params;
  op.v1.params = params;
  op.v1.numOfInputs = n_in;
  op.v1.inputTensors = ins;
  op.v1.numOfOutputs = n_out;
  op.v1.outputTensors = outs;
  if (api->graphAddNode(graph, op) != QNN_SUCCESS)
    return Status::Failed(std::string("addNode ") + type);
  return Status::Ok();
}

inline Qnn_Tensor_t native_tensor(const char* name, uint32_t rank, uint32_t* dims, DType dt) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = QNN_TENSOR_VERSION_1;
  t.v1 = QNN_TENSOR_V1_INIT;
  t.v1.name = name;
  t.v1.type = QNN_TENSOR_TYPE_NATIVE;
  t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType = (Qnn_DataType_t)dtype_to_qnn(dt);
  t.v1.rank = rank;
  t.v1.dimensions = dims;
  t.v1.memType = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf.data = nullptr;
  t.v1.clientBuf.dataSize = 0;
  apply_quant_if_needed(&t, dt);
  return t;
}

// RMSNorm compose: y = x * rsqrt(mean(x^2)+eps) * gamma  (native RmsNorm disabled on V73/HNRD)
inline Status run_rmsnorm_compose(Session& sess, int32_t tokens, int32_t dim, float eps,
                                  const float* gamma, const float* x, float* y, DType dt) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");
  const int64_t t0 = mono_ns();
  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), uniq_name("wq_rmsc"), nullptr, &graph) != QNN_SUCCESS)
    return Status::Failed("graphCreate");

  uint32_t x_dims[2] = {(uint32_t)tokens, (uint32_t)dim};
  uint32_t red_dims[2] = {(uint32_t)tokens, 1};
  uint32_t g_dims[1] = {(uint32_t)dim};
  size_t n = (size_t)tokens * dim;
  auto xb = pack_f32(x, n, dt);
  auto gb = pack_f32(gamma, (size_t)dim, dt);
  std::vector<float> eps_host((size_t)tokens, eps);
  auto eb = pack_f32(eps_host.data(), (size_t)tokens, dt);
  std::vector<uint8_t> yb(n * dtype_nbytes(dt), 0);

  Qnn_Tensor_t tx =
      make_tensor(uniq_name("x"), QNN_TENSOR_TYPE_APP_WRITE, 2, x_dims, xb.data(), (uint32_t)xb.size(),
                  dt);
  Qnn_Tensor_t tg =
      make_tensor(uniq_name("g"), QNN_TENSOR_TYPE_STATIC, 1, g_dims, gb.data(), (uint32_t)gb.size(), dt);
  uint32_t eps_dims[2] = {(uint32_t)tokens, 1};
  Qnn_Tensor_t teps = make_tensor(uniq_name("eps"), QNN_TENSOR_TYPE_STATIC, 2, eps_dims, eb.data(),
                                  (uint32_t)eb.size(), dt);

  Qnn_Tensor_t tsq = native_tensor(uniq_name("sq"), 2, x_dims, dt);
  Qnn_Tensor_t tmean = native_tensor(uniq_name("mean"), 2, red_dims, dt);
  Qnn_Tensor_t tadd = native_tensor(uniq_name("add"), 2, red_dims, dt);
  Qnn_Tensor_t trs = native_tensor(uniq_name("rs"), 2, red_dims, dt);
  Qnn_Tensor_t tnorm = native_tensor(uniq_name("norm"), 2, x_dims, dt);
  Qnn_Tensor_t ty =
      make_tensor(uniq_name("y"), QNN_TENSOR_TYPE_APP_READ, 2, x_dims, yb.data(), (uint32_t)yb.size(),
                  dt);

  if (api->tensorCreateGraphTensor(graph, &tx) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tg) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &teps) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tsq) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tmean) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tadd) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &trs) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tnorm) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &ty) != QNN_SUCCESS)
    return Status::Failed("rmsnorm tensors");

  // x*x
  Qnn_Tensor_t mul_in[2] = {tx, tx};
  auto st = add_node(api, graph, uniq_name("sq"), QNN_OP_ELEMENT_WISE_MULTIPLY, mul_in, 2, &tsq, 1);
  if (!st) return st;

  // mean over last axis, keepdims
  int32_t axes_data[1] = {1};
  uint32_t axes_dims[1] = {1};
  Qnn_Tensor_t axes_t = QNN_TENSOR_INIT;
  axes_t.version = QNN_TENSOR_VERSION_1;
  axes_t.v1 = QNN_TENSOR_V1_INIT;
  axes_t.v1.name = uniq_name("axes");
  axes_t.v1.type = QNN_TENSOR_TYPE_STATIC;
  axes_t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  axes_t.v1.dataType = QNN_DATATYPE_INT_32;
  axes_t.v1.rank = 1;
  axes_t.v1.dimensions = axes_dims;
  axes_t.v1.memType = QNN_TENSORMEMTYPE_RAW;
  axes_t.v1.clientBuf.data = axes_data;
  axes_t.v1.clientBuf.dataSize = sizeof(axes_data);
  if (api->tensorCreateGraphTensor(graph, &axes_t) != QNN_SUCCESS) return Status::Failed("axes");
  Qnn_Param_t rparams[2];
  rparams[0] = QNN_PARAM_INIT;
  rparams[0].paramType = QNN_PARAMTYPE_TENSOR;
  rparams[0].name = QNN_OP_REDUCE_MEAN_PARAM_AXES;
  rparams[0].tensorParam = axes_t;
  rparams[1] = QNN_PARAM_INIT;
  rparams[1].paramType = QNN_PARAMTYPE_SCALAR;
  rparams[1].name = QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS;
  rparams[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  rparams[1].scalarParam.bool8Value = 1;
  st = add_node(api, graph, uniq_name("mean"), QNN_OP_REDUCE_MEAN, &tsq, 1, &tmean, 1, rparams, 2);
  if (!st) return st;

  Qnn_Tensor_t add_in[2] = {tmean, teps};
  st = add_node(api, graph, uniq_name("add"), QNN_OP_ELEMENT_WISE_ADD, add_in, 2, &tadd, 1);
  if (!st) return st;
  st = add_node(api, graph, uniq_name("rs"), QNN_OP_ELEMENT_WISE_RSQRT, &tadd, 1, &trs, 1);
  if (!st) return st;
  Qnn_Tensor_t nm_in[2] = {tx, trs};
  st = add_node(api, graph, uniq_name("norm"), QNN_OP_ELEMENT_WISE_MULTIPLY, nm_in, 2, &tnorm, 1);
  if (!st) return st;
  Qnn_Tensor_t sc_in[2] = {tnorm, tg};
  st = add_node(api, graph, uniq_name("scale"), QNN_OP_ELEMENT_WISE_MULTIPLY, sc_in, 2, &ty, 1);
  if (!st) return st;

  const int64_t t1 = mono_ns();
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
  const int64_t t2 = mono_ns();
  Qnn_Tensor_t ex = tx, ey = ty;
  ex.v1.clientBuf.data = xb.data();
  ex.v1.clientBuf.dataSize = (uint32_t)xb.size();
  ey.v1.clientBuf.data = yb.data();
  ey.v1.clientBuf.dataSize = (uint32_t)yb.size();
  if (api->graphExecute(graph, &ex, 1, &ey, 1, nullptr, nullptr) != QNN_SUCCESS)
    return Status::Failed("execute");
  const int64_t t3 = mono_ns();
  unpack_to_f32(yb.data(), n, dt, y);
  mark_phase(sess, t1 - t0, t2 - t1, t3 - t2);
  return Status::Ok();
}

// MHA: Welvet Dense Q/K/V/O + MatMul attention (single-head math on [B*H, S, D/H] via flat MatMul).
// Blank path: scores = Q @ K^T (as [tokens,dim] @ [dim,tokens] via transpose_in1), softmax, @ V, then O.
inline Status run_mha_compose(Session& sess, int32_t batch, int32_t seq, int32_t dim, int32_t heads,
                              const float* w_q, const float* w_k, const float* w_v, const float* w_o,
                              const float* x, float* y, DType dt) {
  (void)heads; // head-split reshape is next; blank uses full-dim attention MatMul
  const int32_t tokens = batch * seq;
  std::vector<float> q((size_t)tokens * dim), k((size_t)tokens * dim), v((size_t)tokens * dim),
      scores((size_t)tokens * tokens), attn((size_t)tokens * tokens), ctx((size_t)tokens * dim);
  auto st = run_fc(sess, tokens, dim, dim, w_q, x, q.data(), dt);
  if (!st) return st;
  st = run_fc(sess, tokens, dim, dim, w_k, x, k.data(), dt);
  if (!st) return st;
  st = run_fc(sess, tokens, dim, dim, w_v, x, v.data(), dt);
  if (!st) return st;

  // MatMul Q @ K^T → [tokens, tokens]
  {
    auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
    Qnn_GraphHandle_t graph = nullptr;
    if (api->graphCreate(sess.context_handle(), uniq_name("wq_mha_qk"), nullptr, &graph) !=
        QNN_SUCCESS)
      return Status::Failed("graphCreate qk");
    uint32_t q_dims[2] = {(uint32_t)tokens, (uint32_t)dim};
    uint32_t k_dims[2] = {(uint32_t)tokens, (uint32_t)dim};
    uint32_t s_dims[2] = {(uint32_t)tokens, (uint32_t)tokens};
    auto qb = pack_f32(q.data(), q.size(), dt);
    auto kb = pack_f32(k.data(), k.size(), dt);
    std::vector<uint8_t> sb((size_t)tokens * tokens * dtype_nbytes(dt), 0);
    Qnn_Tensor_t tq =
        make_tensor(uniq_name("q"), QNN_TENSOR_TYPE_APP_WRITE, 2, q_dims, qb.data(), (uint32_t)qb.size(),
                    dt);
    Qnn_Tensor_t tk =
        make_tensor(uniq_name("k"), QNN_TENSOR_TYPE_APP_WRITE, 2, k_dims, kb.data(), (uint32_t)kb.size(),
                    dt);
    Qnn_Tensor_t ts =
        make_tensor(uniq_name("s"), QNN_TENSOR_TYPE_APP_READ, 2, s_dims, sb.data(), (uint32_t)sb.size(),
                    dt);
    if (api->tensorCreateGraphTensor(graph, &tq) != QNN_SUCCESS ||
        api->tensorCreateGraphTensor(graph, &tk) != QNN_SUCCESS ||
        api->tensorCreateGraphTensor(graph, &ts) != QNN_SUCCESS)
      return Status::Failed("qk tensor");
    Qnn_Param_t p0 = QNN_PARAM_INIT, p1 = QNN_PARAM_INIT;
    p0.paramType = QNN_PARAMTYPE_SCALAR;
    p0.name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    p0.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    p0.scalarParam.bool8Value = 0;
    p1.paramType = QNN_PARAMTYPE_SCALAR;
    p1.name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    p1.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    p1.scalarParam.bool8Value = 1;
    Qnn_Param_t ps[2] = {p0, p1};
    Qnn_Tensor_t ins[2] = {tq, tk};
    st = add_node(api, graph, uniq_name("mm"), QNN_OP_MAT_MUL, ins, 2, &ts, 1, ps, 2);
    if (!st) return st;
    if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("qk fin");
    Qnn_Tensor_t eq = tq, ek = tk, es = ts;
    eq.v1.clientBuf.data = qb.data();
    eq.v1.clientBuf.dataSize = (uint32_t)qb.size();
    ek.v1.clientBuf.data = kb.data();
    ek.v1.clientBuf.dataSize = (uint32_t)kb.size();
    es.v1.clientBuf.data = sb.data();
    es.v1.clientBuf.dataSize = (uint32_t)sb.size();
    Qnn_Tensor_t eins[2] = {eq, ek};
    if (api->graphExecute(graph, eins, 2, &es, 1, nullptr, nullptr) != QNN_SUCCESS)
      return Status::Failed("qk exec");
    unpack_to_f32(sb.data(), (size_t)tokens * tokens, dt, scores.data());
  }

  uint32_t sm_dims[2] = {(uint32_t)tokens, (uint32_t)tokens};
  st = run_unary(sess, "wq_mha_sm", QNN_OP_SOFTMAX, 2, sm_dims, scores.data(), attn.data(),
                 (size_t)tokens * tokens, dt, nullptr, 0, nullptr, nullptr, 0, nullptr);
  if (!st) return st;

  // attn @ V → [tokens, dim]
  {
    auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
    Qnn_GraphHandle_t graph = nullptr;
    if (api->graphCreate(sess.context_handle(), uniq_name("wq_mha_av"), nullptr, &graph) !=
        QNN_SUCCESS)
      return Status::Failed("graphCreate av");
    uint32_t a_dims[2] = {(uint32_t)tokens, (uint32_t)tokens};
    uint32_t v_dims[2] = {(uint32_t)tokens, (uint32_t)dim};
    uint32_t c_dims[2] = {(uint32_t)tokens, (uint32_t)dim};
    auto ab = pack_f32(attn.data(), attn.size(), dt);
    auto vb = pack_f32(v.data(), v.size(), dt);
    std::vector<uint8_t> cb((size_t)tokens * dim * dtype_nbytes(dt), 0);
    Qnn_Tensor_t ta =
        make_tensor(uniq_name("a"), QNN_TENSOR_TYPE_APP_WRITE, 2, a_dims, ab.data(), (uint32_t)ab.size(),
                    dt);
    Qnn_Tensor_t tv =
        make_tensor(uniq_name("v"), QNN_TENSOR_TYPE_APP_WRITE, 2, v_dims, vb.data(), (uint32_t)vb.size(),
                    dt);
    Qnn_Tensor_t tc =
        make_tensor(uniq_name("c"), QNN_TENSOR_TYPE_APP_READ, 2, c_dims, cb.data(), (uint32_t)cb.size(),
                    dt);
    if (api->tensorCreateGraphTensor(graph, &ta) != QNN_SUCCESS ||
        api->tensorCreateGraphTensor(graph, &tv) != QNN_SUCCESS ||
        api->tensorCreateGraphTensor(graph, &tc) != QNN_SUCCESS)
      return Status::Failed("av tensor");
    Qnn_Tensor_t ins[2] = {ta, tv};
    st = add_node(api, graph, uniq_name("mm"), QNN_OP_MAT_MUL, ins, 2, &tc, 1);
    if (!st) return st;
    if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("av fin");
    Qnn_Tensor_t ea = ta, ev = tv, ec = tc;
    ea.v1.clientBuf.data = ab.data();
    ea.v1.clientBuf.dataSize = (uint32_t)ab.size();
    ev.v1.clientBuf.data = vb.data();
    ev.v1.clientBuf.dataSize = (uint32_t)vb.size();
    ec.v1.clientBuf.data = cb.data();
    ec.v1.clientBuf.dataSize = (uint32_t)cb.size();
    Qnn_Tensor_t eins[2] = {ea, ev};
    if (api->graphExecute(graph, eins, 2, &ec, 1, nullptr, nullptr) != QNN_SUCCESS)
      return Status::Failed("av exec");
    unpack_to_f32(cb.data(), (size_t)tokens * dim, dt, ctx.data());
  }
  return run_fc(sess, tokens, dim, dim, w_o, ctx.data(), y, dt);
}

// LSTM on HTP: Welvet gate pack [i|f|g|o] × (IH|HH|bias) — composed from FC + Sigmoid/Tanh + Add/Mul.
// Same packing Welvet Go / C ABI will expose later (not ONNX W/R/B tensors).
inline Status run_lstm_welvet(Session& sess, int32_t seq, int32_t batch, int32_t input_size,
                              int32_t hidden, const float* w_in, const float* w_rec, const float* b,
                              const float* x, float* y, DType dt) {
  // w_in: [4H, I], w_rec: [4H, H], b: [4H]  — gates i,f,g,o contiguous
  const size_t H = (size_t)hidden;
  const size_t I = (size_t)input_size;
  auto slice_row = [&](const float* W, int gate, int cols) -> const float* {
    return W + (size_t)gate * H * (size_t)cols;
  };
  auto slice_b = [&](int gate) -> const float* { return b + (size_t)gate * H; };

  std::vector<float> h((size_t)batch * H, 0.f), c((size_t)batch * H, 0.f);
  std::vector<float> xt((size_t)batch * I), gi((size_t)batch * H), gf((size_t)batch * H),
      gg((size_t)batch * H), go((size_t)batch * H), hi((size_t)batch * H), hf((size_t)batch * H),
      hg((size_t)batch * H), ho((size_t)batch * H), si((size_t)batch * H), sf((size_t)batch * H),
      sg((size_t)batch * H), so((size_t)batch * H), tmp((size_t)batch * H), cnew((size_t)batch * H);

  for (int32_t t = 0; t < seq; ++t) {
    // x_t
    std::memcpy(xt.data(), x + (size_t)t * batch * input_size, xt.size() * sizeof(float));

    auto st = run_fc(sess, batch, input_size, hidden, slice_row(w_in, 0, input_size), xt.data(),
                     gi.data(), dt);
    if (!st) return st;
    st = run_fc(sess, batch, hidden, hidden, slice_row(w_rec, 0, hidden), h.data(), hi.data(), dt);
    if (!st) return st;
    st = run_binary(sess, "wq_lstm_i", QNN_OP_ELEMENT_WISE_ADD, gi.size(), gi.data(), hi.data(),
                    tmp.data(), dt);
    if (!st) return st;
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] += slice_b(0)[i % H];
    uint32_t d2[2] = {(uint32_t)batch, (uint32_t)hidden};
    st = run_unary(sess, "wq_lstm_si", QNN_OP_SIGMOID, 2, d2, tmp.data(), si.data(), tmp.size(), dt,
                   nullptr, 0, nullptr, nullptr, 0, nullptr);
    if (!st) return st;

    st = run_fc(sess, batch, input_size, hidden, slice_row(w_in, 1, input_size), xt.data(),
                gf.data(), dt);
    if (!st) return st;
    st = run_fc(sess, batch, hidden, hidden, slice_row(w_rec, 1, hidden), h.data(), hf.data(), dt);
    if (!st) return st;
    st = run_binary(sess, "wq_lstm_f", QNN_OP_ELEMENT_WISE_ADD, gf.size(), gf.data(), hf.data(),
                    tmp.data(), dt);
    if (!st) return st;
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] += slice_b(1)[i % H];
    st = run_unary(sess, "wq_lstm_sf", QNN_OP_SIGMOID, 2, d2, tmp.data(), sf.data(), tmp.size(), dt,
                   nullptr, 0, nullptr, nullptr, 0, nullptr);
    if (!st) return st;

    st = run_fc(sess, batch, input_size, hidden, slice_row(w_in, 2, input_size), xt.data(),
                gg.data(), dt);
    if (!st) return st;
    st = run_fc(sess, batch, hidden, hidden, slice_row(w_rec, 2, hidden), h.data(), hg.data(), dt);
    if (!st) return st;
    st = run_binary(sess, "wq_lstm_g", QNN_OP_ELEMENT_WISE_ADD, gg.size(), gg.data(), hg.data(),
                    tmp.data(), dt);
    if (!st) return st;
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] += slice_b(2)[i % H];
    st = run_unary(sess, "wq_lstm_tg", QNN_OP_TANH, 2, d2, tmp.data(), sg.data(), tmp.size(), dt,
                   nullptr, 0, nullptr, nullptr, 0, nullptr);
    if (!st) return st;

    st = run_fc(sess, batch, input_size, hidden, slice_row(w_in, 3, input_size), xt.data(),
                go.data(), dt);
    if (!st) return st;
    st = run_fc(sess, batch, hidden, hidden, slice_row(w_rec, 3, hidden), h.data(), ho.data(), dt);
    if (!st) return st;
    st = run_binary(sess, "wq_lstm_o", QNN_OP_ELEMENT_WISE_ADD, go.size(), go.data(), ho.data(),
                    tmp.data(), dt);
    if (!st) return st;
    for (size_t i = 0; i < tmp.size(); ++i) tmp[i] += slice_b(3)[i % H];
    st = run_unary(sess, "wq_lstm_so", QNN_OP_SIGMOID, 2, d2, tmp.data(), so.data(), tmp.size(), dt,
                   nullptr, 0, nullptr, nullptr, 0, nullptr);
    if (!st) return st;

    // c = f⊙c + i⊙g
    std::vector<float> fc((size_t)batch * H), ig((size_t)batch * H);
    st = run_binary(sess, "wq_lstm_fc", QNN_OP_ELEMENT_WISE_MULTIPLY, c.size(), sf.data(), c.data(),
                    fc.data(), dt);
    if (!st) return st;
    st = run_binary(sess, "wq_lstm_ig", QNN_OP_ELEMENT_WISE_MULTIPLY, si.size(), si.data(), sg.data(),
                    ig.data(), dt);
    if (!st) return st;
    st = run_binary(sess, "wq_lstm_c", QNN_OP_ELEMENT_WISE_ADD, fc.size(), fc.data(), ig.data(),
                    cnew.data(), dt);
    if (!st) return st;
    c.swap(cnew);

    // h = o ⊙ tanh(c)
    st = run_unary(sess, "wq_lstm_tc", QNN_OP_TANH, 2, d2, c.data(), tmp.data(), c.size(), dt,
                   nullptr, 0, nullptr, nullptr, 0, nullptr);
    if (!st) return st;
    st = run_binary(sess, "wq_lstm_h", QNN_OP_ELEMENT_WISE_MULTIPLY, so.size(), so.data(), tmp.data(),
                    h.data(), dt);
    if (!st) return st;

    std::memcpy(y + (size_t)t * batch * hidden, h.data(), h.size() * sizeof(float));
  }
  return Status::Ok();
}

} // namespace welvet_qnn::graph

#endif
