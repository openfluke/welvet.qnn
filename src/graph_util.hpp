#pragma once

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnOpDef.h"
#include "welvet_qnn/dtype.hpp"
#include "welvet_qnn/session.hpp"
#include "welvet_qnn/status.hpp"

namespace welvet_qnn::graph {

inline int64_t mono_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline void mark_phase(Session& sess, int64_t create_ns, int64_t finalize_ns, int64_t execute_ns) {
  sess.last_phase.create_ns = create_ns;
  sess.last_phase.finalize_ns = finalize_ns;
  sess.last_phase.execute_ns = execute_ns;
}

// Stable unique C-string (kept for the process). Process-wide counter — cgo may
// call from different OS threads; thread_local counters collide on HTP graphCreate.
inline const char* uniq_name(const char* prefix) {
  static std::mutex mu;
  static std::vector<std::string> pool;
  static std::atomic<uint32_t> n{0};
  char buf[96];
  std::snprintf(buf, sizeof(buf), "%s_%u", prefix, n.fetch_add(1) + 1);
  std::lock_guard<std::mutex> lock(mu);
  pool.emplace_back(buf);
  return pool.back().c_str();
}

inline void apply_quant_if_needed(Qnn_Tensor_t* t, DType dt) {
  // Integer / fixed types need a trivial encoding so create succeeds.
  if (dt == DType::Float32 || dt == DType::Float16 || dt == DType::BFloat16) return;
  t->v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  t->v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  t->v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
  t->v1.quantizeParams.scaleOffsetEncoding.offset = 0;
}

inline Qnn_Tensor_t make_tensor(const char* name, Qnn_TensorType_t type, uint32_t rank,
                                uint32_t* dims, void* data, uint32_t nbytes, DType dt) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = QNN_TENSOR_VERSION_1;
  t.v1 = QNN_TENSOR_V1_INIT;
  t.v1.name = name;
  t.v1.type = type;
  t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType = (Qnn_DataType_t)dtype_to_qnn(dt);
  t.v1.rank = rank;
  t.v1.dimensions = dims;
  t.v1.memType = QNN_TENSORMEMTYPE_RAW;
  // HTP: APP_WRITE/APP_READ must have nullptr clientBuf at tensorCreateGraphTensor (error 7004).
  // Data is bound later in graphExecute. STATIC keeps payload at create.
  const bool app = (type == QNN_TENSOR_TYPE_APP_WRITE || type == QNN_TENSOR_TYPE_APP_READ ||
                    type == QNN_TENSOR_TYPE_APP_READWRITE);
  if (app) {
    t.v1.clientBuf.data = nullptr;
    t.v1.clientBuf.dataSize = 0;
  } else {
    t.v1.clientBuf.data = data;
    t.v1.clientBuf.dataSize = nbytes;
  }
  apply_quant_if_needed(&t, dt);
  return t;
}

// Pack float32 host → target dtype bytes (identity for f32; trunc/cast otherwise).
inline std::vector<uint8_t> pack_f32(const float* src, size_t n, DType dt) {
  std::vector<uint8_t> out(n * dtype_nbytes(dt));
  if (dt == DType::Float32) {
    std::memcpy(out.data(), src, n * 4);
    return out;
  }
  for (size_t i = 0; i < n; ++i) {
    float v = src[i];
    uint8_t* p = out.data() + i * dtype_nbytes(dt);
    switch (dt) {
      case DType::Float16: {
        // IEEE754 binary16 round-to-nearest-even (simple).
        uint32_t x;
        std::memcpy(&x, &v, 4);
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
        uint32_t mant = x & 0x7fffff;
        uint16_t h;
        if (exp <= 0) {
          h = (uint16_t)sign;
        } else if (exp >= 31) {
          h = (uint16_t)(sign | 0x7c00);
        } else {
          h = (uint16_t)(sign | (exp << 10) | (mant >> 13));
        }
        std::memcpy(p, &h, 2);
        break;
      }
      case DType::BFloat16: {
        uint32_t x;
        std::memcpy(&x, &v, 4);
        uint16_t h = (uint16_t)(x >> 16);
        std::memcpy(p, &h, 2);
        break;
      }
      case DType::Int8: {
        int v_i = (int)v;
        if (v_i > 127) v_i = 127;
        if (v_i < -128) v_i = -128;
        *p = (uint8_t)(int8_t)v_i;
        break;
      }
      case DType::UInt8: {
        int v_i = (int)v;
        if (v_i > 255) v_i = 255;
        if (v_i < 0) v_i = 0;
        *p = (uint8_t)v_i;
        break;
      }
      case DType::Int16: {
        int v_i = (int)v;
        if (v_i > 32767) v_i = 32767;
        if (v_i < -32768) v_i = -32768;
        int16_t s = (int16_t)v_i;
        std::memcpy(p, &s, 2);
        break;
      }
      case DType::UInt16: {
        int v_i = (int)v;
        if (v_i > 65535) v_i = 65535;
        if (v_i < 0) v_i = 0;
        uint16_t s = (uint16_t)v_i;
        std::memcpy(p, &s, 2);
        break;
      }
      default:
        break;
    }
  }
  return out;
}

inline void unpack_to_f32(const uint8_t* src, size_t n, DType dt, float* dst) {
  if (dt == DType::Float32) {
    std::memcpy(dst, src, n * 4);
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* p = src + i * dtype_nbytes(dt);
    switch (dt) {
      case DType::Float16: {
        uint16_t h;
        std::memcpy(&h, p, 2);
        uint32_t sign = (uint32_t)(h & 0x8000) << 16;
        int32_t exp = (h >> 10) & 0x1f;
        uint32_t mant = h & 0x3ff;
        uint32_t f;
        if (exp == 0) {
          f = sign;
        } else if (exp == 31) {
          f = sign | 0x7f800000 | (mant << 13);
        } else {
          f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        std::memcpy(&dst[i], &f, 4);
        break;
      }
      case DType::BFloat16: {
        uint16_t h;
        std::memcpy(&h, p, 2);
        uint32_t f = ((uint32_t)h) << 16;
        std::memcpy(&dst[i], &f, 4);
        break;
      }
      case DType::Int8:
        dst[i] = (float)(int8_t)p[0];
        break;
      case DType::UInt8:
        dst[i] = (float)p[0];
        break;
      case DType::Int16: {
        int16_t s;
        std::memcpy(&s, p, 2);
        dst[i] = (float)s;
        break;
      }
      case DType::UInt16: {
        uint16_t s;
        std::memcpy(&s, p, 2);
        dst[i] = (float)s;
        break;
      }
      default:
        dst[i] = 0;
        break;
    }
  }
}

inline Status run_fc(Session& sess, int32_t batch, int32_t in, int32_t out, const float* W,
                     const float* x, float* y, DType dt) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");

  const int64_t t0 = mono_ns();
  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), uniq_name("wq_dense"), nullptr, &graph) != QNN_SUCCESS)
    return Status::Failed("graphCreate");

  uint32_t in_dims[2] = {(uint32_t)batch, (uint32_t)in};
  uint32_t w_dims[2] = {(uint32_t)out, (uint32_t)in};
  uint32_t out_dims[2] = {(uint32_t)batch, (uint32_t)out};
  const size_t n_in = (size_t)batch * in;
  const size_t n_out = (size_t)batch * out;
  const size_t n_w = (size_t)out * in;

  auto in_buf = pack_f32(x, n_in, dt);
  auto w_buf = pack_f32(W, n_w, dt);
  std::vector<uint8_t> out_buf(n_out * dtype_nbytes(dt), 0);

  Qnn_Tensor_t tin = make_tensor(uniq_name("x"), QNN_TENSOR_TYPE_APP_WRITE, 2, in_dims, in_buf.data(),
                                 (uint32_t)in_buf.size(), dt);
  Qnn_Tensor_t tw = make_tensor(uniq_name("w"), QNN_TENSOR_TYPE_STATIC, 2, w_dims, w_buf.data(),
                                (uint32_t)w_buf.size(), dt);
  Qnn_Tensor_t tout = make_tensor(uniq_name("y"), QNN_TENSOR_TYPE_APP_READ, 2, out_dims, out_buf.data(),
                                  (uint32_t)out_buf.size(), dt);

  if (api->tensorCreateGraphTensor(graph, &tin) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tw) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tout) != QNN_SUCCESS)
    return Status::Failed("tensorCreateGraphTensor");

  Qnn_Tensor_t ins[2] = {tin, tw};
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = uniq_name("fc0");
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_FULLY_CONNECTED;
  op.v1.numOfParams = 0;
  op.v1.params = nullptr;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = ins;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &tout;

  if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed("addNode FC");
  const int64_t t1 = mono_ns();
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
  const int64_t t2 = mono_ns();

  Qnn_Tensor_t ein = tin, eout = tout;
  ein.v1.clientBuf.data = in_buf.data();
  ein.v1.clientBuf.dataSize = (uint32_t)in_buf.size();
  eout.v1.clientBuf.data = out_buf.data();
  eout.v1.clientBuf.dataSize = (uint32_t)out_buf.size();
  if (api->graphExecute(graph, &ein, 1, &eout, 1, nullptr, nullptr) != QNN_SUCCESS)
    return Status::Failed("execute");
  const int64_t t3 = mono_ns();
  unpack_to_f32(out_buf.data(), n_out, dt, y);
  mark_phase(sess, t1 - t0, t2 - t1, t3 - t2);
  return Status::Ok();
}

inline Status run_unary(Session& sess, const char* gname, const char* opname, uint32_t rank,
                        uint32_t* dims, const float* in, float* out, size_t n, DType dt,
                        const float* static_a, size_t static_a_n, const char* static_a_name,
                        const float* static_b, size_t static_b_n, const char* static_b_name,
                        float epsilon = -1.f, bool with_axes = false) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");

  const int64_t t0 = mono_ns();
  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), uniq_name(gname), nullptr, &graph) != QNN_SUCCESS)
    return Status::Failed("graphCreate");

  auto in_buf = pack_f32(in, n, dt);
  std::vector<uint8_t> out_buf(n * dtype_nbytes(dt), 0);

  Qnn_Tensor_t tin =
      make_tensor(uniq_name("in"), QNN_TENSOR_TYPE_APP_WRITE, rank, dims, in_buf.data(),
                  (uint32_t)in_buf.size(), dt);
  Qnn_Tensor_t tout =
      make_tensor(uniq_name("out"), QNN_TENSOR_TYPE_APP_READ, rank, dims, out_buf.data(),
                  (uint32_t)out_buf.size(), dt);

  if (api->tensorCreateGraphTensor(graph, &tin) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tout) != QNN_SUCCESS)
    return Status::Failed("createGraphTensor");

  std::vector<Qnn_Tensor_t> inputs{tin};
  std::vector<uint8_t> sa, sb;
  Qnn_Tensor_t tsa{}, tsb{};
  uint32_t sd_a[1]{}, sd_b[1]{};
  // For norms, HTP prefers weight/bias shaped like the normalized axis (1-D [dim]).
  if (static_a && static_a_n) {
    sa = pack_f32(static_a, static_a_n, dt);
    sd_a[0] = (uint32_t)static_a_n;
    tsa = make_tensor(uniq_name(static_a_name ? static_a_name : "sa"), QNN_TENSOR_TYPE_STATIC, 1, sd_a,
                      sa.data(), (uint32_t)sa.size(), dt);
    if (api->tensorCreateGraphTensor(graph, &tsa) != QNN_SUCCESS) return Status::Failed("static A");
    inputs.push_back(tsa);
  }
  if (static_b && static_b_n) {
    sb = pack_f32(static_b, static_b_n, dt);
    sd_b[0] = (uint32_t)static_b_n;
    tsb = make_tensor(uniq_name(static_b_name ? static_b_name : "sb"), QNN_TENSOR_TYPE_STATIC, 1, sd_b,
                      sb.data(), (uint32_t)sb.size(), dt);
    if (api->tensorCreateGraphTensor(graph, &tsb) != QNN_SUCCESS) return Status::Failed("static B");
    inputs.push_back(tsb);
  }

  // Norm ops: epsilon scalar + axes tensor (INT32 — HTP is picky vs UINT32).
  Qnn_Param_t params[2]{};
  uint32_t nparams = 0;
  int32_t axes_data[1] = {rank > 0 ? (int32_t)rank - 1 : 0};
  uint32_t axes_dims[1] = {1};
  Qnn_Tensor_t axes_t{};
  if (epsilon >= 0.f) {
    params[nparams] = QNN_PARAM_INIT;
    params[nparams].paramType = QNN_PARAMTYPE_SCALAR;
    params[nparams].name = (std::strcmp(opname, QNN_OP_RMS_NORM) == 0)
                               ? QNN_OP_RMS_NORM_PARAM_EPSILON
                               : QNN_OP_LAYER_NORM_PARAM_EPSILON;
    params[nparams].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
    params[nparams].scalarParam.floatValue = epsilon;
    ++nparams;
  }
  if (with_axes) {
    axes_t = QNN_TENSOR_INIT;
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
    params[nparams] = QNN_PARAM_INIT;
    params[nparams].paramType = QNN_PARAMTYPE_TENSOR;
    params[nparams].name = (std::strcmp(opname, QNN_OP_RMS_NORM) == 0) ? QNN_OP_RMS_NORM_PARAM_AXES
                                                                      : QNN_OP_LAYER_NORM_PARAM_AXES;
    params[nparams].tensorParam = axes_t;
    ++nparams;
  }

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = uniq_name("node");
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = opname;
  op.v1.numOfParams = nparams;
  op.v1.params = nparams ? params : nullptr;
  op.v1.numOfInputs = (uint32_t)inputs.size();
  op.v1.inputTensors = inputs.data();
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &tout;

  if (api->graphAddNode(graph, op) != QNN_SUCCESS)
    return Status::Failed(std::string("addNode ") + opname);
  const int64_t t1 = mono_ns();
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
  const int64_t t2 = mono_ns();

  Qnn_Tensor_t ein = tin, eout = tout;
  ein.v1.clientBuf.data = in_buf.data();
  ein.v1.clientBuf.dataSize = (uint32_t)in_buf.size();
  eout.v1.clientBuf.data = out_buf.data();
  eout.v1.clientBuf.dataSize = (uint32_t)out_buf.size();
  if (api->graphExecute(graph, &ein, 1, &eout, 1, nullptr, nullptr) != QNN_SUCCESS)
    return Status::Failed("execute");
  const int64_t t3 = mono_ns();
  unpack_to_f32(out_buf.data(), n, dt, out);
  mark_phase(sess, t1 - t0, t2 - t1, t3 - t2);
  return Status::Ok();
}

inline Status run_binary(Session& sess, const char* gname, const char* opname, size_t n,
                         const float* a, const float* b, float* y, DType dt) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");
  const int64_t t0 = mono_ns();
  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), uniq_name(gname), nullptr, &graph) != QNN_SUCCESS)
    return Status::Failed("graphCreate");

  uint32_t dims[1] = {(uint32_t)n};
  auto ab = pack_f32(a, n, dt);
  auto bb = pack_f32(b, n, dt);
  std::vector<uint8_t> yb(n * dtype_nbytes(dt), 0);
  Qnn_Tensor_t ta =
      make_tensor("a", QNN_TENSOR_TYPE_APP_WRITE, 1, dims, ab.data(), (uint32_t)ab.size(), dt);
  Qnn_Tensor_t tb =
      make_tensor("b", QNN_TENSOR_TYPE_APP_WRITE, 1, dims, bb.data(), (uint32_t)bb.size(), dt);
  Qnn_Tensor_t ty =
      make_tensor("y", QNN_TENSOR_TYPE_APP_READ, 1, dims, yb.data(), (uint32_t)yb.size(), dt);
  if (api->tensorCreateGraphTensor(graph, &ta) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tb) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &ty) != QNN_SUCCESS)
    return Status::Failed("tensor");

  Qnn_Tensor_t ins[2] = {ta, tb};
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "bin0";
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = opname;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = ins;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &ty;
  if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed("addNode");
  const int64_t t1 = mono_ns();
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
  const int64_t t2 = mono_ns();
  Qnn_Tensor_t ea = ta, eb = tb, ey = ty;
  ea.v1.clientBuf.data = ab.data();
  ea.v1.clientBuf.dataSize = (uint32_t)ab.size();
  eb.v1.clientBuf.data = bb.data();
  eb.v1.clientBuf.dataSize = (uint32_t)bb.size();
  ey.v1.clientBuf.data = yb.data();
  ey.v1.clientBuf.dataSize = (uint32_t)yb.size();
  Qnn_Tensor_t eins[2] = {ea, eb};
  if (api->graphExecute(graph, eins, 2, &ey, 1, nullptr, nullptr) != QNN_SUCCESS)
    return Status::Failed("execute");
  const int64_t t3 = mono_ns();
  unpack_to_f32(yb.data(), n, dt, y);
  mark_phase(sess, t1 - t0, t2 - t1, t3 - t2);
  return Status::Ok();
}

inline Qnn_Param_t make_u32_tensor_param(const char* name, uint32_t* data, uint32_t rank,
                                         uint32_t* dims, Qnn_Tensor_t* storage) {
  *storage = QNN_TENSOR_INIT;
  storage->version = QNN_TENSOR_VERSION_1;
  storage->v1 = QNN_TENSOR_V1_INIT;
  storage->v1.name = name;
  storage->v1.type = QNN_TENSOR_TYPE_STATIC;
  storage->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  storage->v1.dataType = QNN_DATATYPE_UINT_32;
  storage->v1.rank = rank;
  storage->v1.dimensions = dims;
  storage->v1.memType = QNN_TENSORMEMTYPE_RAW;
  uint32_t n = 1;
  for (uint32_t i = 0; i < rank; ++i) n *= dims[i];
  storage->v1.clientBuf.data = data;
  storage->v1.clientBuf.dataSize = n * sizeof(uint32_t);

  Qnn_Param_t p = QNN_PARAM_INIT;
  p.paramType = QNN_PARAMTYPE_TENSOR;
  p.name = name;
  p.tensorParam = *storage;
  return p;
}

inline Status run_conv2d(Session& sess, int32_t n, int32_t h, int32_t w, int32_t cin, int32_t cout,
                         int32_t kh, int32_t kw, int32_t sh, int32_t sw, int32_t ph, int32_t pw,
                         const float* weight, const float* x, float* y, DType dt,
                         const char* opname = QNN_OP_CONV_2D) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");
  const int64_t t0 = mono_ns();
  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), uniq_name("wq_conv2d"), nullptr, &graph) != QNN_SUCCESS)
    return Status::Failed("graphCreate");

  const int32_t oh = (h + 2 * ph - kh) / sh + 1;
  const int32_t ow = (w + 2 * pw - kw) / sw + 1;
  if (oh <= 0 || ow <= 0) return Status::Failed("bad conv geometry");

  uint32_t in_dims[4] = {(uint32_t)n, (uint32_t)h, (uint32_t)w, (uint32_t)cin};
  uint32_t w_dims[4] = {(uint32_t)cout, (uint32_t)kh, (uint32_t)kw, (uint32_t)cin};
  uint32_t out_dims[4] = {(uint32_t)n, (uint32_t)oh, (uint32_t)ow, (uint32_t)cout};
  size_t n_in = (size_t)n * h * w * cin;
  size_t n_w = (size_t)cout * kh * kw * cin;
  size_t n_out = (size_t)n * oh * ow * cout;

  auto in_buf = pack_f32(x, n_in, dt);
  auto w_buf = pack_f32(weight, n_w, dt);
  std::vector<uint8_t> out_buf(n_out * dtype_nbytes(dt), 0);

  Qnn_Tensor_t tin = make_tensor("x", QNN_TENSOR_TYPE_APP_WRITE, 4, in_dims, in_buf.data(),
                                 (uint32_t)in_buf.size(), dt);
  Qnn_Tensor_t tw = make_tensor("w", QNN_TENSOR_TYPE_STATIC, 4, w_dims, w_buf.data(),
                                (uint32_t)w_buf.size(), dt);
  Qnn_Tensor_t tout = make_tensor("y", QNN_TENSOR_TYPE_APP_READ, 4, out_dims, out_buf.data(),
                                  (uint32_t)out_buf.size(), dt);
  if (api->tensorCreateGraphTensor(graph, &tin) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tw) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tout) != QNN_SUCCESS)
    return Status::Failed("tensor");

  // HTP requires param tensors registered via tensorCreateGraphTensor first.
  const char* stride_name = (std::strcmp(opname, QNN_OP_TRANSPOSE_CONV_2D) == 0)
                                ? QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE
                                : QNN_OP_CONV_2D_PARAM_STRIDE;
  const char* pad_name = (std::strcmp(opname, QNN_OP_TRANSPOSE_CONV_2D) == 0)
                             ? QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT
                             : QNN_OP_CONV_2D_PARAM_PAD_AMOUNT;
  uint32_t stride_data[2] = {(uint32_t)sh, (uint32_t)sw};
  uint32_t stride_dims[1] = {2};
  uint32_t pad_data[4] = {(uint32_t)ph, (uint32_t)ph, (uint32_t)pw, (uint32_t)pw};
  uint32_t pad_dims[2] = {2, 2};
  uint32_t dil_data[2] = {1, 1};
  uint32_t dil_dims[1] = {2};
  Qnn_Tensor_t stride_t{}, pad_t{}, dil_t{};
  Qnn_Param_t params[4];
  params[0] = make_u32_tensor_param(uniq_name("stride"), stride_data, 1, stride_dims, &stride_t);
  params[0].name = stride_name;
  params[1] = make_u32_tensor_param(uniq_name("pad"), pad_data, 2, pad_dims, &pad_t);
  params[1].name = pad_name;
  params[2] = make_u32_tensor_param(uniq_name("dil"), dil_data, 1, dil_dims, &dil_t);
  params[2].name = (std::strcmp(opname, QNN_OP_TRANSPOSE_CONV_2D) == 0)
                       ? "dilation"
                       : QNN_OP_CONV_2D_PARAM_DILATION;
  if (api->tensorCreateGraphTensor(graph, &stride_t) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &pad_t) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &dil_t) != QNN_SUCCESS)
    return Status::Failed("conv param tensors");
  params[0].tensorParam = stride_t;
  params[1].tensorParam = pad_t;
  params[2].tensorParam = dil_t;
  params[3] = QNN_PARAM_INIT;
  params[3].paramType = QNN_PARAMTYPE_SCALAR;
  params[3].name = (std::strcmp(opname, QNN_OP_TRANSPOSE_CONV_2D) == 0)
                       ? QNN_OP_TRANSPOSE_CONV_2D_PARAM_GROUP
                       : QNN_OP_CONV_2D_PARAM_GROUP;
  params[3].scalarParam.dataType = QNN_DATATYPE_UINT_32;
  params[3].scalarParam.uint32Value = 1;

  Qnn_Tensor_t ins[2] = {tin, tw};
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = uniq_name("conv");
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = opname;
  op.v1.numOfParams = 4;
  op.v1.params = params;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = ins;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &tout;
  if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed(std::string("addNode ") + opname);
  const int64_t t1 = mono_ns();
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
  const int64_t t2 = mono_ns();

  Qnn_Tensor_t ein = tin, eout = tout;
  ein.v1.clientBuf.data = in_buf.data();
  ein.v1.clientBuf.dataSize = (uint32_t)in_buf.size();
  eout.v1.clientBuf.data = out_buf.data();
  eout.v1.clientBuf.dataSize = (uint32_t)out_buf.size();
  if (api->graphExecute(graph, &ein, 1, &eout, 1, nullptr, nullptr) != QNN_SUCCESS)
    return Status::Failed("execute");
  const int64_t t3 = mono_ns();
  unpack_to_f32(out_buf.data(), n_out, dt, y);
  mark_phase(sess, t1 - t0, t2 - t1, t3 - t2);
  return Status::Ok();
}

inline Status run_conv3d(Session& sess, int32_t n, int32_t d, int32_t h, int32_t w, int32_t cin,
                         int32_t cout, int32_t kd, int32_t kh, int32_t kw, const float* weight,
                         const float* x, float* y, DType dt, const char* opname = QNN_OP_CONV_3D) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");
  const int64_t t0 = mono_ns();
  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), uniq_name("wq_conv3d"), nullptr, &graph) != QNN_SUCCESS)
    return Status::Failed("graphCreate");

  uint32_t in_dims[5] = {(uint32_t)n, (uint32_t)d, (uint32_t)h, (uint32_t)w, (uint32_t)cin};
  uint32_t w_dims[5] = {(uint32_t)cout, (uint32_t)kd, (uint32_t)kh, (uint32_t)kw, (uint32_t)cin};
  uint32_t out_dims[5] = {(uint32_t)n, (uint32_t)d, (uint32_t)h, (uint32_t)w, (uint32_t)cout};
  size_t n_in = (size_t)n * d * h * w * cin;
  size_t n_w = (size_t)cout * kd * kh * kw * cin;
  size_t n_out = (size_t)n * d * h * w * cout;
  auto in_buf = pack_f32(x, n_in, dt);
  auto w_buf = pack_f32(weight, n_w, dt);
  std::vector<uint8_t> out_buf(n_out * dtype_nbytes(dt), 0);
  Qnn_Tensor_t tin = make_tensor("x", QNN_TENSOR_TYPE_APP_WRITE, 5, in_dims, in_buf.data(),
                                 (uint32_t)in_buf.size(), dt);
  Qnn_Tensor_t tw = make_tensor("w", QNN_TENSOR_TYPE_STATIC, 5, w_dims, w_buf.data(),
                                (uint32_t)w_buf.size(), dt);
  Qnn_Tensor_t tout = make_tensor("y", QNN_TENSOR_TYPE_APP_READ, 5, out_dims, out_buf.data(),
                                  (uint32_t)out_buf.size(), dt);
  if (api->tensorCreateGraphTensor(graph, &tin) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tw) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tout) != QNN_SUCCESS)
    return Status::Failed("tensor");

  const bool tconv3 = (std::strcmp(opname, QNN_OP_TRANSPOSE_CONV_3D) == 0);
  uint32_t stride_data[3] = {1, 1, 1};
  uint32_t stride_dims[1] = {3};
  uint32_t pad_data[6] = {0, 0, 0, 0, 0, 0};
  uint32_t pad_dims[2] = {3, 2};
  uint32_t dil_data[3] = {1, 1, 1};
  uint32_t dil_dims[1] = {3};
  Qnn_Tensor_t stride_t{}, pad_t{}, dil_t{};
  Qnn_Param_t params[4];
  params[0] = make_u32_tensor_param(uniq_name("stride3"), stride_data, 1, stride_dims, &stride_t);
  params[0].name = tconv3 ? QNN_OP_TRANSPOSE_CONV_3D_PARAM_STRIDE : QNN_OP_CONV_3D_PARAM_STRIDE;
  params[1] = make_u32_tensor_param(uniq_name("pad3"), pad_data, 2, pad_dims, &pad_t);
  params[1].name = tconv3 ? QNN_OP_TRANSPOSE_CONV_3D_PARAM_PAD_AMOUNT : QNN_OP_CONV_3D_PARAM_PAD_AMOUNT;
  params[2] = make_u32_tensor_param(uniq_name("dil3"), dil_data, 1, dil_dims, &dil_t);
  params[2].name = tconv3 ? QNN_OP_TRANSPOSE_CONV_3D_PARAM_DILATION : QNN_OP_CONV_3D_PARAM_DILATION;
  if (api->tensorCreateGraphTensor(graph, &stride_t) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &pad_t) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &dil_t) != QNN_SUCCESS)
    return Status::Failed("conv3d param tensors");
  params[0].tensorParam = stride_t;
  params[1].tensorParam = pad_t;
  params[2].tensorParam = dil_t;
  params[3] = QNN_PARAM_INIT;
  params[3].paramType = QNN_PARAMTYPE_SCALAR;
  params[3].name = tconv3 ? QNN_OP_TRANSPOSE_CONV_3D_PARAM_GROUP : QNN_OP_CONV_3D_PARAM_GROUP;
  params[3].scalarParam.dataType = QNN_DATATYPE_UINT_32;
  params[3].scalarParam.uint32Value = 1;

  Qnn_Tensor_t ins[2] = {tin, tw};
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = uniq_name("conv3");
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = opname;
  op.v1.numOfParams = 4;
  op.v1.params = params;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = ins;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &tout;
  if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed("addNode conv3d");
  const int64_t t1 = mono_ns();
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
  const int64_t t2 = mono_ns();
  Qnn_Tensor_t ein = tin, eout = tout;
  ein.v1.clientBuf.data = in_buf.data();
  ein.v1.clientBuf.dataSize = (uint32_t)in_buf.size();
  eout.v1.clientBuf.data = out_buf.data();
  eout.v1.clientBuf.dataSize = (uint32_t)out_buf.size();
  if (api->graphExecute(graph, &ein, 1, &eout, 1, nullptr, nullptr) != QNN_SUCCESS)
    return Status::Failed("execute");
  const int64_t t3 = mono_ns();
  unpack_to_f32(out_buf.data(), n_out, dt, y);
  mark_phase(sess, t1 - t0, t2 - t1, t3 - t2);
  return Status::Ok();
}

inline Status run_gather(Session& sess, int32_t vocab, int32_t dim, const float* table,
                         int32_t n, const int32_t* ids, float* y, DType dt) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");
  const int64_t t0 = mono_ns();
  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), uniq_name("wq_embed"), nullptr, &graph) != QNN_SUCCESS)
    return Status::Failed("graphCreate");

  uint32_t tab_dims[2] = {(uint32_t)vocab, (uint32_t)dim};
  uint32_t id_dims[1] = {(uint32_t)n};
  uint32_t out_dims[2] = {(uint32_t)n, (uint32_t)dim};
  auto tab = pack_f32(table, (size_t)vocab * dim, dt);
  std::vector<int32_t> id_buf(ids, ids + n);
  std::vector<uint8_t> out_buf((size_t)n * dim * dtype_nbytes(dt), 0);

  Qnn_Tensor_t ttab =
      make_tensor("table", QNN_TENSOR_TYPE_STATIC, 2, tab_dims, tab.data(), (uint32_t)tab.size(), dt);
  Qnn_Tensor_t tids = QNN_TENSOR_INIT;
  tids.version = QNN_TENSOR_VERSION_1;
  tids.v1 = QNN_TENSOR_V1_INIT;
  tids.v1.name = "ids";
  tids.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  tids.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tids.v1.dataType = QNN_DATATYPE_INT_32;
  tids.v1.rank = 1;
  tids.v1.dimensions = id_dims;
  tids.v1.memType = QNN_TENSORMEMTYPE_RAW;
  // HTP: APP_WRITE buffer bound at execute, not create.
  tids.v1.clientBuf.data = nullptr;
  tids.v1.clientBuf.dataSize = 0;
  Qnn_Tensor_t tout = make_tensor("y", QNN_TENSOR_TYPE_APP_READ, 2, out_dims, out_buf.data(),
                                  (uint32_t)out_buf.size(), dt);
  if (api->tensorCreateGraphTensor(graph, &ttab) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tids) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tout) != QNN_SUCCESS)
    return Status::Failed("tensor");

  Qnn_Param_t axis = QNN_PARAM_INIT;
  axis.paramType = QNN_PARAMTYPE_SCALAR;
  axis.name = QNN_OP_GATHER_PARAM_AXIS;
  axis.scalarParam.dataType = QNN_DATATYPE_UINT_32;
  axis.scalarParam.uint32Value = 0;

  Qnn_Tensor_t ins[2] = {ttab, tids};
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "gather0";
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_GATHER;
  op.v1.numOfParams = 1;
  op.v1.params = &axis;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = ins;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &tout;
  if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed("addNode gather");
  const int64_t t1 = mono_ns();
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
  const int64_t t2 = mono_ns();
  Qnn_Tensor_t eid = tids, eout = tout;
  eid.v1.clientBuf.data = id_buf.data();
  eid.v1.clientBuf.dataSize = (uint32_t)(n * sizeof(int32_t));
  eout.v1.clientBuf.data = out_buf.data();
  eout.v1.clientBuf.dataSize = (uint32_t)out_buf.size();
  if (api->graphExecute(graph, &eid, 1, &eout, 1, nullptr, nullptr) != QNN_SUCCESS)
    return Status::Failed("execute");
  const int64_t t3 = mono_ns();
  unpack_to_f32(out_buf.data(), (size_t)n * dim, dt, y);
  mark_phase(sess, t1 - t0, t2 - t1, t3 - t2);
  return Status::Ok();
}

// LSTM / MHA / SwiGLU: compose via FC+elementwise for blank smoke (native Lstm when possible).
inline Status run_lstm_native(Session& sess, int32_t seq, int32_t batch, int32_t input_size,
                              int32_t hidden, const float* w_in, const float* w_rec, const float* b,
                              const float* x, float* y, DType dt) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");
  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), uniq_name("wq_lstm"), nullptr, &graph) != QNN_SUCCESS)
    return Status::Failed("graphCreate");

  // QNN Lstm expects packed weights; blank zeros are enough for graph compose.
  uint32_t x_dims[3] = {(uint32_t)seq, (uint32_t)batch, (uint32_t)input_size};
  uint32_t y_dims[3] = {(uint32_t)seq, (uint32_t)batch, (uint32_t)hidden};
  // Weight shapes vary by SDK; use [4*H, I] and [4*H, H] FC-style packs as STATIC inputs.
  uint32_t wi_dims[2] = {(uint32_t)(4 * hidden), (uint32_t)input_size};
  uint32_t wr_dims[2] = {(uint32_t)(4 * hidden), (uint32_t)hidden};
  uint32_t b_dims[1] = {(uint32_t)(4 * hidden)};
  size_t nx = (size_t)seq * batch * input_size;
  size_t ny = (size_t)seq * batch * hidden;
  size_t nwi = (size_t)4 * hidden * input_size;
  size_t nwr = (size_t)4 * hidden * hidden;
  size_t nb = (size_t)4 * hidden;

  auto xb = pack_f32(x, nx, dt);
  auto wib = pack_f32(w_in, nwi, dt);
  auto wrb = pack_f32(w_rec, nwr, dt);
  auto bb = pack_f32(b, nb, dt);
  std::vector<uint8_t> yb(ny * dtype_nbytes(dt), 0);

  Qnn_Tensor_t tx =
      make_tensor("x", QNN_TENSOR_TYPE_APP_WRITE, 3, x_dims, xb.data(), (uint32_t)xb.size(), dt);
  Qnn_Tensor_t twi =
      make_tensor("wi", QNN_TENSOR_TYPE_STATIC, 2, wi_dims, wib.data(), (uint32_t)wib.size(), dt);
  Qnn_Tensor_t twr =
      make_tensor("wr", QNN_TENSOR_TYPE_STATIC, 2, wr_dims, wrb.data(), (uint32_t)wrb.size(), dt);
  Qnn_Tensor_t tb =
      make_tensor("b", QNN_TENSOR_TYPE_STATIC, 1, b_dims, bb.data(), (uint32_t)bb.size(), dt);
  Qnn_Tensor_t ty =
      make_tensor("y", QNN_TENSOR_TYPE_APP_READ, 3, y_dims, yb.data(), (uint32_t)yb.size(), dt);
  if (api->tensorCreateGraphTensor(graph, &tx) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &twi) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &twr) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tb) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &ty) != QNN_SUCCESS)
    return Status::Failed("tensor");

  Qnn_Tensor_t ins[4] = {tx, twi, twr, tb};
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "lstm0";
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_LSTM;
  op.v1.numOfInputs = 4;
  op.v1.inputTensors = ins;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &ty;
  if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed("addNode lstm");
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
  Qnn_Tensor_t ex = tx, ey = ty;
  ex.v1.clientBuf.data = xb.data();
  ex.v1.clientBuf.dataSize = (uint32_t)xb.size();
  ey.v1.clientBuf.data = yb.data();
  ey.v1.clientBuf.dataSize = (uint32_t)yb.size();
  if (api->graphExecute(graph, &ex, 1, &ey, 1, nullptr, nullptr) != QNN_SUCCESS)
    return Status::Failed("execute");
  unpack_to_f32(yb.data(), ny, dt, y);
  return Status::Ok();
}

} // namespace welvet_qnn::graph

#endif
