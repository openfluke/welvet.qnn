#pragma once

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnOpDef.h"
#include "welvet_qnn/session.hpp"
#include "welvet_qnn/status.hpp"

namespace welvet_qnn::graph {

inline Qnn_Tensor_t make_tensor(const char* name, Qnn_TensorType_t type, uint32_t rank,
                                uint32_t* dims, void* data, uint32_t nbytes) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = QNN_TENSOR_VERSION_1;
  t.v1 = QNN_TENSOR_V1_INIT;
  t.v1.name = name;
  t.v1.type = type;
  t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType = QNN_DATATYPE_FLOAT_32;
  t.v1.rank = rank;
  t.v1.dimensions = dims;
  t.v1.memType = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf.data = data;
  t.v1.clientBuf.dataSize = nbytes;
  return t;
}

inline Status run_fc(Session& sess, int32_t batch, int32_t in, int32_t out, const float* W,
                     const float* x, float* y) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");

  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), "wq_dense", nullptr, &graph) != QNN_SUCCESS) {
    return Status::Failed("graphCreate");
  }

  uint32_t in_dims[2] = {(uint32_t)batch, (uint32_t)in};
  uint32_t w_dims[2] = {(uint32_t)out, (uint32_t)in};
  uint32_t out_dims[2] = {(uint32_t)batch, (uint32_t)out};
  const size_t n_in = (size_t)batch * in;
  const size_t n_out = (size_t)batch * out;
  const size_t n_w = (size_t)out * in;

  std::vector<float> in_buf(x, x + n_in);
  std::vector<float> w_buf(W, W + n_w);
  std::vector<float> out_buf(n_out, 0.f);

  Qnn_Tensor_t tin =
      make_tensor("x", QNN_TENSOR_TYPE_APP_WRITE, 2, in_dims, in_buf.data(),
                  (uint32_t)(n_in * sizeof(float)));
  Qnn_Tensor_t tw =
      make_tensor("w", QNN_TENSOR_TYPE_STATIC, 2, w_dims, w_buf.data(), (uint32_t)(n_w * sizeof(float)));
  Qnn_Tensor_t tout =
      make_tensor("y", QNN_TENSOR_TYPE_APP_READ, 2, out_dims, out_buf.data(),
                  (uint32_t)(n_out * sizeof(float)));

  if (api->tensorCreateGraphTensor(graph, &tin) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tw) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tout) != QNN_SUCCESS) {
    return Status::Failed("tensorCreateGraphTensor");
  }

  Qnn_Tensor_t ins[2] = {tin, tw};
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "fc0";
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_FULLY_CONNECTED;
  op.v1.numOfParams = 0;
  op.v1.params = nullptr;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = ins;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &tout;

  if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed("addNode FC");
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");

  Qnn_Tensor_t ein = tin, eout = tout;
  ein.v1.clientBuf.data = in_buf.data();
  ein.v1.clientBuf.dataSize = (uint32_t)(n_in * sizeof(float));
  eout.v1.clientBuf.data = out_buf.data();
  eout.v1.clientBuf.dataSize = (uint32_t)(n_out * sizeof(float));
  if (api->graphExecute(graph, &ein, 1, &eout, 1, nullptr, nullptr) != QNN_SUCCESS) {
    return Status::Failed("execute");
  }
  std::memcpy(y, out_buf.data(), n_out * sizeof(float));
  return Status::Ok();
}

inline Status run_unary(Session& sess, const char* gname, const char* opname, uint32_t rank,
                        uint32_t* dims, const float* in, float* out, size_t n,
                        const float* static_a, size_t static_a_n, const char* static_a_name,
                        const float* static_b, size_t static_b_n, const char* static_b_name) {
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
  if (!api || !sess.context_handle()) return Status::Failed("session not ready");

  Qnn_GraphHandle_t graph = nullptr;
  if (api->graphCreate(sess.context_handle(), gname, nullptr, &graph) != QNN_SUCCESS) {
    return Status::Failed("graphCreate");
  }

  std::vector<float> in_buf(in, in + n);
  std::vector<float> out_buf(n, 0.f);
  const uint32_t nbytes = (uint32_t)(n * sizeof(float));

  Qnn_Tensor_t tin =
      make_tensor("in", QNN_TENSOR_TYPE_APP_WRITE, rank, dims, in_buf.data(), nbytes);
  Qnn_Tensor_t tout =
      make_tensor("out", QNN_TENSOR_TYPE_APP_READ, rank, dims, out_buf.data(), nbytes);

  if (api->tensorCreateGraphTensor(graph, &tin) != QNN_SUCCESS ||
      api->tensorCreateGraphTensor(graph, &tout) != QNN_SUCCESS) {
    return Status::Failed("createGraphTensor");
  }

  std::vector<Qnn_Tensor_t> inputs{tin};
  std::vector<float> sa, sb;
  Qnn_Tensor_t tsa{}, tsb{};
  if (static_a && static_a_n) {
    sa.assign(static_a, static_a + static_a_n);
    uint32_t sd[1] = {(uint32_t)static_a_n};
    tsa = make_tensor(static_a_name, QNN_TENSOR_TYPE_STATIC, 1, sd, sa.data(),
                      (uint32_t)(sa.size() * sizeof(float)));
    if (api->tensorCreateGraphTensor(graph, &tsa) != QNN_SUCCESS) return Status::Failed("static A");
    inputs.push_back(tsa);
  }
  if (static_b && static_b_n) {
    sb.assign(static_b, static_b + static_b_n);
    uint32_t sd[1] = {(uint32_t)static_b_n};
    tsb = make_tensor(static_b_name, QNN_TENSOR_TYPE_STATIC, 1, sd, sb.data(),
                      (uint32_t)(sb.size() * sizeof(float)));
    if (api->tensorCreateGraphTensor(graph, &tsb) != QNN_SUCCESS) return Status::Failed("static B");
    inputs.push_back(tsb);
  }

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "node0";
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = opname;
  op.v1.numOfParams = 0;
  op.v1.params = nullptr;
  op.v1.numOfInputs = (uint32_t)inputs.size();
  op.v1.inputTensors = inputs.data();
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = &tout;

  if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed(std::string("addNode ") + opname);
  if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");

  Qnn_Tensor_t ein = tin, eout = tout;
  ein.v1.clientBuf.data = in_buf.data();
  ein.v1.clientBuf.dataSize = nbytes;
  eout.v1.clientBuf.data = out_buf.data();
  eout.v1.clientBuf.dataSize = nbytes;
  if (api->graphExecute(graph, &ein, 1, &eout, 1, nullptr, nullptr) != QNN_SUCCESS) {
    return Status::Failed("execute");
  }
  std::memcpy(out, out_buf.data(), nbytes);
  return Status::Ok();
}

} // namespace welvet_qnn::graph

#endif
