#pragma once

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK

#include <cstdint>
#include <vector>

#include "QnnInterface.h"
#include "graph_util.hpp"
#include "welvet_qnn/dtype.hpp"
#include "welvet_qnn/session.hpp"
#include "welvet_qnn/status.hpp"

namespace welvet_qnn {

// Prepared Dense: Sync/prepare bakes W into STATIC; execute reuses HTP graph.
// Bias is applied on host after execute (same as Dense::forward).
struct DensePrepared {
  Qnn_GraphHandle_t graph = nullptr;
  Qnn_Tensor_t tin{};
  Qnn_Tensor_t tout{};
  int32_t batch = 0, in = 0, out = 0;
  DType dt = DType::Float32;
  std::vector<uint8_t> w_storage;
  std::vector<float> bias;
  uint32_t in_dims[2]{};
  uint32_t out_dims[2]{};
  bool ready = false;

  void reset() {
    graph = nullptr;
    ready = false;
    w_storage.clear();
    bias.clear();
  }

  Status prepare(Session& sess, int32_t batch_, int32_t in_, int32_t out_, const float* W,
                 const float* bias_in, DType dt_ = DType::Float32) {
    reset();
    auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
    if (!api || !sess.context_handle()) return Status::Failed("session not ready");
    batch = batch_;
    in = in_;
    out = out_;
    dt = dt_;
    if (bias_in)
      bias.assign(bias_in, bias_in + out);
    else
      bias.assign((size_t)out, 0.f);

    const int64_t t0 = graph::mono_ns();
    if (api->graphCreate(sess.context_handle(), graph::uniq_name("wq_dense_p"), nullptr, &graph) !=
        QNN_SUCCESS)
      return Status::Failed("graphCreate");

    in_dims[0] = (uint32_t)batch;
    in_dims[1] = (uint32_t)in;
    uint32_t w_dims[2] = {(uint32_t)out, (uint32_t)in};
    out_dims[0] = (uint32_t)batch;
    out_dims[1] = (uint32_t)out;
    const size_t n_w = (size_t)out * (size_t)in;
    const size_t n_out = (size_t)batch * (size_t)out;

    w_storage = graph::pack_f32(W, n_w, dt);
    std::vector<uint8_t> scratch_in((size_t)batch * (size_t)in * dtype_nbytes(dt), 0);
    std::vector<uint8_t> scratch_out(n_out * dtype_nbytes(dt), 0);

    tin = graph::make_tensor(graph::uniq_name("x"), QNN_TENSOR_TYPE_APP_WRITE, 2, in_dims,
                             scratch_in.data(), (uint32_t)scratch_in.size(), dt);
    Qnn_Tensor_t tw = graph::make_tensor(graph::uniq_name("w"), QNN_TENSOR_TYPE_STATIC, 2, w_dims,
                                         w_storage.data(), (uint32_t)w_storage.size(), dt);
    tout = graph::make_tensor(graph::uniq_name("y"), QNN_TENSOR_TYPE_APP_READ, 2, out_dims,
                              scratch_out.data(), (uint32_t)scratch_out.size(), dt);

    if (api->tensorCreateGraphTensor(graph, &tin) != QNN_SUCCESS ||
        api->tensorCreateGraphTensor(graph, &tw) != QNN_SUCCESS ||
        api->tensorCreateGraphTensor(graph, &tout) != QNN_SUCCESS)
      return Status::Failed("tensorCreateGraphTensor");

    Qnn_Tensor_t ins[2] = {tin, tw};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = graph::uniq_name("fc0");
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_FULLY_CONNECTED;
    op.v1.numOfParams = 0;
    op.v1.params = nullptr;
    op.v1.numOfInputs = 2;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &tout;

    if (api->graphAddNode(graph, op) != QNN_SUCCESS) return Status::Failed("addNode FC");
    const int64_t t1 = graph::mono_ns();
    if (api->graphFinalize(graph, nullptr, nullptr) != QNN_SUCCESS) return Status::Failed("finalize");
    const int64_t t2 = graph::mono_ns();
    graph::mark_phase(sess, t1 - t0, t2 - t1, 0);
    ready = true;
    return Status::Ok();
  }

  Status execute(Session& sess, const float* x, float* y) {
    if (!ready || !graph) return Status::Failed("dense not prepared");
    auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(sess.api());
    if (!api) return Status::Failed("no api");
    const size_t n_in = (size_t)batch * (size_t)in;
    const size_t n_out = (size_t)batch * (size_t)out;
    auto in_buf = graph::pack_f32(x, n_in, dt);
    std::vector<uint8_t> out_buf(n_out * dtype_nbytes(dt), 0);

    const int64_t t0 = graph::mono_ns();
    Qnn_Tensor_t ein = tin, eout = tout;
    ein.v1.clientBuf.data = in_buf.data();
    ein.v1.clientBuf.dataSize = (uint32_t)in_buf.size();
    eout.v1.clientBuf.data = out_buf.data();
    eout.v1.clientBuf.dataSize = (uint32_t)out_buf.size();
    if (api->graphExecute(graph, &ein, 1, &eout, 1, nullptr, nullptr) != QNN_SUCCESS)
      return Status::Failed("execute");
    const int64_t t1 = graph::mono_ns();
    graph::unpack_to_f32(out_buf.data(), n_out, dt, y);
    for (int32_t b = 0; b < batch; ++b)
      for (int32_t o = 0; o < out; ++o) y[b * out + o] += bias[(size_t)o];
    // Preserve create/finalize from prepare; overwrite execute.
    sess.last_phase.execute_ns = t1 - t0;
    return Status::Ok();
  }

  bool matches(int32_t batch_, int32_t in_, int32_t out_) const {
    return ready && batch == batch_ && in == in_ && out == out_;
  }
};

} // namespace welvet_qnn

#endif
