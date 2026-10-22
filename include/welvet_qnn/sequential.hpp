#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "welvet_qnn/layers.hpp"
#include "welvet_qnn/session.hpp"
#include "welvet_qnn/status.hpp"

namespace welvet_qnn {

// Host tensor layout for chaining (Welvet Sequential style).
// Flat size = batch * seq * features.
struct TensorLayout {
  int32_t batch = 1;
  int32_t seq = 1;
  int32_t features = 0;

  int64_t numel() const {
    return (int64_t)batch * (int64_t)seq * (int64_t)features;
  }
};

// Heterogeneous child — same idea as Welvet Go sequential.Ops []any.
using LayerOp = std::variant<Dense, Softmax, RmsNorm, LayerNorm, SwiGlu, Mha, Lstm>;

// Sequential: dynamic list of connected layers. Forward runs child[0]→…→child[n]
// on the Session (each child still its own HTP graph today; fused graph = later).
class Sequential {
 public:
  Sequential() = default;

  // Append a child. Validates that its input features match the current tail
  // output (first child sets the network input size).
  Status push(Dense d);
  Status push(Softmax s);
  Status push(RmsNorm r);
  Status push(LayerNorm l);
  Status push(SwiGlu s);
  Status push(Mha m);
  Status push(Lstm lstm);

  void clear();
  size_t size() const { return ops_.size(); }
  bool empty() const { return ops_.empty(); }

  // Features into first / out of last child (0 if empty).
  int32_t in_features() const { return in_features_; }
  int32_t out_features() const { return out_features_; }

  const std::vector<LayerOp>& ops() const { return ops_; }

  // Run the whole stack. x/y are flat [batch * seq * features].
  // seq defaults to 1 (MLP). Mha/Lstm use batch+seq; Dense/Softmax/norms/SwiGlu
  // treat tokens = batch*seq.
  Status forward(Session& sess, int32_t batch, const float* x, float* y, int32_t seq = 1) const;

  // Convenience: square Dense tower like Welvet sequential.New (depth × dim).
  static Sequential dense_tower(int32_t dim, int32_t depth, DType dt = DType::Float32);

 private:
  Status push_checked(LayerOp op, int32_t expect_in, int32_t produce_out);
  Status forward_one(Session& sess, const LayerOp& op, int32_t batch, int32_t seq,
                     const float* x, float* y) const;

  std::vector<LayerOp> ops_;
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
};

} // namespace welvet_qnn
