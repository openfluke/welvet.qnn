#include "welvet_qnn/sequential.hpp"

#include <cstring>
#include <type_traits>
#include <utility>

namespace welvet_qnn {

Status Sequential::push_checked(LayerOp op, int32_t expect_in, int32_t produce_out) {
  if (expect_in <= 0 || produce_out <= 0) return Status::Failed("sequential: bad child shape");
  if (ops_.empty()) {
    in_features_ = expect_in;
  } else if (expect_in != out_features_) {
    return Status::Failed("sequential: child in=" + std::to_string(expect_in) +
                          " != prev out=" + std::to_string(out_features_));
  }
  ops_.push_back(std::move(op));
  out_features_ = produce_out;
  return Status::Ok();
}

Status Sequential::push(Dense d) {
  const int32_t in = d.in, out = d.out;
  return push_checked(LayerOp{std::move(d)}, in, out);
}

Status Sequential::push(Softmax s) {
  const int32_t c = s.cols;
  return push_checked(LayerOp{std::move(s)}, c, c);
}

Status Sequential::push(RmsNorm r) {
  const int32_t d = r.dim;
  return push_checked(LayerOp{std::move(r)}, d, d);
}

Status Sequential::push(LayerNorm l) {
  const int32_t d = l.dim;
  return push_checked(LayerOp{std::move(l)}, d, d);
}

Status Sequential::push(SwiGlu s) {
  const int32_t in = s.in;
  return push_checked(LayerOp{std::move(s)}, in, in);
}

Status Sequential::push(Mha m) {
  const int32_t d = m.dim;
  return push_checked(LayerOp{std::move(m)}, d, d);
}

Status Sequential::push(Lstm lstm) {
  const int32_t in = lstm.input_size, out = lstm.hidden_size;
  return push_checked(LayerOp{std::move(lstm)}, in, out);
}

void Sequential::clear() {
  ops_.clear();
  in_features_ = 0;
  out_features_ = 0;
}

Sequential Sequential::dense_tower(int32_t dim, int32_t depth, DType dt) {
  Sequential seq;
  for (int32_t i = 0; i < depth; ++i) {
    (void)seq.push(Dense::blank(dim, dim, dt));
  }
  return seq;
}

Status Sequential::forward_one(Session& sess, const LayerOp& op, int32_t batch, int32_t seq,
                               const float* x, float* y) const {
  return std::visit(
      [&](const auto& layer) -> Status {
        using T = std::decay_t<decltype(layer)>;
        if constexpr (std::is_same_v<T, Dense>) {
          return layer.forward(sess, batch * seq, x, y);
        } else if constexpr (std::is_same_v<T, Softmax>) {
          return layer.forward(sess, batch * seq, x, y);
        } else if constexpr (std::is_same_v<T, RmsNorm>) {
          return layer.forward(sess, batch, seq, x, y);
        } else if constexpr (std::is_same_v<T, LayerNorm>) {
          return layer.forward(sess, batch, seq, x, y);
        } else if constexpr (std::is_same_v<T, SwiGlu>) {
          return layer.forward(sess, batch * seq, x, y);
        } else if constexpr (std::is_same_v<T, Mha>) {
          return layer.forward(sess, batch, seq, x, y);
        } else if constexpr (std::is_same_v<T, Lstm>) {
          // x laid out [seq, batch, in] for Lstm API
          return layer.forward(sess, seq, batch, x, y);
        } else {
          return Status::Unsupported("sequential: unknown op");
        }
      },
      op);
}

Status Sequential::forward(Session& sess, int32_t batch, const float* x, float* y,
                           int32_t seq) const {
  if (!sess.ready() || !x || !y || batch <= 0 || seq <= 0)
    return Status::Failed("sequential: bad args / session");
  if (ops_.empty()) return Status::Failed("sequential: empty");

  // Ping-pong buffers sized for max features along the path.
  int32_t max_f = in_features_;
  for (const auto& op : ops_) {
    std::visit(
        [&](const auto& layer) {
          using T = std::decay_t<decltype(layer)>;
          if constexpr (std::is_same_v<T, Dense>)
            max_f = max_f > layer.out ? max_f : layer.out;
          else if constexpr (std::is_same_v<T, Lstm>)
            max_f = max_f > layer.hidden_size ? max_f : layer.hidden_size;
          else if constexpr (std::is_same_v<T, Softmax>)
            max_f = max_f > layer.cols ? max_f : layer.cols;
          else if constexpr (std::is_same_v<T, RmsNorm> || std::is_same_v<T, LayerNorm>)
            max_f = max_f > layer.dim ? max_f : layer.dim;
          else if constexpr (std::is_same_v<T, SwiGlu>)
            max_f = max_f > layer.in ? max_f : layer.in;
          else if constexpr (std::is_same_v<T, Mha>)
            max_f = max_f > layer.dim ? max_f : layer.dim;
        },
        op);
  }

  const size_t tokens = (size_t)batch * (size_t)seq;
  std::vector<float> buf0(tokens * (size_t)max_f), buf1(tokens * (size_t)max_f);
  std::memcpy(buf0.data(), x, tokens * (size_t)in_features_ * sizeof(float));

  float* cur = buf0.data();
  float* nxt = buf1.data();

  for (size_t i = 0; i < ops_.size(); ++i) {
    auto st = forward_one(sess, ops_[i], batch, seq, cur, nxt);
    if (!st) {
      st.message = "sequential[" + std::to_string(i) + "]: " + st.message;
      return st;
    }
    std::swap(cur, nxt);
  }

  std::memcpy(y, cur, tokens * (size_t)out_features_ * sizeof(float));
  return Status::Ok();
}

} // namespace welvet_qnn
