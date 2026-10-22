#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "welvet_qnn/detect.hpp"
#include "welvet_qnn/dtype.hpp"
#include "welvet_qnn/layers.hpp"
#include "welvet_qnn/sequential.hpp"
#include "welvet_qnn/session.hpp"

using namespace welvet_qnn;

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;

static void ok(const char* name) {
  std::printf("  PASS  %s\n", name);
  ++g_pass;
}
static void fail(const char* name, const std::string& why) {
  std::printf("  FAIL  %s — %s\n", name, why.c_str());
  ++g_fail;
}
static void skip(const char* name, const std::string& why) {
  std::printf("  SKIP  %s — %s\n", name, why.c_str());
  ++g_skip;
}

static void print_detect(const DetectResult& d) {
  std::printf("\n=== detect ===\n");
  std::printf("profile=%s sdk_found=%d\n", d.profile.c_str(), (int)d.sdk_found);
  std::printf("sdk_root=%s\n", d.sdk_root.c_str());
  std::printf("lib_dir=%s\n", d.lib_dir.c_str());
  std::printf("htp_library=%d htp_providers=%d\n", (int)d.htp_library(), (int)d.htp_providers());
  for (const auto& b : d.backends) {
    std::printf("  %-6s found=%d providers=%d id=%u api=%u.%u %s\n", b.name.c_str(),
                (int)b.library_found, (int)b.providers_ok, b.backend_id, b.api_major, b.api_minor,
                b.note.c_str());
    if (!b.build_id.empty()) std::printf("         build=%s\n", b.build_id.c_str());
  }
  for (const auto& h : d.hints) std::printf("  hint: %s\n", h.c_str());
}

static void print_caps() {
  std::printf("\n=== HTP caps (Welvet.qnn) ===\n");
  std::printf("%-12s", "layer");
  for (DType dt : kAllDtypes) std::printf(" %7s", dtype_name(dt));
  std::printf("\n");
  for (int i = 0; i < (int)LayerId::Count; ++i) {
    auto id = (LayerId)i;
    std::printf("%-12s", layer_id_name(id));
    for (DType dt : kAllDtypes) {
      if (!htp_supports(id, dt))
        std::printf(" %7s", "—");
      else
        std::printf(" %7s", layer_implemented(id) ? "api" : "gap");
    }
    std::printf("\n");
  }
}

static bool try_session(BackendKind kind, Session* out, std::string* err) {
  auto st = Session::open(kind, out);
  if (!st) {
    if (err) *err = st.message;
    return false;
  }
  return true;
}

static void check(const char* name, const Status& st) {
  if (!st) {
    if (st.code == Code::Unsupported)
      skip(name, st.message);
    else
      fail(name, st.message);
  } else {
    ok(name);
  }
}

static void run_one_dtype(Session& sess, LayerId id, DType dt, bool is_htp, bool* poison) {
  char name[80];
  std::snprintf(name, sizeof(name), "%s/%s", layer_id_name(id), dtype_name(dt));
  if (is_htp && !htp_supports(id, dt)) {
    skip(name, "HTP unsupported");
    return;
  }
  Status st = Status::Failed("unhandled");
  switch (id) {
    case LayerId::Dense: {
      auto d = Dense::blank(4, 4, dt);
      std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
      st = d.forward(sess, 1, x.data(), y.data());
      break;
    }
    case LayerId::Softmax: {
      auto s = Softmax::blank(4, dt);
      std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
      st = s.forward(sess, 1, x.data(), y.data());
      break;
    }
    case LayerId::RmsNorm: {
      auto r = RmsNorm::blank(4, 1e-5f, dt);
      std::vector<float> x = {1, 1, 1, 1}, y(4, 0);
      st = r.forward(sess, 1, 1, x.data(), y.data());
      break;
    }
    case LayerId::LayerNorm: {
      auto l = LayerNorm::blank(4, 1e-5f, dt);
      std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
      st = l.forward(sess, 1, 1, x.data(), y.data());
      break;
    }
    case LayerId::Cnn2: {
      auto c = Cnn2::blank(2, 2, 4, 4, dt);
      std::vector<float> x(2 * 4 * 4 * 2, 1.f), y(2 * 4 * 4 * 2, 0.f);
      st = c.forward(sess, x.data(), y.data());
      break;
    }
    case LayerId::Cnn3: {
      auto c = Cnn3::blank(2, 2, 2, 2, 2, dt);
      std::vector<float> x(1 * 2 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2 * 2, 0.f);
      st = c.forward(sess, x.data(), y.data());
      break;
    }
    case LayerId::ConvT2: {
      auto c = ConvT2::blank(2, 2, 2, 2, dt);
      std::vector<float> x(1 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2, 0.f);
      st = c.forward(sess, x.data(), y.data());
      break;
    }
    case LayerId::ConvT3: {
      auto c = ConvT3::blank(2, 2, 2, 2, 2, dt);
      std::vector<float> x(1 * 2 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2 * 2, 0.f);
      st = c.forward(sess, x.data(), y.data());
      break;
    }
    case LayerId::Lstm: {
      auto l = Lstm::blank(4, 4, dt);
      std::vector<float> x(2 * 1 * 4, 0.1f), y(2 * 1 * 4, 0.f);
      st = l.forward(sess, 2, 1, x.data(), y.data());
      break;
    }
    case LayerId::Embedding: {
      auto e = Embedding::blank(8, 4, dt);
      int32_t ids[2] = {0, 1};
      std::vector<float> y(2 * 4, 0.f);
      st = e.forward(sess, 2, ids, y.data());
      break;
    }
    case LayerId::Residual: {
      auto r = Residual::blank(dt);
      std::vector<float> a = {1, 2, 3, 4}, b = {1, 1, 1, 1}, y(4, 0);
      st = r.forward(sess, 4, a.data(), b.data(), y.data());
      break;
    }
    case LayerId::SwiGlu: {
      auto s = SwiGlu::blank(4, 4, dt);
      std::vector<float> x = {1, 1, 1, 1}, y(4, 0);
      st = s.forward(sess, 1, x.data(), y.data());
      break;
    }
    case LayerId::Mha: {
      auto m = Mha::blank(4, 2, dt);
      std::vector<float> x(1 * 2 * 4, 0.5f), y(1 * 2 * 4, 0.f);
      st = m.forward(sess, 1, 2, x.data(), y.data());
      break;
    }
    default:
      st = Status::Unsupported("no smoke");
      break;
  }
  check(name, st);
  // HTP HNRD can poison the context after a bad finalize/execute — caller reopens.
  if (poison && is_htp && !st && st.code == Code::Failed) *poison = true;
}

static void run_layers_htp() {
  std::printf("\n=== layer×dtype on htp (NPU) ===\n");
  for (int i = 0; i < (int)LayerId::Count; ++i) {
    auto id = (LayerId)i;
    if (!layer_implemented(id)) continue;
    Session sess;
    std::string err;
    if (!try_session(BackendKind::Htp, &sess, &err)) {
      skip(layer_id_name(id), std::string("reopen htp: ") + err);
      continue;
    }
    for (DType dt : kAllDtypes) {
      bool poison = false;
      run_one_dtype(sess, id, dt, true, &poison);
      if (poison) {
        // Fresh context for remaining dtypes of this layer.
        sess = Session{};
        if (!try_session(BackendKind::Htp, &sess, &err)) {
          skip(layer_id_name(id), std::string("reopen after poison: ") + err);
          break;
        }
      }
    }
  }
}

static void run_layers_saver() {
  Session sess;
  std::string err;
  if (!try_session(BackendKind::Saver, &sess, &err)) {
    skip("open_saver", err);
    return;
  }
  ok("open_saver");
  std::printf("\n=== layer×dtype on saver (NOT NPU) build=%s ===\n", sess.build_id().c_str());
  for (int i = 0; i < (int)LayerId::Count; ++i) {
    auto id = (LayerId)i;
    if (!layer_implemented(id)) continue;
    for (DType dt : kAllDtypes) run_one_dtype(sess, id, dt, false, nullptr);
  }
}

int main() {
  std::printf("welvet.qnn smoke v0.2.1 (Welvet Sequential + packs)\n");
  std::printf("supported layers:");
  for (const char** p = supported_layers; *p; ++p) std::printf(" %s", *p);
  std::printf("\n");
  print_caps();

  auto det = detect();
  print_detect(det);

  if (!det.sdk_found) {
    skip("session", "SDK not found");
    std::printf("\nRESULT pass=%d fail=%d skip=%d\n", g_pass, g_fail, g_skip);
    return g_fail ? 1 : 0;
  }

  if (det.htp_library())
    ok("detect_htp_library");
  else
    fail("detect_htp_library", "QnnHtp not found");
  if (det.htp_providers())
    ok("detect_htp_providers");
  else
    skip("detect_htp_providers", "getProviders not available");

  auto run_sequential_smoke = [&](BackendKind kind, const char* label) {
    Session sess;
    std::string err;
    std::printf("\n=== sequential network on %s ===\n", label);
    if (!try_session(kind, &sess, &err)) {
      skip((std::string("sequential_") + label).c_str(), err);
      return;
    }
    // Welvet-style: Dense → RmsNorm → Dense → Softmax
    Sequential net;
    auto st = net.push(Dense::blank(4, 8));
    if (st) st = net.push(RmsNorm::blank(8));
    if (st) st = net.push(Dense::blank(8, 4));
    if (st) st = net.push(Softmax::blank(4));
    if (!st) {
      fail((std::string("sequential_build_") + label).c_str(), st.message);
      return;
    }
    std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
    st = net.forward(sess, /*batch=*/1, x.data(), y.data(), /*seq=*/1);
    check((std::string("sequential_mlp_") + label).c_str(), st);

    // Dense tower (like sequential.New depth×dim)
    auto tower = Sequential::dense_tower(4, 3);
    x = {0.5f, 0.5f, 0.5f, 0.5f};
    y.assign(4, 0);
    st = tower.forward(sess, 1, x.data(), y.data(), 1);
    check((std::string("sequential_tower_") + label).c_str(), st);

    // Shape mismatch must fail
    Sequential bad;
    (void)bad.push(Dense::blank(4, 4));
    st = bad.push(Dense::blank(8, 4)); // in 8 != prev out 4
    if (!st)
      ok((std::string("sequential_reject_bad_link_") + label).c_str());
    else
      fail((std::string("sequential_reject_bad_link_") + label).c_str(), "expected reject");
  };

  {
    Session probe;
    std::string err;
    std::printf("\n=== open htp (NPU) ===\n");
    if (try_session(BackendKind::Htp, &probe, &err)) {
      ok("open_htp");
      run_layers_htp();
      run_sequential_smoke(BackendKind::Htp, "htp");
    } else {
      skip("open_htp", err);
      std::printf("  !! NPU NOT RUNNING\n");
    }
  }

  std::printf("\n=== open saver (graph-only, NOT NPU) ===\n");
  run_layers_saver();
  run_sequential_smoke(BackendKind::Saver, "saver");

  std::printf("\nRESULT pass=%d fail=%d skip=%d\n", g_pass, g_fail, g_skip);
  std::printf("NOTE: only HTP rows with PASS are live NPU.\n");
  return g_fail ? 1 : 0;
}
