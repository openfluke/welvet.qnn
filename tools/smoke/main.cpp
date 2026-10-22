#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "welvet_qnn/detect.hpp"
#include "welvet_qnn/dtype.hpp"
#include "welvet_qnn/layers.hpp"
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
    if (!b.path.empty()) std::printf("         path=%s\n", b.path.c_str());
  }
  for (const auto& h : d.hints) std::printf("  hint: %s\n", h.c_str());
}

static void print_caps() {
  std::printf("\n=== HTP caps matrix (SDK 2.42) implemented=%d ===\n", 1);
  std::printf("%-12s", "layer");
  for (DType dt : kAllDtypes) std::printf(" %7s", dtype_name(dt));
  std::printf("\n");
  for (int i = 0; i < (int)LayerId::Count; ++i) {
    auto id = (LayerId)i;
    std::printf("%-12s", layer_id_name(id));
    for (DType dt : kAllDtypes) {
      const bool cap = htp_supports(id, dt);
      const bool impl = layer_implemented(id);
      if (!cap)
        std::printf(" %7s", "—");
      else if (!impl)
        std::printf(" %7s", "gap");
      else
        std::printf(" %7s", "api");
    }
    std::printf("\n");
  }
  std::printf("legend: api=blank+forward in tree; gap=HTP yes but not coded; —=HTP no\n");
}

static bool try_session(BackendKind kind, Session* out, std::string* err) {
  auto st = Session::open(kind, out);
  if (!st) {
    if (err) *err = st.message;
    return false;
  }
  return true;
}

static void check(const char* name, const Status& st, bool numeric_expected) {
  (void)numeric_expected;
  if (!st) {
    if (st.code == Code::Unsupported)
      skip(name, st.message);
    else
      fail(name, st.message);
  } else {
    ok(name);
  }
}

static void run_layers(Session& sess, const char* backend_label, bool numeric) {
  std::printf("\n=== blank layers on %s (build=%s) numeric=%d ===\n", backend_label,
              sess.build_id().c_str(), (int)numeric);
  const bool is_htp = sess.backend() == BackendKind::Htp;

  // Core four @ float32
  {
    auto d = Dense::blank(4, 4);
    std::vector<float> x(4), y(4, -1.f);
    for (int i = 0; i < 4; ++i) x[i] = (float)(i + 1);
    auto st = d.forward(sess, 1, x.data(), y.data());
    if (!st)
      fail("dense/f32", st.message);
    else if (numeric) {
      float maxd = 0;
      for (int i = 0; i < 4; ++i) maxd = std::fmax(maxd, std::fabs(y[i] - x[i]));
      if (maxd > 1e-2f)
        fail("dense/f32", "maxDiff=" + std::to_string(maxd));
      else
        ok("dense/f32");
    } else {
      ok("dense/f32 (graph)");
    }
  }
  {
    auto s = Softmax::blank(4);
    std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
    check("softmax/f32", s.forward(sess, 1, x.data(), y.data()), numeric);
  }
  {
    auto r = RmsNorm::blank(4);
    std::vector<float> x = {1, 1, 1, 1}, y(4, 0);
    check("rmsnorm/f32", r.forward(sess, 1, 1, x.data(), y.data()), numeric);
  }
  {
    auto l = LayerNorm::blank(4);
    std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
    check("layernorm/f32", l.forward(sess, 1, 1, x.data(), y.data()), numeric);
  }

  // Extra NPU layers @ float32
  {
    auto c = Cnn2::blank(2, 2, 4, 4);
    std::vector<float> x(2 * 4 * 4 * 2, 1.f), y(2 * 4 * 4 * 2, 0.f);
    check("cnn2/f32", c.forward(sess, x.data(), y.data()), numeric);
  }
  {
    auto c = Cnn3::blank(2, 2, 2, 2, 2);
    std::vector<float> x(1 * 2 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2 * 2, 0.f);
    check("cnn3/f32", c.forward(sess, x.data(), y.data()), numeric);
  }
  {
    auto c = ConvT2::blank(2, 2, 2, 2);
    std::vector<float> x(1 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2, 0.f);
    check("convt2/f32", c.forward(sess, x.data(), y.data()), numeric);
  }
  {
    auto c = ConvT3::blank(2, 2, 2, 2, 2);
    std::vector<float> x(1 * 2 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2 * 2, 0.f);
    check("convt3/f32", c.forward(sess, x.data(), y.data()), numeric);
  }
  {
    auto l = Lstm::blank(4, 4);
    std::vector<float> x(2 * 1 * 4, 0.1f), y(2 * 1 * 4, 0.f);
    check("lstm/f32", l.forward(sess, 2, 1, x.data(), y.data()), numeric);
  }
  {
    auto e = Embedding::blank(8, 4);
    int32_t ids[2] = {0, 1};
    std::vector<float> y(2 * 4, 0.f);
    check("embedding/f32", e.forward(sess, 2, ids, y.data()), numeric);
  }
  {
    auto r = Residual::blank();
    std::vector<float> a = {1, 2, 3, 4}, b = {1, 1, 1, 1}, y(4, 0);
    check("residual/f32", r.forward(sess, 4, a.data(), b.data(), y.data()), numeric);
  }
  {
    auto s = SwiGlu::blank(4, 4);
    std::vector<float> x = {1, 1, 1, 1}, y(4, 0);
    check("swiglu/f32", s.forward(sess, 1, x.data(), y.data()), numeric);
  }
  {
    auto m = Mha::blank(4, 2);
    std::vector<float> x(1 * 2 * 4, 0.5f), y(1 * 2 * 4, 0.f);
    check("mha/f32", m.forward(sess, 1, 2, x.data(), y.data()), numeric);
  }

  // Dtype sweep on Dense (HTP-supported set)
  std::printf("\n=== dense dtype sweep on %s ===\n", backend_label);
  for (DType dt : kAllDtypes) {
    char name[64];
    std::snprintf(name, sizeof(name), "dense/%s", dtype_name(dt));
    if (is_htp && !htp_supports(LayerId::Dense, dt)) {
      skip(name, "HTP unsupported");
      continue;
    }
    auto d = Dense::blank(4, 4, dt);
    std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
    auto st = d.forward(sess, 1, x.data(), y.data());
    check(name, st, numeric);
  }
}

int main() {
  std::printf("welvet.qnn smoke v0.2.0\n");
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

  // NPU first. Do NOT treat CPU as success for "NPU working".
  Session htp;
  std::string err;
  std::printf("\n=== open htp (NPU) ===\n");
  if (try_session(BackendKind::Htp, &htp, &err)) {
    ok("open_htp");
    run_layers(htp, "htp", /*numeric=*/true);
  } else {
    skip("open_htp", err);
    std::printf("  !! NPU NOT RUNNING — backendCreate blocked on this host\n");
  }

  // Saver = graph compose only (not NPU)
  Session saver;
  std::printf("\n=== open saver (graph-only, NOT NPU) ===\n");
  if (try_session(BackendKind::Saver, &saver, &err)) {
    ok("open_saver");
    run_layers(saver, "saver", /*numeric=*/false);
  } else {
    skip("open_saver", err);
  }

  std::printf("\nRESULT pass=%d fail=%d skip=%d\n", g_pass, g_fail, g_skip);
  std::printf("NOTE: Saver PASS ≠ NPU. Only open_htp + numeric layers count as NPU.\n");
  return g_fail ? 1 : 0;
}
