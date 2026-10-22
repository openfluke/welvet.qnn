#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "welvet_qnn/detect.hpp"
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

static bool try_session(BackendKind kind, Session* out, std::string* err) {
  auto st = Session::open(kind, out);
  if (!st) {
    if (err) *err = st.message;
    return false;
  }
  return true;
}

static void run_layers(Session& sess, const char* backend_label, bool numeric) {
  std::printf("\n=== blank layers on %s (build=%s) numeric=%d ===\n", backend_label,
              sess.build_id().c_str(), (int)numeric);

  // Dense
  {
    auto d = Dense::blank(4, 4);
    std::vector<float> x(4), y(4, -1.f);
    for (int i = 0; i < 4; ++i) x[i] = (float)(i + 1);
    auto st = d.forward(sess, 1, x.data(), y.data());
    if (!st) {
      fail("dense", st.message);
    } else if (numeric) {
      float maxd = 0;
      for (int i = 0; i < 4; ++i) maxd = std::fmax(maxd, std::fabs(y[i] - x[i]));
      if (maxd > 1e-2f)
        fail("dense", "maxDiff=" + std::to_string(maxd));
      else
        ok("dense");
    } else {
      ok("dense (graph ok, non-compute backend)");
    }
  }

  // Softmax
  {
    auto s = Softmax::blank(4);
    std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
    auto st = s.forward(sess, 1, x.data(), y.data());
    if (!st) {
      fail("softmax", st.message);
    } else if (numeric) {
      float sum = 0;
      for (float v : y) sum += v;
      if (std::fabs(sum - 1.f) > 1e-2f)
        fail("softmax", "sum=" + std::to_string(sum));
      else
        ok("softmax");
    } else {
      ok("softmax (graph ok, non-compute backend)");
    }
  }

  // RMSNorm
  {
    auto r = RmsNorm::blank(4);
    std::vector<float> x = {1, 1, 1, 1}, y(4, 0);
    auto st = r.forward(sess, 1, 1, x.data(), y.data());
    if (!st)
      fail("rmsnorm", st.message);
    else
      ok(numeric ? "rmsnorm" : "rmsnorm (graph ok, non-compute backend)");
  }

  // LayerNorm
  {
    auto l = LayerNorm::blank(4);
    std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
    auto st = l.forward(sess, 1, 1, x.data(), y.data());
    if (!st)
      fail("layernorm", st.message);
    else
      ok(numeric ? "layernorm" : "layernorm (graph ok, non-compute backend)");
  }
}

int main() {
  std::printf("welvet.qnn smoke\n");
  std::printf("supported layers:");
  for (const char** p = supported_layers; *p; ++p) std::printf(" %s", *p);
  std::printf("\n");

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
    skip("detect_htp_providers", "getProviders not available (no SDK build or load fail)");

  // Try backends in order: HTP (NPU), CPU, Saver
  struct Try {
    BackendKind k;
    const char* label;
  };
  Try tries[] = {{BackendKind::Htp, "htp"}, {BackendKind::Cpu, "cpu"}, {BackendKind::Saver, "saver"}};

  bool ran = false;
  for (auto t : tries) {
    Session sess;
    std::string err;
    std::printf("\n=== open %s ===\n", t.label);
    if (!try_session(t.k, &sess, &err)) {
      skip((std::string("open_") + t.label).c_str(), err);
      continue;
    }
    ok((std::string("open_") + t.label).c_str());
    const bool numeric = (t.k == BackendKind::Htp || t.k == BackendKind::Cpu);
    run_layers(sess, t.label, numeric);
    ran = true;
    // First backend that opens is enough for this smoke.
    break;
  }

  if (!ran) skip("layers", "no backend opened");

  std::printf("\nRESULT pass=%d fail=%d skip=%d\n", g_pass, g_fail, g_skip);
  return g_fail ? 1 : 0;
}
