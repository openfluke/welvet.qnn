#include "welvet_qnn/detect.hpp"

#include <cstdlib>
#include <filesystem>
#include <utility>
#include <vector>

#include "platform.hpp"

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
#include "QnnInterface.h"
#endif

namespace welvet_qnn {
namespace {

namespace fs = std::filesystem;

std::string sdk_root() {
  auto a = plat::getenv_str("QAIRT_SDK_ROOT");
  if (!a.empty()) return a;
  return plat::getenv_str("QNN_SDK_ROOT");
}

std::pair<std::string, std::string> lib_candidates() {
#if defined(_WIN32)
#  if defined(_M_ARM64) || defined(__aarch64__)
  return {"windows_arm64", "aarch64-windows-msvc"};
#  else
  return {"windows_amd64", "x86_64-windows-msvc"};
#  endif
#elif defined(__APPLE__)
  return {"darwin", "aarch64-apple"}; // rarely shipped; detect will miss libs
#elif defined(__aarch64__)
  return {"linux_aarch64", "aarch64-ubuntu-gcc9.4"};
#else
  return {"linux_x86_64", "x86_64-linux-clang"};
#endif
}

std::string find_lib_dir(const std::string& root) {
  if (root.empty()) return {};
  auto [profile, primary] = lib_candidates();
  (void)profile;
  std::vector<std::string> cands = {primary};
#if defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
  cands.push_back("arm64x-windows-msvc");
#endif
  for (const auto& rel : cands) {
    fs::path p = fs::path(root) / "lib" / rel;
    if (fs::is_directory(p)) return p.string();
  }
  return {};
}

#if defined(_WIN32)
const char* htp_name = "QnnHtp.dll";
const char* cpu_name = "QnnCpu.dll";
const char* gpu_name = "QnnGpu.dll";
const char* saver_name = "QnnSaver.dll";
const char* ir_name = "QnnIr.dll";
#else
const char* htp_name = "libQnnHtp.so";
const char* cpu_name = "libQnnCpu.so";
const char* gpu_name = "libQnnGpu.so";
const char* saver_name = "libQnnSaver.so";
const char* ir_name = "libQnnIr.so";
#endif

BackendInfo probe_lib(BackendKind kind, const std::string& lib_dir, const char* fname) {
  BackendInfo b;
  b.kind = kind;
  b.name = backend_kind_name(kind);
  fs::path path = lib_dir.empty() ? fs::path(fname) : (fs::path(lib_dir) / fname);
  b.path = path.string();
  if (!fs::exists(path)) {
    b.note = "library not found";
    return b;
  }
  b.library_found = true;

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
#if defined(_WIN32)
  if (!lib_dir.empty()) SetDllDirectoryA(lib_dir.c_str());
#endif
  void* lib = plat::dl_open(b.path.c_str());
  if (!lib) {
    b.note = "LoadLibrary/dlopen failed";
    return b;
  }
  using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  auto getProviders = (GetProvidersFn)plat::dl_sym(lib, "QnnInterface_getProviders");
  if (!getProviders) {
    b.note = "missing QnnInterface_getProviders";
    plat::dl_close(lib);
    return b;
  }
  QnnInterface_t** providers = nullptr;
  uint32_t n = 0;
  if (getProviders((const QnnInterface_t***)&providers, &n) != QNN_SUCCESS || !providers || !n) {
    b.note = "getProviders failed";
    plat::dl_close(lib);
    return b;
  }
  b.providers_ok = true;
  b.backend_id = providers[0]->backendId;
  b.api_major = providers[0]->apiVersion.coreApiVersion.major;
  b.api_minor = providers[0]->apiVersion.coreApiVersion.minor;
  auto api = providers[0]->QNN_INTERFACE_VER_NAME;
  if (api.backendGetBuildId) {
    const char* bid = nullptr;
    if (api.backendGetBuildId(&bid) == QNN_SUCCESS && bid) b.build_id = bid;
  }
  b.note = "getProviders ok";
  plat::dl_close(lib);
#else
  b.note = "built without QAIRT headers (WELVET_QNN_HAS_SDK=0); library present only";
#endif
  return b;
}

} // namespace

const char* backend_kind_name(BackendKind k) {
  switch (k) {
    case BackendKind::Cpu:
      return "cpu";
    case BackendKind::Gpu:
      return "gpu";
    case BackendKind::Htp:
      return "htp";
    case BackendKind::Saver:
      return "saver";
    case BackendKind::Ir:
      return "ir";
    default:
      return "unknown";
  }
}

bool DetectResult::htp_library() const {
  for (const auto& b : backends) {
    if (b.kind == BackendKind::Htp && b.library_found) return true;
  }
  return false;
}

bool DetectResult::htp_providers() const {
  for (const auto& b : backends) {
    if (b.kind == BackendKind::Htp && b.providers_ok) return true;
  }
  return false;
}

DetectResult detect() {
  DetectResult r;
  auto [profile, primary] = lib_candidates();
  r.profile = profile;
  (void)primary;

  r.sdk_root = sdk_root();
  if (r.sdk_root.empty()) {
    r.hints.push_back("set QAIRT_SDK_ROOT or QNN_SDK_ROOT to the QAIRT install root");
    return r;
  }
  if (!fs::is_directory(r.sdk_root)) {
    r.hints.push_back("SDK root is not a directory: " + r.sdk_root);
    return r;
  }
  r.sdk_found = true;
  r.lib_dir = find_lib_dir(r.sdk_root);
  if (r.lib_dir.empty()) {
    r.hints.push_back("no platform lib/ folder under SDK (tried aarch64-windows-msvc / arm64x / …)");
    return r;
  }

  r.backends.push_back(probe_lib(BackendKind::Htp, r.lib_dir, htp_name));
  r.backends.push_back(probe_lib(BackendKind::Cpu, r.lib_dir, cpu_name));
  r.backends.push_back(probe_lib(BackendKind::Gpu, r.lib_dir, gpu_name));
  r.backends.push_back(probe_lib(BackendKind::Saver, r.lib_dir, saver_name));
  r.backends.push_back(probe_lib(BackendKind::Ir, r.lib_dir, ir_name));

  if (r.htp_library()) {
    r.hints.push_back("QnnHtp present — Hexagon NPU software stack visible");
  } else {
    r.hints.push_back("QnnHtp not found — NPU backend DLL missing for this platform");
  }
#if !defined(WELVET_QNN_HAS_SDK) || !WELVET_QNN_HAS_SDK
  r.hints.push_back("rebuild with -DQAIRT_SDK_ROOT=... to enable getProviders / graphs");
#endif
  return r;
}

} // namespace welvet_qnn
