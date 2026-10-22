#include "welvet_qnn/session.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <utility>
#include <vector>

#include "platform.hpp"
#include "welvet_qnn/detect.hpp"

#if !defined(_WIN32)
#include <cstdlib>
#endif

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
#include "QnnInterface.h"
#endif

namespace welvet_qnn {

Session::~Session() {
#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK
  auto* api = reinterpret_cast<QNN_INTERFACE_VER_TYPE*>(api_);
  if (api) {
    if (context_ && api->contextFree) api->contextFree(context_, nullptr);
    if (device_ && api->deviceFree) api->deviceFree(device_);
    if (backend_ && api->backendFree) api->backendFree(backend_);
  }
#endif
  if (api_) {
    operator delete(api_);
    api_ = nullptr;
  }
  if (lib_) {
    plat::dl_close(lib_);
    lib_ = nullptr;
  }
  ok_ = false;
}

Session::Session(Session&& o) noexcept { *this = std::move(o); }

Session& Session::operator=(Session&& o) noexcept {
  if (this == &o) return *this;
  Session tmp;
  tmp.lib_ = o.lib_;
  tmp.api_ = o.api_;
  tmp.backend_ = o.backend_;
  tmp.device_ = o.device_;
  tmp.context_ = o.context_;
  tmp.kind_ = o.kind_;
  tmp.path_ = std::move(o.path_);
  tmp.build_id_ = std::move(o.build_id_);
  tmp.ok_ = o.ok_;
  o.lib_ = o.api_ = o.backend_ = o.device_ = o.context_ = nullptr;
  o.ok_ = false;

  std::swap(lib_, tmp.lib_);
  std::swap(api_, tmp.api_);
  std::swap(backend_, tmp.backend_);
  std::swap(device_, tmp.device_);
  std::swap(context_, tmp.context_);
  std::swap(kind_, tmp.kind_);
  std::swap(path_, tmp.path_);
  std::swap(build_id_, tmp.build_id_);
  std::swap(ok_, tmp.ok_);
  return *this;
}

#if defined(WELVET_QNN_HAS_SDK) && WELVET_QNN_HAS_SDK

namespace {

const char* dll_name(BackendKind k) {
#if defined(_WIN32)
  switch (k) {
    case BackendKind::Htp:
      return "QnnHtp.dll";
    case BackendKind::Cpu:
      return "QnnCpu.dll";
    case BackendKind::Gpu:
      return "QnnGpu.dll";
    case BackendKind::Saver:
      return "QnnSaver.dll";
    case BackendKind::Ir:
      return "QnnIr.dll";
    default:
      return nullptr;
  }
#else
  switch (k) {
    case BackendKind::Htp:
      return "libQnnHtp.so";
    case BackendKind::Cpu:
      return "libQnnCpu.so";
    case BackendKind::Gpu:
      return "libQnnGpu.so";
    case BackendKind::Saver:
      return "libQnnSaver.so";
    case BackendKind::Ir:
      return "libQnnIr.so";
    default:
      return nullptr;
  }
#endif
}

} // namespace

namespace {

// Prepend Hexagon skel dirs so HTP can find libQnnHtpV*Skel.so (Windows DSP loader).
void ensure_adsp_library_path(const std::string& sdk_root) {
  if (sdk_root.empty()) return;
  if (!plat::getenv_str("ADSP_LIBRARY_PATH").empty() && !plat::env_truthy("WELVET_QNN_ADSP_FORCE"))
    return;
  namespace fs = std::filesystem;
  std::string joined;
  // Prefer V73 (Snapdragon X) then V81 then others that exist under SDK.
  const char* arches[] = {"hexagon-v73", "hexagon-v81", "hexagon-v79", "hexagon-v75",
                          "hexagon-v69", "hexagon-v68"};
  for (const char* a : arches) {
    fs::path p = fs::path(sdk_root) / "lib" / a / "unsigned";
    if (!fs::is_directory(p)) continue;
    if (!joined.empty()) joined += ";";
    joined += p.string();
  }
  if (joined.empty()) return;
#if defined(_WIN32)
  SetEnvironmentVariableA("ADSP_LIBRARY_PATH", joined.c_str());
#else
  setenv("ADSP_LIBRARY_PATH", joined.c_str(), 1);
#endif
}

} // namespace

Status Session::load_locked(BackendKind kind) {
  DetectResult det = detect();
  if (!det.sdk_found || det.lib_dir.empty()) {
    return Status::Unavailable("QAIRT SDK not detected (set QAIRT_SDK_ROOT)");
  }
  ensure_adsp_library_path(det.sdk_root);
  const char* name = dll_name(kind);
  if (!name) return Status::Unsupported("unknown backend kind");

  path_ = det.lib_dir;
#if defined(_WIN32)
  path_ += "\\";
  SetDllDirectoryA(det.lib_dir.c_str());
#else
  path_ += "/";
#endif
  path_ += name;

  // Gate dangerous backends (Cpu/Gpu/Htp) unless INPROC or Saver/Ir.
  const bool risky =
      kind == BackendKind::Htp || kind == BackendKind::Cpu || kind == BackendKind::Gpu;
  if (risky && !plat::env_truthy("WELVET_QNN_INPROC")) {
    auto pr = probe_backend_create(path_);
    if (!pr) return pr;
  }

  lib_ = plat::dl_open(path_.c_str());
  if (!lib_) return Status::Failed("dl_open failed: " + path_);

  using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  auto getProviders = (GetProvidersFn)plat::dl_sym(lib_, "QnnInterface_getProviders");
  if (!getProviders) return Status::Failed("missing getProviders");

  QnnInterface_t** providers = nullptr;
  uint32_t n = 0;
  if (getProviders((const QnnInterface_t***)&providers, &n) != QNN_SUCCESS || !providers || !n) {
    return Status::Failed("getProviders failed");
  }

  QNN_INTERFACE_VER_TYPE* api = new QNN_INTERFACE_VER_TYPE{};
  bool found = false;
  for (uint32_t i = 0; i < n; ++i) {
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        QNN_API_VERSION_MINOR <= providers[i]->apiVersion.coreApiVersion.minor) {
      *api = providers[i]->QNN_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found || !api->backendCreate) {
    delete api;
    return Status::Failed("no matching QNN interface");
  }
  api_ = api;

  if (api->backendGetBuildId) {
    const char* bid = nullptr;
    if (api->backendGetBuildId(&bid) == QNN_SUCCESS && bid) build_id_ = bid;
  }

  Qnn_ErrorHandle_t e = api->backendCreate(nullptr, nullptr, &backend_);
  if (e != QNN_SUCCESS || !backend_) {
    return Status::Failed("backendCreate failed");
  }
  if (api->deviceCreate) {
    (void)api->deviceCreate(nullptr, nullptr, &device_);
  }
  e = api->contextCreate(backend_, device_, nullptr, &context_);
  if (e != QNN_SUCCESS || !context_) {
    return Status::Failed("contextCreate failed");
  }

  kind_ = kind;
  ok_ = true;
  return Status::Ok();
}

Status Session::open(BackendKind kind, Session* out) {
  if (!out) return Status::Failed("nil out");
  Session s;
  auto st = s.load_locked(kind);
  if (!st) return st;
  *out = std::move(s);
  return Status::Ok();
}

Status Session::open_default(Session* out) {
  const auto pref = plat::getenv_str("WELVET_QNN_BACKEND");
  if (!pref.empty()) {
    if (pref[0] == 'h' || pref[0] == 'H') return open(BackendKind::Htp, out);
    if (pref[0] == 'c' || pref[0] == 'C') return open(BackendKind::Cpu, out);
    if (pref[0] == 's' || pref[0] == 'S') return open(BackendKind::Saver, out);
    if (pref[0] == 'g' || pref[0] == 'G') return open(BackendKind::Gpu, out);
  }
  // Prefer NPU, then CPU, then Saver (ABI smoke).
  Status last = Status::Unavailable("no backend");
  for (BackendKind k : {BackendKind::Htp, BackendKind::Cpu, BackendKind::Saver}) {
    last = open(k, out);
    if (last) return last;
  }
  return last;
}

Status probe_backend_create(const std::string& backend_dll_path) {
#if defined(_WIN32)
  // Prefer sibling welvet_qnn_probe.exe next to this process, else PATH.
  char exe[MAX_PATH]{};
  GetModuleFileNameA(nullptr, exe, MAX_PATH);
  std::string dir(exe);
  auto slash = dir.find_last_of("\\/");
  std::string probe = (slash == std::string::npos) ? "welvet_qnn_probe.exe"
                                                   : dir.substr(0, slash + 1) + "welvet_qnn_probe.exe";
  if (GetFileAttributesA(probe.c_str()) == INVALID_FILE_ATTRIBUTES) {
    probe = "welvet_qnn_probe.exe";
  }
  std::string cmd = "\"" + probe + "\" \"" + backend_dll_path + "\"";
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  std::vector<char> buf(cmd.begin(), cmd.end());
  buf.push_back('\0');
  if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
                      nullptr, &si, &pi)) {
    return Status::Failed(
        "welvet_qnn_probe.exe missing — build tools/probe, or set WELVET_QNN_INPROC=1");
  }
  WaitForSingleObject(pi.hProcess, 60000);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  if (code == 0) return Status::Ok();
  char msg[192];
  std::snprintf(msg, sizeof(msg),
                "backendCreate probe exit=0x%lX (QnnCpu/Htp/Gpu often STACK_OVERFLOW on some "
                "Snapdragon hosts)",
                static_cast<unsigned long>(code));
  return Status::Failed(msg);
#else
  (void)backend_dll_path;
  // Non-Windows: skip subprocess gate (caller may set WELVET_QNN_INPROC).
  return Status::Ok();
#endif
}

#else // !HAS_SDK

Status Session::load_locked(BackendKind) {
  return Status::Unavailable("built without QAIRT SDK headers");
}
Status Session::open(BackendKind, Session*) {
  return Status::Unavailable("built without QAIRT SDK headers");
}
Status Session::open_default(Session*) {
  return Status::Unavailable("built without QAIRT SDK headers");
}
Status probe_backend_create(const std::string&) {
  return Status::Unavailable("built without QAIRT SDK headers");
}

#endif

} // namespace welvet_qnn
