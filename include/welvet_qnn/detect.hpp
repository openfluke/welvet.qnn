#pragma once

#include <string>
#include <vector>

#include "welvet_qnn/status.hpp"

namespace welvet_qnn {

enum class BackendKind {
  Unknown = 0,
  Cpu = 1,
  Gpu = 2,
  Htp = 3, // Hexagon NPU
  Saver = 4,
  Ir = 5,
};

struct BackendInfo {
  BackendKind kind = BackendKind::Unknown;
  std::string name;     // e.g. "htp"
  std::string path;     // full path to shared library if found
  bool library_found = false;
  bool providers_ok = false; // LoadLibrary + getProviders succeeded
  uint32_t backend_id = 0;
  uint32_t api_major = 0;
  uint32_t api_minor = 0;
  std::string build_id;
  std::string note;
};

struct DetectResult {
  bool sdk_found = false;
  std::string sdk_root;   // QAIRT_SDK_ROOT / QNN_SDK_ROOT
  std::string lib_dir;    // platform lib folder
  std::string profile;    // e.g. windows_arm64, linux_aarch64
  std::vector<BackendInfo> backends;
  std::vector<std::string> hints;

  // True if QnnHtp library is present (NPU software stack visible).
  bool htp_library() const;
  // True if getProviders succeeded on HTP (still may fail backendCreate).
  bool htp_providers() const;
};

// Filesystem + light DLL probe (getProviders / getBuildId). Does not call backendCreate.
DetectResult detect();

const char* backend_kind_name(BackendKind k);

} // namespace welvet_qnn
