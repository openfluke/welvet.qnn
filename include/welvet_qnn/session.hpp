#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "welvet_qnn/detect.hpp"
#include "welvet_qnn/status.hpp"

namespace welvet_qnn {

// Last HTP graph phase costs (filled by graph_util / prepared execute).
struct PhaseTiming {
  int64_t create_ns = 0;   // graphCreate + tensorCreate + addNode
  int64_t finalize_ns = 0; // graphFinalize (compile)
  int64_t execute_ns = 0;  // graphExecute only
};

// Live QNN backend + context. Forward-only graphs.
class Session {
 public:
  Session() = default;
  ~Session();

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;

  // Prefer HTP (NPU), then CPU. Override with WELVET_QNN_BACKEND=htp|cpu|saver.
  static Status open_default(Session* out);
  static Status open(BackendKind kind, Session* out);

  bool ready() const { return ok_; }
  BackendKind backend() const { return kind_; }
  const std::string& backend_path() const { return path_; }
  const std::string& build_id() const { return build_id_; }

  // Opaque QNN handles for layer graphs (valid while Session lives).
  void* backend_handle() const { return backend_; }
  void* context_handle() const { return context_; }
  void* api() const { return api_; } // QNN_INTERFACE_VER_TYPE*

  PhaseTiming last_phase{};

 private:
  Status load_locked(BackendKind kind);

  void* lib_ = nullptr;
  void* api_ = nullptr; // owned copy on heap of interface table
  void* backend_ = nullptr;
  void* device_ = nullptr;
  void* context_ = nullptr;
  BackendKind kind_ = BackendKind::Unknown;
  std::string path_;
  std::string build_id_;
  bool ok_ = false;
};

// Child-process safe check: can backendCreate succeed for this DLL?
// On Windows, STACK_OVERFLOW in-process must not kill the caller — use this first.
Status probe_backend_create(const std::string& backend_dll_path);

} // namespace welvet_qnn
