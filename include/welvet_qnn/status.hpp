#pragma once

#include <string>

namespace welvet_qnn {

enum class Code : int {
  Ok = 0,
  Unavailable = 1, // SDK / DLL / NPU missing
  Failed = 2,      // load or graph error
  Unsupported = 3,
};

struct Status {
  Code code = Code::Ok;
  std::string message;

  explicit operator bool() const { return code == Code::Ok; }

  static Status Ok() { return {}; }
  static Status Unavailable(std::string m) {
    return {Code::Unavailable, std::move(m)};
  }
  static Status Failed(std::string m) { return {Code::Failed, std::move(m)}; }
  static Status Unsupported(std::string m) {
    return {Code::Unsupported, std::move(m)};
  }
};

} // namespace welvet_qnn
