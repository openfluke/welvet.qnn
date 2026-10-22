#pragma once

#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace welvet_qnn::plat {

inline void* dl_open(const char* path) {
#if defined(_WIN32)
  return (void*)LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
  return dlopen(path, RTLD_NOW | RTLD_GLOBAL);
#endif
}

inline void* dl_sym(void* h, const char* name) {
#if defined(_WIN32)
  return (void*)GetProcAddress((HMODULE)h, name);
#else
  return dlsym(h, name);
#endif
}

inline void dl_close(void* h) {
  if (!h) return;
#if defined(_WIN32)
  FreeLibrary((HMODULE)h);
#else
  dlclose(h);
#endif
}

inline std::string getenv_str(const char* k) {
  const char* v = std::getenv(k);
  return (v && *v) ? std::string(v) : std::string{};
}

inline bool env_truthy(const char* k) {
  const auto v = getenv_str(k);
  if (v.empty()) return false;
  char c = v[0];
  return !(c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N');
}

} // namespace welvet_qnn::plat
