// welvet_qnn_probe — child-process gate for backendCreate (exit 0 = ok).
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if !defined(WELVET_QNN_HAS_SDK) || !WELVET_QNN_HAS_SDK
int main() {
  std::fprintf(stderr, "built without QAIRT SDK\n");
  return 1;
}
#else

#include "QnnInterface.h"

static void* open_lib(const char* p) {
#if defined(_WIN32)
  return (void*)LoadLibraryExA(p, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
  return dlopen(p, RTLD_NOW | RTLD_GLOBAL);
#endif
}
static void* sym(void* h, const char* n) {
#if defined(_WIN32)
  return (void*)GetProcAddress((HMODULE)h, n);
#else
  return dlsym(h, n);
#endif
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: welvet_qnn_probe <QnnHtp.dll|…>\n");
    return 1;
  }
  const char* path = argv[1];
  std::printf("probe loading %s\n", path);
  fflush(stdout);
  void* lib = open_lib(path);
  if (!lib) {
    std::fprintf(stderr, "open failed\n");
    return 2;
  }
  using G = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  auto gp = (G)sym(lib, "QnnInterface_getProviders");
  if (!gp) return 3;
  QnnInterface_t** p = nullptr;
  uint32_t n = 0;
  if (gp((const QnnInterface_t***)&p, &n) != QNN_SUCCESS || !p || !n) return 4;
  auto api = p[0]->QNN_INTERFACE_VER_NAME;
  if (!api.backendCreate) return 5;
  Qnn_BackendHandle_t be = nullptr;
  std::printf("backendCreate...\n");
  fflush(stdout);
  auto e = api.backendCreate(nullptr, nullptr, &be);
  std::printf("status=%llu be=%p\n", (unsigned long long)e, be);
  if (e != QNN_SUCCESS || !be) return 6;
  if (api.backendFree) api.backendFree(be);
  return 0;
}
#endif
