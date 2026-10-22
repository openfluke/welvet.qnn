// welvet_qnn_probe — child-process gate for backendCreate (exit 0 = ok).
// Runs create on a large-stack thread (HTP often overflows the default 1MB stack).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
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

struct CreateJob {
  const char* path = nullptr;
  int result = 1;
  unsigned long long status = 0;
  void* be = nullptr;
};

#if defined(_WIN32)
static DWORD WINAPI create_thread_fn(void* arg) {
#else
static void* create_thread_fn(void* arg) {
#endif
  auto* job = (CreateJob*)arg;
  void* lib = open_lib(job->path);
  if (!lib) {
    job->result = 2;
#if defined(_WIN32)
    return 0;
#else
    return nullptr;
#endif
  }
  using G = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  auto gp = (G)sym(lib, "QnnInterface_getProviders");
  if (!gp) {
    job->result = 3;
#if defined(_WIN32)
    return 0;
#else
    return nullptr;
#endif
  }
  QnnInterface_t** p = nullptr;
  uint32_t n = 0;
  if (gp((const QnnInterface_t***)&p, &n) != QNN_SUCCESS || !p || !n) {
    job->result = 4;
#if defined(_WIN32)
    return 0;
#else
    return nullptr;
#endif
  }
  auto api = p[0]->QNN_INTERFACE_VER_NAME;
  if (!api.backendCreate) {
    job->result = 5;
#if defined(_WIN32)
    return 0;
#else
    return nullptr;
#endif
  }
  Qnn_BackendHandle_t be = nullptr;
  std::printf("backendCreate (large stack)...\n");
  fflush(stdout);
  auto e = api.backendCreate(nullptr, nullptr, &be);
  job->status = (unsigned long long)e;
  job->be = be;
  if (e != QNN_SUCCESS || !be) {
    job->result = 6;
  } else {
    if (api.backendFree) api.backendFree(be);
    job->result = 0;
  }
#if defined(_WIN32)
  return 0;
#else
  return nullptr;
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

  CreateJob job;
  job.path = path;

  // 64 MiB stack — QnnHtp/Cpu/Gpu backendCreate recurses deeply on some hosts.
  constexpr size_t kStack = 64ull * 1024ull * 1024ull;
#if defined(_WIN32)
  HANDLE th = CreateThread(nullptr, kStack, create_thread_fn, &job, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
  if (!th) {
    std::fprintf(stderr, "CreateThread failed\n");
    return 7;
  }
  WaitForSingleObject(th, 120000);
  DWORD code = 0;
  GetExitCodeThread(th, &code);
  CloseHandle(th);
  if (code != 0 && job.result == 1) {
    // Thread itself died (e.g. uncaught SEH) — surface as overflow-ish failure.
    std::fprintf(stderr, "create thread exited 0x%lX\n", (unsigned long)code);
    return (int)code;
  }
#else
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, kStack);
  pthread_t th{};
  if (pthread_create(&th, &attr, create_thread_fn, &job) != 0) {
    std::fprintf(stderr, "pthread_create failed\n");
    return 7;
  }
  pthread_join(th, nullptr);
  pthread_attr_destroy(&attr);
#endif

  std::printf("status=%llu be=%p result=%d\n", job.status, job.be, job.result);
  return job.result;
}
#endif
