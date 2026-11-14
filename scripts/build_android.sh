#!/usr/bin/env bash
# Cross-compile welvet.qnn for Android arm64-v8a (NDK).
# Optional: set QAIRT_SDK_ROOT / QNN_SDK_ROOT for a real HTP build; otherwise stub ABI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<'HELP'
Usage: ./scripts/build_android.sh [arm64|all]
  ANDROID_NDK / ANDROID_HOME   NDK root (default: newest under $ANDROID_HOME/ndk)
  ANDROID_PLATFORM             default android-28
  BUILD_DIR_PREFIX             default $ROOT/build-android
  QAIRT_SDK_ROOT / QNN_SDK_ROOT  optional Qualcomm SDK (enables HTP graphs)
HELP
}

ABI_ARG="${1:-arm64}"
case "$ABI_ARG" in
  -h|--help) usage; exit 0 ;;
  arm64|arm64-v8a|all) ABIS=(arm64-v8a) ;;
  *) echo "Unknown ABI: $ABI_ARG (Android QNN ships arm64-v8a only)" >&2; usage >&2; exit 2 ;;
esac

if [[ -z "${ANDROID_NDK:-}" ]]; then
  if [[ -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}/ndk" ]]; then
    ANDROID_NDK="$(ls -1d "${ANDROID_HOME}/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "${ANDROID_NDK:-}" && -d "${HOME}/Android/Sdk/ndk" ]]; then
    ANDROID_NDK="$(ls -1d "${HOME}/Android/Sdk/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
  fi
fi
[[ -n "${ANDROID_NDK:-}" && -f "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" ]] \
  || { echo "ANDROID_NDK not found (set ANDROID_NDK or ANDROID_HOME)" >&2; exit 1; }

PLATFORM="${ANDROID_PLATFORM:-android-28}"
PREFIX="${BUILD_DIR_PREFIX:-$ROOT/build-android}"
JOBS="$(nproc 2>/dev/null || echo 4)"
TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"

SDK_ARGS=()
if [[ -n "${QAIRT_SDK_ROOT:-}" ]]; then
  SDK_ARGS+=(-DQAIRT_SDK_ROOT="${QAIRT_SDK_ROOT}")
elif [[ -n "${QNN_SDK_ROOT:-}" ]]; then
  SDK_ARGS+=(-DQAIRT_SDK_ROOT="${QNN_SDK_ROOT}")
fi

echo "NDK:      $ANDROID_NDK"
echo "Platform: $PLATFORM"
echo "SDK:      ${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-(none — stub ABI)}}"

for abi in "${ABIS[@]}"; do
  BUILD="${PREFIX}-${abi}"
  echo "=== building ${abi} → ${BUILD} ==="
  cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI="$abi" \
    -DANDROID_PLATFORM="$PLATFORM" \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DWELVET_QNN_BUILD_TOOLS=OFF \
    "${SDK_ARGS[@]}"
  cmake --build "$BUILD" --parallel "$JOBS"
  # Prefer shared lib next to build root / lib/
  SO=""
  for cand in "$BUILD/libwelvet_qnn.so" "$BUILD/lib/libwelvet_qnn.so"; do
    [[ -f "$cand" ]] && SO="$cand" && break
  done
  [[ -n "$SO" ]] || { echo "missing libwelvet_qnn.so under $BUILD" >&2; exit 1; }
  if [[ "$SO" != "$BUILD/libwelvet_qnn.so" ]]; then
    cp -f "$SO" "$BUILD/libwelvet_qnn.so"
  fi
  echo "built: $BUILD/libwelvet_qnn.so"
done
