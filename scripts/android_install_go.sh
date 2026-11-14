#!/usr/bin/env bash
# Build Android arm64-v8a libwelvet_qnn.so and install into a local Go Welvet checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_GO_ROOT="${WELVET_GO_ROOT:-$ROOT/../welvet}"
if [[ ! -f "$WELVET_GO_ROOT/go.mod" ]]; then
  echo "Go Welvet checkout not found: $WELVET_GO_ROOT" >&2
  exit 1
fi

if [[ -z "${ANDROID_NDK:-}" ]]; then
  if [[ -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}/ndk" ]]; then
    ANDROID_NDK="$(ls -1d "${ANDROID_HOME}/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "${ANDROID_NDK:-}" && -d "${HOME}/Android/Sdk/ndk" ]]; then
    ANDROID_NDK="$(ls -1d "${HOME}/Android/Sdk/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
  fi
fi
export ANDROID_NDK

"$ROOT/scripts/build_android.sh" arm64

PREFIX="${BUILD_DIR_PREFIX:-$ROOT/build-android}"
SRC="${PREFIX}-arm64-v8a/libwelvet_qnn.so"
[[ -f "$SRC" ]] || { echo "missing $SRC" >&2; exit 1; }

INCLUDE_DIR="$WELVET_GO_ROOT/native/qnn/include"
mkdir -p "$INCLUDE_DIR"
cp "$ROOT/include/welvet_qnn_c.h" "$INCLUDE_DIR/welvet_qnn_c.h"

VERSION="1.1.0"
if grep -q 'wqnn_version' "$ROOT/src/c_abi.cpp" 2>/dev/null; then
  VERSION="$(sed -n 's/.*return "\([0-9][^"]*\)".*/\1/p' "$ROOT/src/c_abi.cpp" | head -1)"
  VERSION="${VERSION:-1.1.0}"
fi

STRIP_BIN=""
for cand in \
  "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" \
  "$(command -v llvm-strip 2>/dev/null || true)" \
  "$(command -v strip || true)"
do
  if [[ -n "$cand" && -x "$cand" ]]; then
    STRIP_BIN="$cand"
    break
  fi
done

DEST_DIR="$WELVET_GO_ROOT/native/qnn/android_arm64"
mkdir -p "$DEST_DIR"
if [[ -n "$STRIP_BIN" ]]; then
  "$STRIP_BIN" --strip-unneeded -o "$DEST_DIR/libwelvet_qnn.so" "$SRC"
else
  cp "$SRC" "$DEST_DIR/libwelvet_qnn.so"
fi
(
  cd "$DEST_DIR"
  sha256sum libwelvet_qnn.so > SHA256SUMS
)

cat > "$DEST_DIR/README.md" <<EOF
# android_arm64 — welvet.qnn (Hexagon NPU)

Shipped \`libwelvet_qnn.so\` for Android arm64-v8a (API 28+). Artifacts for
\`jniLibs\` / NDK packaging. Go \`nativeqnn\` links this path on
\`GOOS=android\` + \`arm64\` when cgo is enabled.

\`\`\`bash
cd welvet.qnn
./scripts/android_install_go.sh
\`\`\`

Runtime needs QAIRT Android HTP libs + Hexagon skels packaged with the app
(or on device). Rebuild with \`QAIRT_SDK_ROOT\` set for a full HTP binary.

Engine pin: see \`../MANIFEST.txt\` (welvet.qnn **${VERSION}**).
EOF

{
  echo "engine=welvet.qnn"
  echo "version=$VERSION"
  echo "platforms=windows_arm64,android_arm64"
  echo "source_checkout=$ROOT"
  if [[ -f "$WELVET_GO_ROOT/native/qnn/windows_arm64/libwelvet_qnn.dll" ]]; then
    sha256sum "$WELVET_GO_ROOT/native/qnn/windows_arm64/libwelvet_qnn.dll"
  fi
  sha256sum "$DEST_DIR/libwelvet_qnn.so"
} > "$WELVET_GO_ROOT/native/qnn/MANIFEST.txt"

echo "Installed $DEST_DIR/libwelvet_qnn.so"
echo "Go: package nativeqnn (cgo + android/arm64 or windows/arm64)"
