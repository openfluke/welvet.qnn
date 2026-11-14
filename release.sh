#!/usr/bin/env bash
# Publish this independent Qualcomm Hexagon / QNN HTP backend.
#
# Default / --source-only: commit local source, push branch + immutable tag,
# then create a GitHub Release (source tag; no binary build).
#
# Optional --package-android: also NDK-cross-compile arm64-v8a libwelvet_qnn.so
# and attach a tarball to the release (needs ANDROID_NDK; QAIRT optional).
#
# The script never force-pushes or moves an existing tag.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DRY_RUN=0
PLAN=0
SOURCE_ONLY=1
PACKAGE_ANDROID=0
SKIP_BUILD=0
DRAFT=0
AUTO_COMMIT=1
BUILD_ROOT="$ROOT/build-release"
DIST="$ROOT/dist"

usage() {
  cat <<'HELP'
Usage: ./release.sh [options]
  --source-only       Tag + GitHub Release from source only (default)
  --package-android   Also build/package android-arm64-v8a shared lib (needs NDK)
  --dry-run           Do everything except commit/tag/push/publish
  --plan              Show version and actions only
  --skip-build        With --package-android: reuse build-release/android-arm64-v8a
  --no-commit         Require a clean source tree instead of committing local changes
  --draft             Create a draft GitHub Release
  -h, --help          Show this help

Default: release v<current CMake/README/runtime version> as a source tag to origin.
Publishing requires an authenticated GitHub CLI (gh). No sibling repo is modified.

Windows ARM64 HTP binaries are built on-device (llvm-mingw + QAIRT); install into
Go Welvet with a local cmake build → native/qnn/windows_arm64/.
Android install helper: ./scripts/android_install_go.sh
HELP
}

for arg in "$@"; do
  case "$arg" in
    --source-only) SOURCE_ONLY=1; PACKAGE_ANDROID=0 ;;
    --package-android) PACKAGE_ANDROID=1; SOURCE_ONLY=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --plan) PLAN=1; DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --no-commit) AUTO_COMMIT=0 ;;
    --draft) DRAFT=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command missing: $1"; }

need git
need python3

VERSION="$(python3 - "$ROOT" <<'PY'
from pathlib import Path
import re, sys
r = Path(sys.argv[1])
cm = re.search(r'project\(welvet_qnn\s+VERSION\s+([0-9.]+)', (r / 'CMakeLists.txt').read_text())
rm = re.search(r'\|\s*\*\*Library\*\*\s*\|\s*\*\*([0-9.]+)\*\*', (r / 'README.md').read_text())
abi = re.search(r'wqnn_version\(void\)\s*\{\s*return\s+"([0-9.]+)"', (r / 'src/c_abi.cpp').read_text())
if not all((cm, rm, abi)) or len({m.group(1) for m in (cm, rm, abi)}) != 1:
    raise SystemExit('CMake, README (**Library**) and wqnn_version() must agree')
print(cm.group(1))
PY
)"

TAG="v$VERSION"
BRANCH="$(git symbolic-ref --quiet --short HEAD)" || fail "Check out a branch before releasing"
REPO="$(git remote get-url origin | python3 -c '
import re, sys
url = sys.stdin.read().strip()
m = re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+/[^/]+?)(?:\.git)?/?", url)
if not m:
    raise SystemExit("origin must point to a GitHub repository")
print(m.group(1))
')"
[[ "$REPO" == */welvet.qnn ]] || fail "origin must name a welvet.qnn repository"

if [[ "$DRY_RUN" -eq 0 ]]; then
  need gh
  gh auth status --hostname github.com >/dev/null 2>&1 || fail "Run gh auth login first"
  if [[ "$AUTO_COMMIT" -eq 0 && -n "$(git status --porcelain)" ]]; then
    fail "Source tree is dirty; commit changes or omit --no-commit"
  fi
  if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
    [[ "$(git rev-parse "$TAG^{commit}")" == "$(git rev-parse HEAD)" ]] \
      || fail "$TAG already names another commit; tags are never moved"
    [[ -z "$(git status --porcelain)" ]] || fail "$TAG already exists; do not change the tagged source"
  fi
fi

KIND="source-only"
[[ "$PACKAGE_ANDROID" -eq 1 ]] && KIND="source + android-arm64-v8a package"
printf 'welvet.qnn %s (%s)\nGitHub: %s\nBranch: %s\n' "$TAG" "$KIND" "$REPO" "$BRANCH"

if [[ "$PLAN" -eq 1 ]]; then
  if [[ "$PACKAGE_ANDROID" -eq 1 ]]; then
    echo "Plan: NDK cmake arm64-v8a shared lib -> package + SHA256"
  else
    echo "Plan: source tag + GitHub Release notes (no NDK / no binary assets)"
  fi
  echo "Publish: commit local source (unless --no-commit), push branch + immutable tag, gh release create"
  echo "Plan only: no files, git state or remote state changed"
  exit 0
fi

mkdir -p "$DIST"
NOTES="$DIST/release-$TAG.md"
ANDROID_CHECK="not run (source-only release)"
ARCHIVE=""
SHA=""

# ---- optional Android package (needs NDK) ----
if [[ "$PACKAGE_ANDROID" -eq 1 ]]; then
  need cmake
  need tar
  need sha256sum

  if [[ -z "${ANDROID_NDK:-}" ]]; then
    if [[ -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}/ndk" ]]; then
      ANDROID_NDK="$(ls -1d "${ANDROID_HOME}/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
    fi
    if [[ -z "${ANDROID_NDK:-}" && -d "${HOME}/Android/Sdk/ndk" ]]; then
      ANDROID_NDK="$(ls -1d "${HOME}/Android/Sdk/ndk/"* 2>/dev/null | sort -V | tail -1 || true)"
    fi
  fi
  [[ -n "${ANDROID_NDK:-}" && -f "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" ]] \
    || fail "ANDROID_NDK not found (set ANDROID_NDK or ANDROID_HOME)"

  PLATFORM="${ANDROID_PLATFORM:-android-28}"
  ABI="arm64-v8a"
  JOBS="$(nproc 2>/dev/null || echo 4)"
  TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
  BUILD="${BUILD_ROOT}/android-${ABI}"
  mkdir -p "$BUILD_ROOT"

  STAGE="$(mktemp -d "$DIST/.stage.XXXXXX")"
  cleanup() {
    python3 - "$STAGE" <<'PY'
from pathlib import Path
import shutil, sys
p = Path(sys.argv[1])
if p.name.startswith('.stage.'):
    shutil.rmtree(p, ignore_errors=True)
PY
  }
  trap cleanup EXIT

  NAME="welvet-qnn-$TAG-android-arm64"
  PACKAGE="$STAGE/$NAME"
  mkdir -p "$PACKAGE/lib" "$PACKAGE/include"

  SDK_ARGS=()
  if [[ -n "${QAIRT_SDK_ROOT:-}" ]]; then
    SDK_ARGS+=(-DQAIRT_SDK_ROOT="${QAIRT_SDK_ROOT}")
  elif [[ -n "${QNN_SDK_ROOT:-}" ]]; then
    SDK_ARGS+=(-DQAIRT_SDK_ROOT="${QNN_SDK_ROOT}")
  fi

  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    cmake -S "$ROOT" -B "$BUILD" \
      -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
      -DANDROID_ABI="$ABI" \
      -DANDROID_PLATFORM="$PLATFORM" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DWELVET_QNN_BUILD_TOOLS=OFF \
      "${SDK_ARGS[@]}"
    cmake --build "$BUILD" --parallel "$JOBS"
  fi

  SO=""
  for cand in "$BUILD/libwelvet_qnn.so" "$BUILD/lib/libwelvet_qnn.so"; do
    [[ -f "$cand" ]] && SO="$cand" && break
  done
  [[ -n "$SO" ]] || fail "Missing libwelvet_qnn.so under $BUILD; run without --skip-build"

  STRIP_BIN=""
  for cand in \
    "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" \
    "$(command -v llvm-strip 2>/dev/null || true)"
  do
    if [[ -n "$cand" && -x "$cand" ]]; then
      STRIP_BIN="$cand"
      break
    fi
  done
  if [[ -n "$STRIP_BIN" ]]; then
    "$STRIP_BIN" --strip-unneeded -o "$PACKAGE/lib/libwelvet_qnn.so" "$SO"
  else
    cp "$SO" "$PACKAGE/lib/libwelvet_qnn.so"
  fi

  cp "$ROOT/include/welvet_qnn_c.h" "$PACKAGE/include/"
  cp -r "$ROOT/include/welvet_qnn" "$PACKAGE/include/"
  cp "$ROOT/README.md" "$ROOT/VERSIONING.md" "$ROOT/LICENSE" "$PACKAGE/"

  HAS_SDK=0
  [[ ${#SDK_ARGS[@]} -gt 0 ]] && HAS_SDK=1
  if [[ "$HAS_SDK" -eq 1 ]]; then
    ANDROID_CHECK="packaged android-arm64-v8a shared lib (API ${PLATFORM#android-}; QAIRT headers linked)"
  else
    ANDROID_CHECK="packaged android-arm64-v8a shared lib (API ${PLATFORM#android-}; stub ABI — rebuild with QAIRT_SDK_ROOT for HTP graphs)"
  fi

  python3 - "$PACKAGE" "$VERSION" "PENDING" "false" "$ANDROID_CHECK" "$HAS_SDK" <<'PY'
import json, sys
from pathlib import Path
p, v, commit, dirty, android, has_sdk = sys.argv[1:]
manifest = dict(
    engine='welvet.qnn',
    version=v,
    platform='android-arm64-v8a',
    api_level=28,
    source_commit=commit,
    source_dirty_at_build=dirty == 'true',
    android_validation=android,
    qairt_sdk_linked=has_sdk == '1',
)
Path(p, 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY

  ARCHIVE="$DIST/$NAME.tar.gz"
  tar -czf "$ARCHIVE" -C "$STAGE" "$NAME"
  (cd "$DIST" && sha256sum "$NAME.tar.gz" > "$NAME.sha256")
  SHA="$DIST/$NAME.sha256"
fi

# ---- git commit / tag / notes ----
if [[ "$DRY_RUN" -eq 0 && -n "$(git status --porcelain)" ]]; then
  [[ "$AUTO_COMMIT" -eq 1 ]] || fail "Dirty source tree"
  git add -A
  git commit -m "Release welvet.qnn $TAG"
fi
COMMIT="$(git rev-parse HEAD)"
DIRTY=false
[[ -z "$(git status --porcelain)" ]] || DIRTY=true

# Refresh MANIFEST commit if we packaged before committing.
if [[ -n "$ARCHIVE" && -d "${STAGE:-}" ]]; then
  python3 - "$PACKAGE" "$VERSION" "$COMMIT" "$DIRTY" "$ANDROID_CHECK" "${HAS_SDK:-0}" <<'PY'
import json, sys
from pathlib import Path
p, v, commit, dirty, android, has_sdk = sys.argv[1:]
manifest = dict(
    engine='welvet.qnn',
    version=v,
    platform='android-arm64-v8a',
    api_level=28,
    source_commit=commit,
    source_dirty_at_build=dirty == 'true',
    android_validation=android,
    qairt_sdk_linked=has_sdk == '1',
)
Path(p, 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
  tar -czf "$ARCHIVE" -C "$STAGE" "$NAME"
  (cd "$DIST" && sha256sum "$NAME.tar.gz" > "$NAME.sha256")
fi

cat > "$NOTES" <<NOTES
# welvet.qnn $TAG

Independent Qualcomm Hexagon NPU (QNN HTP) backend for Welvet.
Forward-only C ABI + Welvet-shaped layers; Go binding is \`nativeqnn\`.

- Hosts: Windows ARM64 (Snapdragon X / Hexagon V73) · Android arm64-v8a
- User-owned \`SyncToQNN\` / \`SyncFromQNN\`; layer×dtype caps via \`wqnn_layer_supported\`
- Separate from welvet.cpp / cuda / metal / vulkan / mlx — no sibling runtime dependency
- See VERSIONING.md for 1.x policy

## Source

Tag \`$TAG\` on branch \`$BRANCH\` (commit \`$COMMIT\`).

\`\`\`bash
# Windows ARM64 (on device): cmake + QAIRT → libwelvet_qnn.dll
# Android arm64:
./scripts/android_install_go.sh   # → ../welvet/native/qnn/android_arm64/
\`\`\`

Android package check: $ANDROID_CHECK.
Dirty working-tree package: $DIRTY.
NOTES

if [[ -n "$ARCHIVE" ]]; then
  cat >> "$NOTES" <<NOTES

## Android arm64-v8a binary artifact

\`$(basename "$ARCHIVE")\` — shared \`libwelvet_qnn.so\`, C headers, docs, MANIFEST.
Install into Go Welvet: extract \`lib/\` → \`welvet/native/qnn/android_arm64/\`.
Verify with: \`sha256sum -c $(basename "$SHA")\`
NOTES
fi

printf '\nPrepared notes:\n  %s\n' "$NOTES"
[[ -z "$ARCHIVE" ]] || printf '  %s\n  %s\n' "$ARCHIVE" "$SHA"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete: no commit, tag, push or GitHub release performed"
  exit 0
fi

if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
  [[ "$(git rev-parse "$TAG^{commit}")" == "$COMMIT" ]] || fail "$TAG is not the release commit"
else
  git tag -a "$TAG" -m "welvet.qnn $TAG"
fi
git push origin "HEAD:refs/heads/$BRANCH"
git push origin "refs/tags/$TAG"

release_args=("$TAG" --repo "$REPO" --verify-tag --title "welvet.qnn $TAG" --notes-file "$NOTES")
[[ -z "$ARCHIVE" ]] || release_args+=("$ARCHIVE" "$SHA")
[[ "$DRAFT" -eq 0 ]] || release_args+=(--draft)
gh release create "${release_args[@]}"
echo "Release: https://github.com/$REPO/releases/tag/$TAG"
