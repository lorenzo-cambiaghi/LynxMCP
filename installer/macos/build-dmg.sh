#!/usr/bin/env bash
# build-dmg.sh — assemble Lynx.app and wrap it in a drag-to-install .dmg.
#
# Bundles only the `uv` binary (~30 MB each arch); the heavy Python stack is
# downloaded on first launch by bootstrap.sh. Run on macOS (locally or on a
# GitHub Actions macos-latest runner).
#
# Inputs (env vars):
#   VERSION        Version string for the bundle / dmg name. Default: read
#                  from pyproject.toml, else "0.0.0".
#   UV_ARM64_BIN   Path to the arm64 `uv` binary to bundle.  (at least one of
#   UV_X86_64_BIN  Path to the x86_64 `uv` binary to bundle.   these is required)
#
# Output: dist/Lynx-<VERSION>-macos.dmg
#
# Optional: if installer/macos/Lynx.png exists it is converted to an .icns
# and used as the app icon.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- resolve version --------------------------------------------------------
if [ -z "${VERSION:-}" ]; then
    VERSION="$(grep -m1 '^version[[:space:]]*=' "$REPO_ROOT/pyproject.toml" \
        | sed -E 's/.*"([^"]+)".*/\1/')"
    VERSION="${VERSION:-0.0.0}"
fi
echo "[build-dmg] version: $VERSION"

# --- validate uv inputs -----------------------------------------------------
if [ -z "${UV_ARM64_BIN:-}" ] && [ -z "${UV_X86_64_BIN:-}" ]; then
    echo "❌ Provide at least one of UV_ARM64_BIN / UV_X86_64_BIN." >&2
    exit 1
fi

# --- staging dirs -----------------------------------------------------------
BUILD_DIR="$REPO_ROOT/build/macos"
DIST_DIR="$REPO_ROOT/dist"
APP="$BUILD_DIR/Lynx.app"
rm -rf "$BUILD_DIR"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$DIST_DIR"

# --- Info.plist (version-substituted) ---------------------------------------
sed "s/__VERSION__/$VERSION/g" "$SCRIPT_DIR/Info.plist.template" \
    > "$APP/Contents/Info.plist"

# --- executable + bootstrap -------------------------------------------------
cp "$SCRIPT_DIR/launcher.sh"  "$APP/Contents/MacOS/Lynx"
cp "$SCRIPT_DIR/bootstrap.sh" "$APP/Contents/Resources/bootstrap.sh"
chmod +x "$APP/Contents/MacOS/Lynx" "$APP/Contents/Resources/bootstrap.sh"

# --- bundle uv binaries -----------------------------------------------------
if [ -n "${UV_ARM64_BIN:-}" ]; then
    cp "$UV_ARM64_BIN" "$APP/Contents/Resources/uv-arm64"
    chmod +x "$APP/Contents/Resources/uv-arm64"
    echo "[build-dmg] bundled uv-arm64"
fi
if [ -n "${UV_X86_64_BIN:-}" ]; then
    cp "$UV_X86_64_BIN" "$APP/Contents/Resources/uv-x86_64"
    chmod +x "$APP/Contents/Resources/uv-x86_64"
    echo "[build-dmg] bundled uv-x86_64"
fi

# --- optional icon ----------------------------------------------------------
if [ -f "$SCRIPT_DIR/Lynx.png" ]; then
    echo "[build-dmg] generating icon from Lynx.png"
    ICONSET="$BUILD_DIR/Lynx.iconset"
    mkdir -p "$ICONSET"
    for sz in 16 32 64 128 256 512; do
        sips -z $sz $sz       "$SCRIPT_DIR/Lynx.png" --out "$ICONSET/icon_${sz}x${sz}.png"      >/dev/null
        sips -z $((sz*2)) $((sz*2)) "$SCRIPT_DIR/Lynx.png" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
    done
    iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/Lynx.icns"
else
    echo "[build-dmg] no Lynx.png — using default app icon"
fi

# --- ad-hoc codesign (no warning suppression, but avoids 'damaged' errors) --
# A real Developer ID signature + notarization can be added in CI later.
codesign --force --deep --sign - "$APP" 2>/dev/null \
    && echo "[build-dmg] ad-hoc signed" \
    || echo "[build-dmg] codesign unavailable — shipping unsigned"

# --- assemble the .dmg staging folder (app + /Applications symlink) ---------
DMG_STAGE="$BUILD_DIR/dmg"
rm -rf "$DMG_STAGE"
mkdir -p "$DMG_STAGE"
cp -R "$APP" "$DMG_STAGE/Lynx.app"
ln -s /Applications "$DMG_STAGE/Applications"

DMG_PATH="$DIST_DIR/Lynx-$VERSION-macos.dmg"
rm -f "$DMG_PATH"
hdiutil create \
    -volname "Lynx" \
    -srcfolder "$DMG_STAGE" \
    -ov -format UDZO \
    "$DMG_PATH"

echo "[build-dmg] ✅ created $DMG_PATH"
