#!/usr/bin/env bash
# bootstrap.sh — first-run installer + UI launcher for the macOS Lynx.app.
#
# Lives in Lynx.app/Contents/Resources/. Run inside Terminal.app by the
# bundle's MacOS/Lynx launcher, so the user sees the one-time ~1 GB
# download and can Ctrl+C the UI server.
#
# Everything is kept inside a private per-user data dir so we never touch
# the system Python or PATH. Uninstall = drag Lynx.app to the Trash and
# delete ~/Library/Application Support/Lynx.

set -euo pipefail

# --- where this script lives (Contents/Resources) ---------------------------
RES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- private, self-contained data dir ---------------------------------------
DATA_DIR="$HOME/Library/Application Support/Lynx"
mkdir -p "$DATA_DIR"

# Isolate uv: tools + their bin shims live under DATA_DIR, not ~/.local.
export UV_TOOL_DIR="$DATA_DIR/uv/tools"
export UV_TOOL_BIN_DIR="$DATA_DIR/uv/bin"
# Keep uv's own cache local too, so a clean uninstall removes everything.
export UV_CACHE_DIR="$DATA_DIR/uv/cache"
mkdir -p "$UV_TOOL_DIR" "$UV_TOOL_BIN_DIR" "$UV_CACHE_DIR"

# Stable config location (an app launched from Finder has an unpredictable cwd).
export RAG_CONFIG_PATH="$DATA_DIR/config.json"

# --- pick the bundled uv binary for this Mac's architecture -----------------
ARCH="$(uname -m)"
case "$ARCH" in
    arm64)  UV_BIN="$RES_DIR/uv-arm64" ;;
    x86_64) UV_BIN="$RES_DIR/uv-x86_64" ;;
    *)      echo "❌ Unsupported architecture: $ARCH"; UV_BIN="$RES_DIR/uv-x86_64" ;;
esac

if [ ! -x "$UV_BIN" ]; then
    # Fallback: whichever single bundled binary exists, or uv on PATH.
    if [ -x "$RES_DIR/uv-arm64" ]; then UV_BIN="$RES_DIR/uv-arm64"
    elif [ -x "$RES_DIR/uv-x86_64" ]; then UV_BIN="$RES_DIR/uv-x86_64"
    elif command -v uv >/dev/null 2>&1; then UV_BIN="$(command -v uv)"
    else
        echo "❌ No bundled 'uv' binary found in $RES_DIR and none on PATH."
        echo "   Press Enter to close this window."
        read -r _
        exit 1
    fi
fi

LYNX_BIN="$UV_TOOL_BIN_DIR/lynx"

# Source of the package. Swap to "lynx-mcp" once it is published on PyPI.
LYNX_PKG="git+https://github.com/lorenzo-cambiaghi/LynxMCP.git"

cat <<'BANNER'
   /\     /\
  {  `---'  }   LynxMCP
  {  O   O  }   100% local semantic code search for your AI coding assistant
  ~~>  V  <~~
   \  \|/  /
    `-----'
BANNER
echo

# --- first run: install lynx via uv (downloads ~1 GB once) ------------------
if ! "$UV_BIN" tool list 2>/dev/null | grep -q '^lynx\b'; then
    echo "[Lynx] First launch — setting up (this downloads ~1 GB once and"
    echo "       can take several minutes on a slow connection). Subsequent"
    echo "       launches are instant."
    echo
    "$UV_BIN" tool install --force "$LYNX_PKG"
    echo
    echo "[Lynx] Core install complete."
fi

if [ ! -x "$LYNX_BIN" ]; then
    echo "❌ Install finished but '$LYNX_BIN' is missing. Press Enter to close."
    read -r _
    exit 1
fi

# --- write a default config + pre-download the embedding model (once) -------
if [ ! -f "$RAG_CONFIG_PATH" ]; then
    echo "[Lynx] Writing default config + downloading embedding model (~130 MB)..."
    "$LYNX_BIN" manager init --non-interactive --output "$RAG_CONFIG_PATH"
    echo "[Lynx] Setup done."
    echo
fi

# --- launch the web UI (opens the browser; Ctrl+C here to stop) -------------
echo "[Lynx] Starting LynxManager — your browser will open shortly."
echo "[Lynx] Press Ctrl+C in this window to stop the server."
echo
exec "$LYNX_BIN" manager ui
