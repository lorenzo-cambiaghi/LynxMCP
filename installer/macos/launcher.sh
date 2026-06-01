#!/usr/bin/env bash
# launcher.sh — becomes Lynx.app/Contents/MacOS/Lynx (the bundle executable).
#
# Finder runs this with no visible window. We want the user to see the
# one-time download progress and be able to Ctrl+C the UI server, so we
# hand off to bootstrap.sh inside a Terminal.app window.

set -eu

MACOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_DIR="$(cd "$MACOS_DIR/../Resources" && pwd)"

chmod +x "$RES_DIR/bootstrap.sh" 2>/dev/null || true

# `open -a Terminal <script>` opens a new Terminal window running the script.
exec open -a Terminal "$RES_DIR/bootstrap.sh"
