#!/usr/bin/env bash
# LynxManager.command — macOS / Linux double-click launcher.
#
# Right-click → Open (or just double-click) this file from Finder to
# start the LynxManager web UI. A Terminal window opens, the UI boots,
# and your default browser is pointed at http://127.0.0.1:8765.
#
# Prereq: `lynx` is installed somewhere we can find. The launcher tries,
# in order:
#   1. `lynx` on PATH (this is what `pipx install lynx` gives you).
#   2. `.venv/bin/lynx` relative to this script (a dev clone with an
#      editable install in a local virtualenv).
#   3. `python3 -m lynx` as a last resort, using whichever python3 is
#      on PATH.
#
# If none work, we print the install command and pause so you can read
# the message before the window closes.

set -u

# cd to the folder containing this script so a sibling config.json /
# .venv is picked up automatically.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "[LynxManager] launching from: $SCRIPT_DIR"
echo

# Pick the best invocation we can find.
LYNX_CMD=""
if command -v lynx >/dev/null 2>&1; then
    LYNX_CMD="lynx"
    echo "[LynxManager] using \`lynx\` on PATH ($(command -v lynx))"
elif [ -x "$SCRIPT_DIR/.venv/bin/lynx" ]; then
    LYNX_CMD="$SCRIPT_DIR/.venv/bin/lynx"
    echo "[LynxManager] using local venv: .venv/bin/lynx"
elif command -v python3 >/dev/null 2>&1; then
    # Verify the module is actually importable before committing.
    if python3 -c "import lynx" >/dev/null 2>&1; then
        LYNX_CMD="python3 -m lynx"
        echo "[LynxManager] using \`python3 -m lynx\`"
    fi
fi

if [ -z "$LYNX_CMD" ]; then
    cat <<EOF

❌ Couldn't find a working \`lynx\` install on this machine.

The recommended one-time setup is pipx (creates an isolated env and
puts \`lynx\` on your PATH):

    brew install pipx                # if you don't have it yet
    pipx install git+https://github.com/lorenzo-cambiaghi/LynxMCP.git

Alternative — uv:

    uv tool install git+https://github.com/lorenzo-cambiaghi/LynxMCP.git

After that, double-click this file again.

Press Enter to close this window.
EOF
    read -r _
    exit 1
fi

echo "[LynxManager] starting web UI — your browser will open shortly."
echo "[LynxManager] press Ctrl+C in this window to stop the server."
echo

# Run the UI in the foreground so the user can see logs + Ctrl+C it.
# `$LYNX_CMD` is unquoted on purpose so "python3 -m lynx" splits correctly.
# shellcheck disable=SC2086
$LYNX_CMD manager ui "$@"

# Keep the window open if the server crashed quickly, so the user can
# read the error message before macOS closes the Terminal window.
echo
echo "[LynxManager] server stopped. Press Enter to close this window."
read -r _
