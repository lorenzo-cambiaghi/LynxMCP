@echo off
REM ===========================================================================
REM  LynxManager.bat - Windows double-click launcher.
REM
REM  Double-click this file in Explorer to start the LynxManager web UI.
REM  A console window opens, the UI boots, and your default browser is
REM  pointed at http://127.0.0.1:8765.
REM
REM  Prereq: `lynx` is installed somewhere we can find. We try, in order:
REM    1. `lynx` on PATH (this is what `pipx install lynx` gives you).
REM    2. `.venv\Scripts\lynx.exe` relative to this file (dev clone with
REM       an editable install in a local virtualenv).
REM    3. `py -m lynx` as a last resort, using the Windows launcher.
REM    4. `python -m lynx` as a final fallback.
REM
REM  If none work, we print the install command and pause so you can read
REM  the message before the window closes.
REM ===========================================================================

setlocal EnableDelayedExpansion

REM cd to the folder containing this script so a sibling config.json /
REM .venv is picked up automatically.
cd /d "%~dp0"
echo [LynxManager] launching from: %CD%
echo.

set "LYNX_CMD="

REM 1. `lynx` on PATH (pipx / uv tool install).
where lynx >nul 2>&1
if %errorlevel% equ 0 (
    set "LYNX_CMD=lynx"
    echo [LynxManager] using `lynx` on PATH.
    goto :run
)

REM 2. Local editable install in .venv.
if exist "%CD%\.venv\Scripts\lynx.exe" (
    set "LYNX_CMD=%CD%\.venv\Scripts\lynx.exe"
    echo [LynxManager] using local venv: .venv\Scripts\lynx.exe
    goto :run
)

REM 3. Windows `py` launcher with -m lynx.
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -c "import lynx" >nul 2>&1
    if !errorlevel! equ 0 (
        set "LYNX_CMD=py -m lynx"
        echo [LynxManager] using `py -m lynx`.
        goto :run
    )
)

REM 4. Plain `python -m lynx`.
where python >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import lynx" >nul 2>&1
    if !errorlevel! equ 0 (
        set "LYNX_CMD=python -m lynx"
        echo [LynxManager] using `python -m lynx`.
        goto :run
    )
)

echo.
echo X  Couldn't find a working `lynx` install on this machine.
echo.
echo The recommended one-time setup is pipx (creates an isolated env and
echo puts `lynx` on your PATH):
echo.
echo     py -m pip install --user pipx
echo     pipx install git+https://github.com/lorenzo-cambiaghi/LynxMCP.git
echo.
echo Alternative -- uv:
echo.
echo     uv tool install git+https://github.com/lorenzo-cambiaghi/LynxMCP.git
echo.
echo After that, double-click this file again.
echo.
pause
exit /b 1

:run
REM HTTPS-inspecting antivirus (Avast/AVG) injects SSLKEYLOGFILE pointing at a
REM device path like \\.\aswMonFltProxy\... . Python's `ssl` opens it via
REM OpenSSL's file BIO on first TLS use, which aborts the bundled interpreter
REM ("OPENSSL_Uplink: no OPENSSL_Applink") mid-startup with no traceback. Lynx
REM never needs a TLS key-log, so clear it for this process. The Python entry
REM point sanitizes it too, but clearing it here keeps even odd launchers safe.
set "SSLKEYLOGFILE="

echo [LynxManager] starting web UI -- your browser will open shortly.
echo [LynxManager] press Ctrl+C in this window to stop the server.
echo.

REM Run the UI in the foreground so the user can see logs + Ctrl+C it.
%LYNX_CMD% manager ui %*

echo.
echo [LynxManager] server stopped. Press any key to close this window.
pause >nul
endlocal
