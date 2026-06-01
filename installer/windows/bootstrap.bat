@echo off
REM ===========================================================================
REM  bootstrap.bat - first-run installer + UI launcher for the Windows build.
REM
REM  Installed to %LOCALAPPDATA%\Lynx alongside uv.exe by the Inno Setup
REM  installer. The Start Menu / desktop shortcut points here, so a console
REM  window opens showing the one-time ~1 GB download, then the UI starts.
REM
REM  Everything is kept under %LOCALAPPDATA%\Lynx so we never touch the
REM  system Python or PATH. Uninstall removes the program; you can also
REM  delete %LOCALAPPDATA%\Lynx to reclaim the downloaded environment.
REM ===========================================================================

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "DATA_DIR=%LOCALAPPDATA%\Lynx"
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM Isolate uv: tools, shims and cache all live under the data dir.
set "UV_TOOL_DIR=%DATA_DIR%\uv\tools"
set "UV_TOOL_BIN_DIR=%DATA_DIR%\uv\bin"
set "UV_CACHE_DIR=%DATA_DIR%\uv\cache"
if not exist "%UV_TOOL_DIR%"     mkdir "%UV_TOOL_DIR%"
if not exist "%UV_TOOL_BIN_DIR%" mkdir "%UV_TOOL_BIN_DIR%"
if not exist "%UV_CACHE_DIR%"    mkdir "%UV_CACHE_DIR%"

REM Stable config path (a shortcut-launched process has an unpredictable cwd).
set "RAG_CONFIG_PATH=%DATA_DIR%\config.json"

set "UV_BIN=%SCRIPT_DIR%uv.exe"
set "LYNX_BIN=%UV_TOOL_BIN_DIR%\lynx.exe"

REM Source of the package. Swap to "lynx" once it's published on PyPI.
set "LYNX_PKG=git+https://github.com/lorenzo-cambiaghi/LynxMCP.git"

echo.
echo    /\     /\
echo   {  `---'  }   LynxMCP
echo   {  O   O  }   Privacy-first long-term memory for your AI coding assistant
echo   ~~^>  V  ^<~~
echo    \  \^|/  /
echo     `-----'
echo.

if not exist "%UV_BIN%" (
    echo X  Bundled uv.exe not found at "%UV_BIN%".
    echo    Try reinstalling Lynx. Press any key to close.
    pause >nul
    exit /b 1
)

REM --- first run: install lynx via uv (downloads ~1 GB once) ---------------
"%UV_BIN%" tool list 2>nul | findstr /b /c:"lynx" >nul
if errorlevel 1 (
    echo [Lynx] First launch -- setting up. This downloads ~1 GB once and can
    echo        take several minutes. Subsequent launches are instant.
    echo.
    "%UV_BIN%" tool install --force "%LYNX_PKG%"
    if errorlevel 1 (
        echo.
        echo X  Setup failed. Check your internet connection and try again.
        pause >nul
        exit /b 1
    )
    echo.
    echo [Lynx] Core install complete.
)

if not exist "%LYNX_BIN%" (
    echo X  Install finished but "%LYNX_BIN%" is missing. Press any key to close.
    pause >nul
    exit /b 1
)

REM --- write default config + pre-download embedding model (once) ----------
if not exist "%RAG_CONFIG_PATH%" (
    echo [Lynx] Writing default config + downloading embedding model (~130 MB)...
    "%LYNX_BIN%" manager init --non-interactive --output "%RAG_CONFIG_PATH%"
    echo [Lynx] Setup done.
    echo.
)

REM --- launch the web UI (opens the browser; Ctrl+C here to stop) ----------
echo [Lynx] Starting LynxManager -- your browser will open shortly.
echo [Lynx] Press Ctrl+C in this window to stop the server.
echo.
"%LYNX_BIN%" manager ui

echo.
echo [Lynx] Server stopped. Press any key to close this window.
pause >nul
endlocal
