<#
.SYNOPSIS
  Shows which Lynx code every running Lynx process runs, stops them, and
  starts them again on the code of this checkout.

.DESCRIPTION
  Three installs can run Lynx on this machine at once: the interpreter your
  MCP client starts (an editable install points it at this checkout), the
  `lynx` on PATH (a uv-tool or pipx copy of a release), and any venv. They
  all report a version number, and an editable checkout reports the same
  number before and after you change its code. So this script compares what
  matters: a hash of the package's sources, computed the way Lynx itself
  writes it into every index claim (`lynx.ownership.code_identity`).

  For each running process it says CURRENT (same code as this checkout,
  started after its last edit), STALE (same install, but started before the
  last edit: it still runs the old code from memory) or OLD (another install,
  other code). Then, unless -Status is given, it stops them all, optionally
  reinstalls the PATH copy from this checkout, restarts the web UI if it was
  running, and waits for the MCP server to come back on the new code.

  An MCP server is started by its client, not by this script: after the stop,
  reconnect it from the client (Claude Code: /mcp, select lynx, Reconnect).

.PARAMETER Status
  Only report; stop nothing.

.PARAMETER Reinstall
  Also reinstall the uv-tool copy on PATH from this checkout, so `lynx` in a
  terminal runs the same code as the MCP server.

.PARAMETER NoUI
  Do not restart the web UI even if it was running.

.PARAMETER WaitSeconds
  After the stop, wait up to this many seconds for every index to be claimed
  again by a process running this checkout's code (default 180; 0 = do not wait).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\restart-lynx.ps1 -Status

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\restart-lynx.ps1 -Reinstall
#>
[CmdletBinding()]
param(
    [switch]$Status,
    [switch]$Reinstall,
    [switch]$NoUI,
    [int]$WaitSeconds = 180
)

$ErrorActionPreference = 'Stop'
# Avast/AVG inject a TLS key-log path that crashes the bundled OpenSSL.
$env:SSLKEYLOGFILE = $null

$Repo = Split-Path -Parent $PSScriptRoot
$SrcPkg = Join-Path $Repo 'src\lynx'
$ConfigPath = Join-Path $Repo 'config.json'

function Write-Line([string]$text, [string]$color = 'Gray') {
    Write-Host $text -ForegroundColor $color
}

# ---------------------------------------------------------------------------
# The identity of a Lynx install: version, package path, code hash. Run with
# the interpreter of that install, so it reports what that install imports.
# Same algorithm as lynx.ownership.code_identity (sha256 of every .py, by
# relative path, CRLF folded, first 12 hex digits).
# ---------------------------------------------------------------------------
$identityPy = @'
import hashlib, json, sys
from pathlib import Path
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
try:
    import lynx
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
    sys.exit(0)
package = Path(lynx.__file__).resolve().parent
digest = hashlib.sha256()
newest = 0.0
for source in sorted(package.rglob("*.py")):
    digest.update(source.relative_to(package).as_posix().encode())
    try:
        digest.update(source.read_bytes().replace(b"\r\n", b"\n"))
        newest = max(newest, source.stat().st_mtime)
    except OSError:
        pass
print(json.dumps({"version": getattr(lynx, "__version__", "?"), "path": str(package),
                  "code": digest.hexdigest()[:12], "newest": newest}))
'@
$identityFile = Join-Path $env:TEMP 'lynx-identity.py'
Set-Content -Path $identityFile -Value $identityPy -Encoding ASCII

$identityCache = @{}
function Get-LynxIdentity([string]$python, [string]$srcDir = '') {
    $key = "$python|$srcDir"
    if ($identityCache.ContainsKey($key)) { return $identityCache[$key] }
    $result = $null
    if ($python -and (Test-Path $python)) {
        try {
            if ($srcDir) { $out = & $python $identityFile $srcDir 2>$null }
            else { $out = & $python $identityFile 2>$null }
            if ($out) { $result = ($out | Select-Object -Last 1) | ConvertFrom-Json }
        } catch { $result = $null }
    }
    $identityCache[$key] = $result
    return $result
}

function Get-UnixTime([datetime]$when) {
    return ([DateTimeOffset]$when.ToUniversalTime()).ToUnixTimeSeconds()
}

# ---------------------------------------------------------------------------
# What "new" means: the code of this checkout.
# ---------------------------------------------------------------------------
$pyForRepo = $null
foreach ($cand in @((Join-Path $Repo '.venv\Scripts\python.exe'))) {
    if (Test-Path $cand) { $pyForRepo = $cand; break }
}

# The MCP client's command for lynx, from Claude Code's config.
$mcpCommands = @()
$claudeJson = Join-Path $env:USERPROFILE '.claude.json'
if (Test-Path $claudeJson) {
    try {
        $cj = Get-Content $claudeJson -Raw -Encoding UTF8 | ConvertFrom-Json
        $entries = @()
        if ($cj.mcpServers -and $cj.mcpServers.lynx) { $entries += ,@('user', $cj.mcpServers.lynx) }
        if ($cj.projects) {
            foreach ($p in $cj.projects.PSObject.Properties) {
                $m = $p.Value.mcpServers
                if ($m -and $m.lynx) { $entries += ,@($p.Name, $m.lynx) }
            }
        }
        foreach ($e in $entries) {
            $mcpCommands += [pscustomobject]@{ Scope = $e[0]; Command = $e[1].command; Args = @($e[1].args) }
        }
    } catch { }
}
if (-not $pyForRepo) {
    $firstPy = $mcpCommands | Where-Object { $_.Command -match 'python(\.exe)?$' } | Select-Object -First 1
    if ($firstPy) { $pyForRepo = $firstPy.Command }
}
if (-not $pyForRepo) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { $pyForRepo = $py.Source }
}
if (-not $pyForRepo) { throw "No Python found to read this checkout's code identity." }

$repoId = Get-LynxIdentity $pyForRepo (Join-Path $Repo 'src')
if (-not $repoId -or $repoId.error) { throw "Cannot read this checkout's identity: $($repoId.error)" }
$repoNewest = [double]$repoId.newest

Write-Line ''
Write-Line "This checkout:  $Repo" 'Cyan'
Write-Line ("  code {0}  (last edit {1})" -f $repoId.code,
    ([DateTimeOffset]::FromUnixTimeSeconds([long]$repoNewest).LocalDateTime.ToString('yyyy-MM-dd HH:mm:ss'))) 'Cyan'

function Get-Verdict($identity, [datetime]$started) {
    if (-not $identity -or $identity.error) { return @('UNKNOWN', 'DarkYellow') }
    if ($identity.code -ne $repoId.code) { return @('OLD', 'Red') }
    if ((Get-UnixTime $started) -lt $repoNewest) { return @('STALE', 'Yellow') }
    return @('CURRENT', 'Green')
}

# ---------------------------------------------------------------------------
# What the MCP client will start on reconnect.
# ---------------------------------------------------------------------------
Write-Line ''
Write-Line 'What your MCP client starts for lynx:' 'White'
if (-not $mcpCommands) { Write-Line '  (no lynx entry in ~/.claude.json)' 'DarkGray' }
foreach ($c in $mcpCommands) {
    $id = $null
    if ($c.Command -match 'python(\.exe)?$') { $id = Get-LynxIdentity $c.Command }
    elseif ($c.Command -match 'lynx(\.exe)?$') {
        $toolPy = $null
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($uv) {
            $toolDir = (& $uv.Source tool dir 2>$null | Select-Object -First 1)
            if ($toolDir) { $toolPy = Join-Path $toolDir 'lynx-mcp\Scripts\python.exe' }
        }
        if ($toolPy) { $id = Get-LynxIdentity $toolPy }
    }
    $same = if ($id -and -not $id.error -and $id.code -eq $repoId.code) { 'same code as this checkout' } else { 'DIFFERENT code' }
    $color = if ($same -like 'same*') { 'Green' } else { 'Red' }
    Write-Line ("  [{0}] {1} {2}" -f $c.Scope, $c.Command, ($c.Args -join ' ')) 'Gray'
    if ($id -and -not $id.error) {
        Write-Line ("      -> lynx {0}, code {1}, from {2}: {3}" -f $id.version, $id.code, $id.path, $same) $color
    } else {
        Write-Line '      -> could not read which code it runs' 'DarkYellow'
    }
}

# The `lynx` a terminal runs.
$pathLynx = Get-Command lynx -ErrorAction SilentlyContinue
$uvToolPy = $null
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
    $toolDir = (& $uvCmd.Source tool dir 2>$null | Select-Object -First 1)
    if ($toolDir) {
        $cand = Join-Path $toolDir 'lynx-mcp\Scripts\python.exe'
        if (Test-Path $cand) { $uvToolPy = $cand }
    }
}
if ($pathLynx) {
    Write-Line ''
    Write-Line "The lynx on PATH: $($pathLynx.Source)" 'White'
    if ($uvToolPy) {
        $tid = Get-LynxIdentity $uvToolPy
        if ($tid -and -not $tid.error) {
            $same = $tid.code -eq $repoId.code
            Write-Line ("  -> lynx {0}, code {1}, from {2}: {3}" -f $tid.version, $tid.code, $tid.path,
                ($(if ($same) { 'same code as this checkout' } else { 'DIFFERENT code (use -Reinstall to align it)' }))) ($(if ($same) { 'Green' } else { 'Yellow' }))
        }
    }
}

# ---------------------------------------------------------------------------
# Running processes.
# ---------------------------------------------------------------------------
function Get-LynxProcesses {
    $me = $PID
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $me -and $_.CommandLine -and (
            $_.CommandLine -match '(-m\s+lynx\b)|(\\lynx(\.exe)?"?\s)|(\\lynx(\.exe)?"?$)'
        ) -and $_.CommandLine -notmatch 'restart-lynx\.ps1'
    }
}

function Get-Kind([string]$cmd) {
    if ($cmd -match 'manager\s+ui') { return 'web UI' }
    if ($cmd -match '\bserve\b') { return 'MCP server' }
    if ($cmd -match '-m\s+lynx\s*$' -or $cmd -match 'lynx(\.exe)?"?\s*$') { return 'MCP server' }
    if ($cmd -match '\b(build|reset|graph|manager)\b') { return 'CLI' }
    return 'lynx'
}

$procs = @(Get-LynxProcesses)
# A uv trampoline (lynx.exe) spawns the real python.exe as its child: report
# the child, which holds the code, and stop both.
Write-Line ''
Write-Line 'Running Lynx processes:' 'White'
$hadUI = $false
$uiCommand = $null
if (-not $procs) { Write-Line '  (none)' 'DarkGray' }
foreach ($p in $procs) {
    $kind = Get-Kind $p.CommandLine
    if ($kind -eq 'web UI') { $hadUI = $true; $uiCommand = $p.CommandLine }
    $exe = $p.ExecutablePath
    $id = $null
    if ($exe -match 'python(w)?\.exe$') { $id = Get-LynxIdentity $exe }
    elseif ($uvToolPy -and $exe -match 'lynx\.exe$') { $id = Get-LynxIdentity $uvToolPy }
    $started = $p.CreationDate
    $v = Get-Verdict $id $started
    $what = if ($id -and -not $id.error) { "lynx $($id.version), code $($id.code)" } else { 'code unknown' }
    Write-Line ("  pid {0,-6} {1,-10} started {2}  {3}  -> {4}" -f $p.ProcessId, $kind,
        $started.ToString('yyyy-MM-dd HH:mm:ss'), $what, $v[0]) $v[1]
    if ($v[0] -eq 'STALE') {
        Write-Line '               same install, but started before the last edit: it runs the old code from memory' 'DarkYellow'
    }
}

# ---------------------------------------------------------------------------
# Who owns each index, and on which code.
# ---------------------------------------------------------------------------
$storage = Join-Path $Repo 'rag_storage'
if (Test-Path $ConfigPath) {
    try {
        $cfg = Get-Content $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($cfg.storage_path) {
            $sp = [string]$cfg.storage_path
            if (-not [System.IO.Path]::IsPathRooted($sp)) { $sp = Join-Path $Repo $sp }
            $storage = $sp
        }
    } catch { }
}
function Get-Claims {
    $out = @()
    if (-not (Test-Path $storage)) { return $out }
    foreach ($d in Get-ChildItem $storage -Directory) {
        $f = Join-Path $d.FullName 'owner.json'
        if (Test-Path $f) {
            try {
                $c = Get-Content $f -Raw -Encoding UTF8 | ConvertFrom-Json
                # The owner rewrites its claim every 30 s; a pid alone can
                # belong to an unrelated process by now (Windows reuses them).
                $beatAge = (Get-UnixTime (Get-Date)) - [double]$c.beat
                $alive = ($null -ne (Get-Process -Id $c.pid -ErrorAction SilentlyContinue)) -and ($beatAge -lt 150)
                $out += [pscustomobject]@{ Source = $d.Name; Pid = $c.pid; Alive = $alive; Lynx = $c.lynx; Since = $c.since }
            } catch { }
        }
    }
    return $out
}
Write-Line ''
Write-Line "Index owners ($storage):" 'White'
$claims = @(Get-Claims)
if (-not $claims) { Write-Line '  (no index is claimed)' 'DarkGray' }
foreach ($c in $claims) {
    if (-not $c.Alive) {
        Write-Line ("  {0,-14} pid {1} is gone (stale claim, the next process takes it over)" -f $c.Source, $c.Pid) 'DarkGray'
        continue
    }
    $code = if ($c.Lynx) { $c.Lynx.code } else { '?' }
    $ok = $code -eq $repoId.code
    Write-Line ("  {0,-14} pid {1,-6} lynx {2}, code {3}: {4}" -f $c.Source, $c.Pid,
        ($(if ($c.Lynx) { $c.Lynx.version } else { '?' })), $code,
        ($(if ($ok) { 'this checkout' } else { 'OTHER code' }))) ($(if ($ok) { 'Green' } else { 'Red' }))
}

if ($Status) { Write-Line ''; return }

# ---------------------------------------------------------------------------
# Stop. A process killed in the middle of an index write can leave the store
# wedged, so wait for the stores to stop changing first.
# ---------------------------------------------------------------------------
if ($procs) {
    Write-Line ''
    Write-Line 'Waiting for index writes to settle...' 'White'
    $deadline = (Get-Date).AddSeconds(60)
    do {
        $busy = @()
        foreach ($d in (Get-ChildItem $storage -Directory -ErrorAction SilentlyContinue)) {
            foreach ($name in @('chroma.sqlite3-wal', 'chroma.sqlite3')) {
                $f = Join-Path $d.FullName $name
                if ((Test-Path $f) -and ((Get-Date) - (Get-Item $f).LastWriteTime).TotalSeconds -lt 5) { $busy += $d.Name }
            }
        }
        if ($busy) { Start-Sleep -Seconds 2 }
    } while ($busy -and (Get-Date) -lt $deadline)
    if ($busy) { Write-Line ("  still writing after 60 s: {0}. Stopping anyway; run 'lynx manager doctor' after." -f (($busy | Sort-Object -Unique) -join ', ')) 'Yellow' }

    Write-Line 'Stopping:' 'White'
    foreach ($p in $procs) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            Write-Line ("  stopped pid {0} ({1})" -f $p.ProcessId, (Get-Kind $p.CommandLine)) 'Gray'
        } catch {
            Write-Line ("  pid {0} already gone" -f $p.ProcessId) 'DarkGray'
        }
    }
    Start-Sleep -Seconds 1
    $left = @(Get-LynxProcesses)
    if ($left) { Write-Line ("  still running: {0}" -f (($left | ForEach-Object { $_.ProcessId }) -join ', ')) 'Yellow' }
}

# ---------------------------------------------------------------------------
# Align the PATH copy (only now: a running lynx.exe locks its own files).
# ---------------------------------------------------------------------------
if ($Reinstall) {
    if (-not $uvCmd) { Write-Line 'uv not found: cannot reinstall the PATH copy.' 'Yellow' }
    else {
        Write-Line ''
        Write-Line 'Reinstalling the PATH copy from this checkout (uv tool, no cache)...' 'White'
        # --no-cache: uv otherwise reuses the wheel it built for the same
        # version number, and installs the old code again without a word.
        & $uvCmd.Source tool install --force --reinstall --no-cache --system-certs --from $Repo lynx-mcp
        if ($uvToolPy) {
            $identityCache.Remove("$uvToolPy|") | Out-Null
            $tid = Get-LynxIdentity $uvToolPy
            if ($tid -and $tid.code -eq $repoId.code) { Write-Line "  PATH lynx now runs code $($tid.code): this checkout" 'Green' }
            else { Write-Line "  PATH lynx runs code $($tid.code), not $($repoId.code): check the uv output above" 'Red' }
        }
    }
}

# ---------------------------------------------------------------------------
# Start the web UI again, on this checkout's code.
# ---------------------------------------------------------------------------
if ($hadUI -and -not $NoUI) {
    $uiPython = ($mcpCommands | Where-Object { $_.Command -match 'python(\.exe)?$' } | Select-Object -First 1).Command
    if (-not $uiPython) { $uiPython = $pyForRepo }
    $uiId = Get-LynxIdentity $uiPython
    if ($uiId -and $uiId.code -eq $repoId.code) {
        Write-Line ''
        Write-Line "Starting the web UI with $uiPython ..." 'White'
        Start-Process -FilePath $uiPython -ArgumentList @('-m', 'lynx', 'manager', 'ui', '--config', "`"$ConfigPath`"") -WorkingDirectory $Repo
    } else {
        Write-Line ''
        Write-Line "Not restarting the web UI: $uiPython does not run this checkout's code. Start it by hand." 'Yellow'
    }
}

# ---------------------------------------------------------------------------
# Wait for the MCP server to come back on the new code.
# ---------------------------------------------------------------------------
Write-Line ''
Write-Line 'Now reconnect the MCP server from your client (Claude Code: /mcp, lynx, Reconnect).' 'Cyan'
if ($WaitSeconds -le 0) { return }
$sources = @()
if ($cfg -and $cfg.sources) { $sources = @($cfg.sources.PSObject.Properties | ForEach-Object { $_.Name }) }
if (-not $sources) { return }
Write-Line ("Waiting up to {0} s for every index to be claimed by this checkout's code..." -f $WaitSeconds) 'White'
$deadline = (Get-Date).AddSeconds($WaitSeconds)
do {
    $claims = @(Get-Claims)
    $missing = @($sources | Where-Object {
        $s = $_
        -not ($claims | Where-Object { $_.Source -eq $s -and $_.Alive -and $_.Lynx -and $_.Lynx.code -eq $repoId.code })
    })
    if ($missing) { Start-Sleep -Seconds 3 }
} while ($missing -and (Get-Date) -lt $deadline)
if ($missing) {
    Write-Line ("Not yet on the new code: {0}. Reconnect the client, then run this script with -Status." -f ($missing -join ', ')) 'Yellow'
} else {
    Write-Line ("Every index is served by code {0}: this checkout." -f $repoId.code) 'Green'
}
