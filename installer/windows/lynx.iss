; lynx.iss — Inno Setup script for the Windows Lynx installer.
;
; Produces a single Lynx-Setup-<version>.exe that installs (no admin) into
; %LOCALAPPDATA%\Programs\Lynx, bundling only uv.exe + bootstrap.bat. The
; heavy Python stack is downloaded on first launch by bootstrap.bat.
;
; Build (on Windows, with uv.exe already staged next to this script):
;   ISCC /DMyAppVersion=0.9.0 lynx.iss
;
; Compiled with ISCC. SourceDir defaults to this script's folder, so the
; [Files] entries below are relative to installer/windows/.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#define MyAppName "Lynx"
#define MyAppPublisher "Lorenzo Cambiaghi"
#define MyAppURL "https://github.com/lorenzo-cambiaghi/LynxMCP"

[Setup]
AppId={{B7E6F3A2-5C4D-4E8B-9A1F-LYNXMCP00001}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={localappdata}\Programs\Lynx
DefaultGroupName=Lynx
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=Lynx-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#if FileExists("Lynx.ico")
SetupIconFile=Lynx.ico
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "uv.exe";        DestDir: "{app}"; Flags: ignoreversion
Source: "bootstrap.bat"; DestDir: "{app}"; Flags: ignoreversion
#if FileExists("Lynx.ico")
Source: "Lynx.ico";      DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
; Start Menu + optional desktop shortcut, both launching bootstrap.bat.
Name: "{group}\Lynx"; Filename: "{app}\bootstrap.bat"; WorkingDir: "{app}"; \
#if FileExists("Lynx.ico")
    IconFilename: "{app}\Lynx.ico"; \
#endif
    Comment: "Launch LynxManager (first run downloads the environment)"
Name: "{group}\Uninstall Lynx"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Lynx"; Filename: "{app}\bootstrap.bat"; WorkingDir: "{app}"; \
#if FileExists("Lynx.ico")
    IconFilename: "{app}\Lynx.ico"; \
#endif
    Tasks: desktopicon

[Run]
; Offer to launch right after install (unchecked-by-default would need
; postinstall flag; here we let the user opt out via the wizard checkbox).
Filename: "{app}\bootstrap.bat"; Description: "Launch Lynx now"; \
    Flags: postinstall nowait skipifsilent shellexec

[UninstallDelete]
; The downloaded environment + config live outside {app}; leave them by
; default (a reinstall reuses them). Users can delete %LOCALAPPDATA%\Lynx
; manually for a full wipe. Nothing extra to remove here.
