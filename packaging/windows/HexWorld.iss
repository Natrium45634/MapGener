; Inno Setup script of the HexWorld installer.
; Build (from the repository root, after PyInstaller):
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAppVersion=1.0.0 packaging\windows\HexWorld.iss
; Result: packaging\windows\Output\HexWorld-<version>-Setup.exe

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "HexWorld"
#define AppExe "HexWorld.exe"
#define SrcDir "..\..\dist\HexWorld"

[Setup]
AppId={{6E3B1C52-4A1F-4F7D-9C3E-2B8E5D0A7F41}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=HexWorld
VersionInfoVersion={#AppVersion}
VersionInfoDescription=HexWorld — генератор карт миров
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; installs without administrator rights (for the current user) unless the user chooses
; «для всех пользователей»
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=Output
OutputBaseFilename=HexWorld-{#AppVersion}-Setup
SetupIconFile=build\hexworld.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=auto

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
russian.DesktopIcon=Создать значок на рабочем столе
english.DesktopIcon=Create a desktop shortcut
russian.RunApp=Запустить HexWorld
english.RunApp=Launch HexWorld
russian.Guide=Как пользоваться HexWorld
english.Guide=How to use HexWorld

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README_WINDOWS.txt"; DestDir: "{app}"; DestName: "Как пользоваться.txt"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:Guide}"; Filename: "{app}\Как пользоваться.txt"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:RunApp}"; Flags: nowait postinstall skipifsilent
