# Installer test: silent install, smoke suite against the installed exe,
# silent uninstall, and proof that Will's real app data is untouched.
#   powershell -NoProfile -File tests\e2e\installer.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$setup = Get-ChildItem "$root\target\release\bundle\nsis\*-setup.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $setup) { throw "no installer built" }
"installer: $($setup.Name) ($([math]::Round($setup.Length/1MB,2)) MB)"

# Fingerprint of the real app data (must not change).
$real = "$env:APPDATA\LocalAIStudio"
function Fingerprint { Get-ChildItem $real -Recurse -File -Force -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notlike "*\logs\*" } | ForEach-Object { "$($_.FullName)|$($_.Length)|$((Get-FileHash $_.FullName -Algorithm SHA256).Hash)" } | Sort-Object }
$before = Fingerprint
"real data files fingerprinted: $($before.Count)"

$t0 = Get-Date
$p = Start-Process $setup.FullName -ArgumentList '/S' -Wait -PassThru
"install exit code $($p.ExitCode) in $([math]::Round(((Get-Date)-$t0).TotalSeconds,1)) s"
$uninstKey = Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall | Where-Object { (Get-ItemProperty $_.PSPath).DisplayName -eq 'Local AI Studio' } | Select-Object -First 1
if (-not $uninstKey) { throw "no uninstall entry registered" }
$props = Get-ItemProperty $uninstKey.PSPath
$installDir = $props.InstallLocation.Trim('"')
if (-not $installDir) { $installDir = Split-Path -Parent ($props.UninstallString.Trim('"')) }
$exe = Get-ChildItem $installDir -Filter 'local-ai-studio.exe' -Recurse | Select-Object -First 1
"installed to: $installDir"
"installed size: $([math]::Round(((Get-ChildItem $installDir -Recurse -File | Measure-Object Length -Sum).Sum)/1MB,1)) MB"
$shortcut = Get-ChildItem "$env:APPDATA\Microsoft\Windows\Start Menu\Programs" -Recurse -Filter '*Local AI Studio*.lnk' -ErrorAction SilentlyContinue | Select-Object -First 1
"start menu shortcut: $(if ($shortcut) { $shortcut.FullName } else { 'none' })"

# Smoke: the phase 4 suite against the installed exe (isolated profile).
$env:LAS_E2E_EXE = $exe.FullName
Push-Location $root
$out = & node tests/e2e/phase4.mjs 2>&1 | Select-String -Pattern '^(FAIL|PASS)|passed\.' | ForEach-Object { $_.Line }
Pop-Location
Remove-Item Env:LAS_E2E_EXE
$out | Where-Object { $_ -match '^FAIL|passed\.' }

# Uninstall.
$un = $props.UninstallString.Trim('"')
$t1 = Get-Date
$q = Start-Process $un -ArgumentList '/S' -Wait -PassThru
Start-Sleep 3
"uninstall exit code $($q.ExitCode) in $([math]::Round(((Get-Date)-$t1).TotalSeconds,1)) s"
"program folder gone: $(-not (Test-Path $exe.FullName))"
"uninstall entry gone: $(-not (Test-Path $uninstKey.PSPath))"
"shortcut gone: $(if ($shortcut) { -not (Test-Path $shortcut.FullName) } else { 'n/a' })"
$after = Fingerprint
$diff = Compare-Object $before $after
"real app data unchanged: $(-not $diff)"
if ($diff) { $diff | Select-Object -First 10 }
