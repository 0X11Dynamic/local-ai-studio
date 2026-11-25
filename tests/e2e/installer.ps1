# Installer test: silent install, desktop + Start menu shortcuts, smoke suite
# against the installed exe, silent uninstall (shortcuts removed), and proof
# that the real app data is untouched.
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

# Desktop shortcut (the installer creates it by default; silent installs too).
$desktopLnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Local AI Studio.lnk'
if (Test-Path $desktopLnk) { throw "a '$desktopLnk' already exists; remove it first so the test can tell the installer made it" }
$fails = @()

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
if (-not $shortcut) { $fails += 'no Start menu shortcut' }
$desktopTarget = if (Test-Path $desktopLnk) { (New-Object -ComObject WScript.Shell).CreateShortcut($desktopLnk).TargetPath } else { $null }
"desktop shortcut: $(if ($desktopTarget) { "$desktopLnk -> $desktopTarget" } else { 'none' })"
if (-not $desktopTarget) { $fails += 'no desktop shortcut' }
elseif ($desktopTarget -ne $exe.FullName) { $fails += "desktop shortcut points at $desktopTarget, not $($exe.FullName)" }

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
if ($shortcut -and (Test-Path $shortcut.FullName)) { $fails += 'Start menu shortcut left after uninstall' }
"desktop shortcut gone: $(-not (Test-Path $desktopLnk))"
if (Test-Path $desktopLnk) { $fails += 'desktop shortcut left after uninstall' }
$after = Fingerprint
$diff = Compare-Object $before $after
"real app data unchanged: $(-not $diff)"
if ($diff) { $diff | Select-Object -First 10; $fails += 'real app data changed' }
if ($out | Where-Object { $_ -match '^FAIL' }) { $fails += 'phase4 smoke had failures' }
if ($fails.Count) { "INSTALLER TEST FAILED: $($fails -join '; ')"; exit 1 }
"INSTALLER TEST PASSED"
