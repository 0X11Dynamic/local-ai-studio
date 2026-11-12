---
name: safe-powershell
description: "Run PowerShell commands on Windows carefully: look first, quote paths, preview destructive commands"
version: 1
keywords: [powershell, windows, shell, command, script, delete, remove]
---
# Using PowerShell safely

- Look before you change: use `Get-ChildItem`, `Get-Content`, `Test-Path` and `Get-Item` to check what is there before writing, moving or deleting anything.
- Quote every path with single quotes (`'C:\My Folder\file.txt'`), and use `-LiteralPath` when a path may contain `[` or `]`.
- For destructive cmdlets (`Remove-Item`, `Move-Item`, `Set-Content` on existing files, `Stop-Process`), run the command with `-WhatIf` first, read what it would do, then run it for real.
- Never run `Remove-Item -Recurse` on a drive root, a user profile folder or one of the allowed root folders. Delete the specific files or subfolders instead. Prefer the fs_delete tool where it is offered: on this PC it moves things to the app's trash.
- Commands get no keyboard input. Add `-Confirm:$false` only where you have already checked the target, and `-Force` only for hidden or read-only files you meant to change.
- Do not download and run scripts from the internet, change the execution policy, edit the registry, or run anything as administrator.
- Check `$LASTEXITCODE` after running an external program, and report errors with their exact text.
