Microsoft Visual C++ runtime (x64), app-local copies
=====================================================
Files: vcruntime140.dll, vcruntime140_1.dll, msvcp140.dll
Source: the DLLs are NOT stored in git. src-tauri/build.rs copies them unmodified from
%SystemRoot%\System32 into this folder when they are missing (it only warns if they are
absent). Building therefore needs the Microsoft Visual C++ 2015-2022 Redistributable (x64)
installed on the build machine. tauri.conf.json (bundle.resources) ships this folder in the installer.

Why: the official llama.cpp Windows builds (llama-server.exe, ggml-rpc-server.exe)
need these DLLs. A fresh Windows has none and the process exits with 0xC0000135
(STATUS_DLL_NOT_FOUND). Local AI Studio copies them next to a runtime's executable
(only when same-named files are not already there).

License: these files are redistributable under the Microsoft Visual C++ Redistributable
license terms (the "Distributable Code" / app-local deployment terms for the
VC++ runtime files vcruntime/msvcp), see
https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files
and https://visualstudio.microsoft.com/license-terms/vs2022-ga-vc-redist/ .
