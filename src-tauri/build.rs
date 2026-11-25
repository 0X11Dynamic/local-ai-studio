use std::path::PathBuf;

/// The Microsoft Visual C++ runtime DLLs the official llama.cpp Windows builds need.
/// They are not stored in git (Microsoft redistributable); the build copies them from
/// the build machine's System32 into `resources/vcrt/`, which `tauri.conf.json`
/// bundles into the installer. Missing files only produce a warning.
const VCRT_DLLS: [&str; 3] = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"];

fn ensure_vcrt() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let dest = manifest.join("resources").join("vcrt");
    let _ = std::fs::create_dir_all(&dest);
    let sys = PathBuf::from(std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into())).join("System32");
    for name in VCRT_DLLS {
        let target = dest.join(name);
        if target.exists() {
            continue;
        }
        let source = sys.join(name);
        if let Err(e) = std::fs::copy(&source, &target) {
            println!(
                "cargo:warning=could not copy {} into resources/vcrt ({e}); install the Microsoft Visual C++ 2015-2022 x64 Redistributable before building an installer",
                source.display()
            );
        }
    }
}

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        ensure_vcrt();
    }
    tauri_build::build()
}
