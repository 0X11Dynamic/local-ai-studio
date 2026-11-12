//! Adversarial path tests for This PC: every trick a model (or a file it
//! read) might use to reach outside the allowed folders. Windows only.
#![cfg(windows)]

use std::path::{Path, PathBuf};

use execution::local::LocalEnv;
use execution::paths::RootPolicy;
use execution::{ExecutionEnvironment, fs_tools};

struct Fx {
    _d: tempfile::TempDir,
    base: PathBuf,
    root: PathBuf,
    outside: PathBuf,
    env: LocalEnv,
}

fn fx() -> Fx {
    let d = tempfile::tempdir().unwrap();
    let base = d.path().to_path_buf();
    let root = base.join("work");
    let outside = base.join("outside_folder_with_a_long_name");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("secret.txt"), "TOP SECRET").unwrap();
    std::fs::write(root.join("notes.txt"), "hello").unwrap();
    let env = LocalEnv::new(RootPolicy::new(vec![root.clone()], vec![]), base.join("trash"));
    Fx { _d: d, base, root, outside, env }
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// The 8.3 short name of `p`, if the volume generates them.
fn short_name(p: &Path) -> Option<String> {
    let script = format!("(New-Object -ComObject Scripting.FileSystemObject).GetFolder('{}').ShortPath", s(p).replace('\'', "''"));
    let out = std::process::Command::new("powershell.exe").args(["-NoProfile", "-Command", &script]).output().ok()?;
    let short = String::from_utf8_lossy(&out.stdout).trim().to_string();
    // A real 8.3 path has a `~N` component; anything else means none was generated.
    (short.contains('~') && Path::new(&short).exists()).then_some(short)
}

fn junction(link: &Path, target: &Path) {
    let ok = std::process::Command::new("cmd").args(["/c", "mklink", "/J", &s(link), &s(target)]).output().unwrap().status.success();
    assert!(ok, "mklink /J failed");
}

#[tokio::test]
async fn short_names_case_and_dot_tricks_cannot_escape() {
    let f = fx();
    let p = &f.env.policy;
    // Protected folders by their 8.3 names.
    assert!(p.resolve(r"C:\PROGRA~1\x.txt").is_err(), "Program Files via PROGRA~1");
    assert!(p.resolve(r"C:\WINDOWS\SYSTEM32\..\win.ini").is_err());
    // A folder outside the root by its short name, absolute and via `..`.
    match short_name(&f.outside) {
        Some(short) => {
            eprintln!("note: outside folder short name {short}");
            assert!(std::fs::read_to_string(format!(r"{short}\secret.txt")).unwrap() == "TOP SECRET", "the short name really reaches the folder");
            assert!(p.resolve(&format!(r"{short}\secret.txt")).is_err(), "short name of an outside folder: {short}");
            assert!(f.env.read_file(&format!(r"{short}\secret.txt"), 100).await.is_err());
            let rel = format!(r"..\{}\secret.txt", Path::new(&short).file_name().unwrap().to_string_lossy());
            assert!(p.resolve(&rel).is_err(), "relative short name: {rel}");
        }
        None => eprintln!("note: this volume does not generate 8.3 names; outside short-name check skipped"),
    }
    // Case changes stay inside (Windows is case-insensitive) and do not open a way out.
    assert!(p.resolve(&s(&f.root).to_uppercase()).is_ok());
    assert!(p.resolve(&format!("{}\\..\\OUTSIDE_FOLDER_WITH_A_LONG_NAME\\secret.txt", s(&f.root).to_uppercase())).is_err());
    // `..` with mixed separators, `.` and doubled separators.
    for bad in [
        "../outside_folder_with_a_long_name/secret.txt",
        r"..\outside_folder_with_a_long_name\secret.txt",
        r"sub/..\..\outside_folder_with_a_long_name/secret.txt",
        r".\.\..\outside_folder_with_a_long_name\secret.txt",
        r"sub\\..\\..\\outside_folder_with_a_long_name\\secret.txt",
    ] {
        assert!(p.resolve(bad).is_err(), "{bad}");
    }
    // Trailing dots and spaces: Windows strips them, so `.. ` and `...` act like `..`.
    for bad in [r".. \outside_folder_with_a_long_name\secret.txt", r"..  \outside_folder_with_a_long_name\secret.txt", r"...\outside_folder_with_a_long_name\secret.txt", r"sub\.. \.. \outside_folder_with_a_long_name\secret.txt"] {
        let r = p.resolve(bad);
        assert!(r.is_err() || r.as_ref().unwrap().starts_with(&f.root), "{bad} -> {r:?}");
        assert!(f.env.read_file(bad, 100).await.map(|(b, _)| b != b"TOP SECRET").unwrap_or(true), "{bad} read the secret");
    }
    // `root\..\.` is the parent: refused.
    assert!(p.resolve(&format!(r"{}\..\.", s(&f.root))).is_err());
    // Drive-relative (`C:secret`) and root-relative (`\Windows`) forms.
    assert!(p.resolve(r"C:secret.txt").is_err());
    assert!(p.resolve(r"\Windows\win.ini").is_err());
    let _ = &f.base;
}

#[tokio::test]
async fn unc_device_and_verbatim_prefixes_are_refused_without_touching_the_network() {
    let f = fx();
    let p = &f.env.policy;
    for bad in [
        r"\\server\share\x.txt",
        "//server/share/x.txt",
        r"\/server\share\x.txt",
        r"/\server/share/x.txt",
        r"\\?\C:\Windows\win.ini",
        r"\\?\UNC\server\share\x",
        r"\\.\C:\Windows\win.ini",
        r"\\.\PhysicalDrive0",
        r"\\.\pipe\docker_engine",
        r"\??\C:\Windows\win.ini",
    ] {
        let t = std::time::Instant::now();
        let r = p.resolve(bad);
        assert!(r.is_err(), "{bad} -> {r:?}");
        assert!(t.elapsed().as_secs() < 2, "{bad} took {:?} (network lookup?)", t.elapsed());
    }
}

#[tokio::test]
async fn device_names_and_alternate_streams_are_refused_or_contained() {
    let f = fx();
    for dev in ["CON", "NUL", "COM1", "AUX", "PRN", "LPT1", "CON.txt", "nul.json", r"sub\COM1"] {
        let w = f.env.write_file(dev, b"x", true).await;
        let r = f.env.read_file(dev, 10).await;
        // Either refused, or a real file inside the root (newer Windows treat
        // device names in folders as ordinary names). Never the device itself.
        if w.is_ok() {
            assert!(f.root.join(dev).is_file() || r.is_ok(), "{dev} wrote somewhere outside the root");
        }
    }
    // Alternate data streams are refused (a hiding place, no legitimate use).
    let e = f.env.write_file("notes.txt:hidden", b"stream data", false).await;
    assert!(e.as_ref().is_err_and(|x| x.to_string().contains("alternate data stream")), "{e:?}");
    assert!(f.env.read_file("notes.txt:hidden", 100).await.is_err());
    assert!(f.env.read_file(&format!("{}::$DATA", s(&f.root.join("notes.txt"))), 100).await.is_err());
    assert_eq!(std::fs::read_to_string(f.root.join("notes.txt")).unwrap(), "hello", "main stream untouched");
    // ...and are no way to reach a file outside.
    let via = format!(r"..\{}\secret.txt:$DATA", f.outside.file_name().unwrap().to_string_lossy());
    assert!(f.env.read_file(&via, 100).await.is_err(), "{via}");
    assert!(f.env.read_file(&format!("{}::$DATA", s(&f.outside.join("secret.txt"))), 100).await.is_err());
}

#[tokio::test]
async fn long_and_unicode_paths_work_inside_the_root() {
    let f = fx();
    let mut deep = String::new();
    while deep.len() < 300 {
        deep.push_str("very_long_folder_name_segment_");
        deep.push('\\');
    }
    let file = format!("{deep}deep file.txt");
    assert!(s(&f.root.join(&file)).len() > 260);
    f.env.write_file(&file, b"deep", true).await.unwrap();
    assert_eq!(f.env.read_file(&file, 10).await.unwrap().0, b"deep");
    assert!(!f.env.list_dir(&deep).await.unwrap().is_empty());
    for name in ["résumé 📄.txt", "שלום.txt", "日本語 ファイル.md", "emoji 🔥🔥.txt"] {
        f.env.write_file(name, name.as_bytes(), false).await.unwrap();
        assert_eq!(f.env.read_file(name, 100).await.unwrap().0, name.as_bytes());
    }
    let listed = fs_tools::list(&f.env, &s(&f.root)).await.unwrap();
    assert!(listed.contains("emoji 🔥🔥.txt") && listed.contains("שלום.txt"), "{listed}");
    // Delete of a long path goes to the trash too.
    let moved = f.env.remove(&file, false).await.unwrap();
    assert!(Path::new(&moved).exists());
}

#[tokio::test]
async fn junctions_and_symlinks_out_of_the_root_are_refused_everywhere() {
    let f = fx();
    let link = f.root.join("link");
    junction(&link, &f.outside);
    let inside_link = s(&link.join("secret.txt"));
    assert!(f.env.read_file(&inside_link, 100).await.is_err(), "read through a junction");
    assert!(f.env.write_file(&s(&link.join("new.txt")), b"x", false).await.is_err(), "write through a junction");
    assert!(!f.outside.join("new.txt").exists());
    assert!(f.env.list_dir(&s(&link)).await.is_err(), "list through a junction");
    assert!(f.env.rename(&inside_link, &s(&f.root.join("stolen.txt"))).await.is_err(), "move out of a junction");
    assert!(f.env.rename(&s(&f.root.join("notes.txt")), &s(&link.join("planted.txt"))).await.is_err(), "move into a junction");
    assert!(f.env.remove(&inside_link, false).await.is_err(), "delete through a junction");
    assert!(f.env.remove(&s(&link), true).await.is_err(), "deleting the junction would follow it");
    assert!(f.outside.join("secret.txt").exists(), "outside file untouched");
    // The search tool skips the junction.
    let found = fs_tools::search(&f.env, &s(&f.root), "TOP SECRET", None).await.unwrap();
    assert!(!found.contains("secret.txt"), "{found}");
    std::process::Command::new("cmd").args(["/c", "rmdir", &s(&link)]).output().unwrap();
    // A file symlink (needs Developer Mode or admin; skipped if not allowed).
    let sl = f.root.join("sym.txt");
    if std::os::windows::fs::symlink_file(f.outside.join("secret.txt"), &sl).is_ok() {
        assert!(f.env.read_file(&s(&sl), 100).await.is_err(), "read through a file symlink");
        let _ = std::fs::remove_file(&sl);
    } else {
        eprintln!("note: file symlinks not permitted here (no Developer Mode); symlink check skipped");
    }
}

#[tokio::test]
async fn concurrent_writes_are_atomic() {
    let f = fx();
    let env = std::sync::Arc::new(f.env);
    let a = "A".repeat(2_000_000);
    let b = "B".repeat(1_000);
    let mut handles = vec![];
    for i in 0..16 {
        let env = env.clone();
        let content = if i % 2 == 0 { a.clone() } else { b.clone() };
        handles.push(tokio::spawn(async move { env.write_file("race.txt", content.as_bytes(), false).await }));
    }
    let mut ok = 0;
    for h in handles {
        if h.await.unwrap().is_ok() {
            ok += 1;
        }
    }
    let got = std::fs::read_to_string(f.root.join("race.txt")).unwrap();
    assert!(got == a || got == b, "torn write: {} bytes, starts {:?}", got.len(), &got[..10.min(got.len())]);
    assert_eq!(ok, 16, "every concurrent write should succeed (one wins)");
    let leftovers: Vec<_> = std::fs::read_dir(&f.root).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains("las-tmp")).collect();
    assert!(leftovers.is_empty(), "temp files left: {leftovers:?}");
}

#[tokio::test]
async fn reads_patches_moves_and_trash_restore() {
    let f = fx();
    let big = "x".repeat(10 * 1024 * 1024);
    std::fs::write(f.root.join("big.txt"), &big).unwrap();
    let r = fs_tools::read(&f.env, "big.txt", None, None).await.unwrap();
    assert!(r.len() < 300 * 1024 && r.contains("[truncated"), "{} bytes", r.len());
    std::fs::write(f.root.join("bin.dat"), [0u8, 159, 146, 150, 0, 1]).unwrap();
    assert!(fs_tools::read(&f.env, "bin.dat", None, None).await.is_err());
    std::fs::write(f.root.join("dup.txt"), "same\nsame\n").unwrap();
    assert!(fs_tools::patch(&f.env, "dup.txt", "same", "x", false).await.is_err());
    std::fs::write(f.root.join("taken.txt"), "keep me").unwrap();
    assert!(fs_tools::move_path(&f.env, "notes.txt", "taken.txt").await.is_err());
    assert_eq!(std::fs::read_to_string(f.root.join("taken.txt")).unwrap(), "keep me");
    // Delete -> trash -> the user can move it back.
    let msg = fs_tools::delete(&f.env, "notes.txt", false).await.unwrap();
    assert!(!f.root.join("notes.txt").exists() && msg.contains("trash"));
    let trashed = msg.split('(').nth(1).and_then(|t| t.split(')').next()).unwrap().to_string();
    assert!(Path::new(&trashed).exists(), "{trashed}");
    std::fs::rename(&trashed, f.root.join("notes.txt")).unwrap();
    assert_eq!(std::fs::read_to_string(f.root.join("notes.txt")).unwrap(), "hello");
    // The trash folder itself is not reachable through the tools.
    assert!(f.env.read_file(&trashed, 10).await.is_err());
}
