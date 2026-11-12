//! File tools built on any `ExecutionEnvironment`: list, read, search,
//! write (atomic, returns a diff), patch (one unique replacement), move and
//! delete (to the trash on this PC). Results are plain text for the model.

use app_core::{AppError, AppResult};
use serde::Serialize;

use crate::ExecutionEnvironment;

pub const MAX_READ_BYTES: usize = 256 * 1024;
const MAX_SEARCH_FILES: usize = 5000;
const MAX_SEARCH_HITS: usize = 200;
const MAX_SEARCH_FILE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct WriteOutcome {
    pub path: String,
    pub created: bool,
    pub bytes: usize,
    /// Unified diff of the change (empty when the content was identical).
    pub diff: String,
}

pub fn unified_diff(path: &str, old: &str, new: &str) -> String {
    similar::TextDiff::from_lines(old, new).unified_diff().context_radius(3).header(&format!("a/{path}"), &format!("b/{path}")).to_string()
}

fn as_text(bytes: Vec<u8>, path: &str) -> AppResult<String> {
    if bytes.iter().take(8192).any(|&b| b == 0) {
        return Err(AppError::InvalidInput(format!("{path} looks like a binary file; not reading it as text")));
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

pub async fn list(env: &dyn ExecutionEnvironment, path: &str) -> AppResult<String> {
    let entries = env.list_dir(path).await?;
    if entries.is_empty() {
        return Ok(format!("{path} is empty"));
    }
    let mut out = String::new();
    for e in &entries {
        if e.is_dir {
            out.push_str(&format!("{}/\n", e.name));
        } else {
            out.push_str(&format!("{}  ({} bytes)\n", e.name, e.size));
        }
    }
    Ok(out)
}

/// Read a text file, optionally a 1-based line range.
pub async fn read(env: &dyn ExecutionEnvironment, path: &str, start_line: Option<usize>, max_lines: Option<usize>) -> AppResult<String> {
    let (bytes, truncated) = env.read_file(path, MAX_READ_BYTES).await?;
    let text = as_text(bytes, path)?;
    let mut out = if start_line.is_some() || max_lines.is_some() {
        let start = start_line.unwrap_or(1).max(1);
        let n = max_lines.unwrap_or(usize::MAX);
        text.lines().skip(start - 1).take(n).collect::<Vec<_>>().join("\n")
    } else {
        text
    };
    if truncated {
        out.push_str(&format!("\n[truncated: only the first {} KB were read]", MAX_READ_BYTES / 1024));
    }
    Ok(out)
}

/// Regex search over text files under `path` (recursive). Returns
/// `file:line: text` lines.
pub async fn search(env: &dyn ExecutionEnvironment, path: &str, pattern: &str, file_glob: Option<&str>) -> AppResult<String> {
    let re = regex::RegexBuilder::new(pattern).size_limit(1 << 20).build().map_err(|e| AppError::InvalidInput(format!("bad pattern: {e}")))?;
    let glob = file_glob.map(glob_to_regex).transpose()?;
    let mut stack = vec![path.to_string()];
    let (mut files, mut hits) = (0usize, Vec::<String>::new());
    let mut capped = false;
    while let Some(dir) = stack.pop() {
        let entries = match env.list_dir(&dir).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        for e in entries {
            if e.is_dir {
                if !matches!(e.name.as_str(), ".git" | "node_modules" | "target" | ".venv" | "__pycache__") {
                    stack.push(e.path);
                }
                continue;
            }
            if glob.as_ref().is_some_and(|g| !g.is_match(&e.name)) || e.size as usize > MAX_SEARCH_FILE_BYTES {
                continue;
            }
            files += 1;
            if files > MAX_SEARCH_FILES {
                capped = true;
                break;
            }
            let Ok((bytes, _)) = env.read_file(&e.path, MAX_SEARCH_FILE_BYTES).await else { continue };
            let Ok(text) = as_text(bytes, &e.path) else { continue };
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    let shown: String = line.chars().take(300).collect();
                    hits.push(format!("{}:{}: {}", e.path, i + 1, shown.trim_end()));
                    if hits.len() >= MAX_SEARCH_HITS {
                        capped = true;
                        break;
                    }
                }
            }
            if capped {
                break;
            }
        }
        if capped {
            break;
        }
    }
    let mut out = if hits.is_empty() { format!("no matches for /{pattern}/ in {files} files") } else { hits.join("\n") };
    if capped {
        out.push_str("\n[stopped early: too many files or matches; narrow the path or pattern]");
    }
    Ok(out)
}

/// `*.rs`, `notes-?.md` style name globs (no path separators).
fn glob_to_regex(g: &str) -> AppResult<regex::Regex> {
    let mut s = String::from("(?i)^");
    for c in g.chars() {
        match c {
            '*' => s.push_str(".*"),
            '?' => s.push('.'),
            c => s.push_str(&regex::escape(&c.to_string())),
        }
    }
    s.push('$');
    regex::Regex::new(&s).map_err(|e| AppError::InvalidInput(format!("bad file pattern: {e}")))
}

pub async fn write(env: &dyn ExecutionEnvironment, path: &str, content: &str, create_dirs: bool) -> AppResult<WriteOutcome> {
    let old = match env.stat(path).await? {
        Some(e) if e.is_dir => return Err(AppError::InvalidInput(format!("{path} is a folder"))),
        Some(_) => Some(as_text(env.read_file(path, MAX_SEARCH_FILE_BYTES * 4).await?.0, path).unwrap_or_default()),
        None => None,
    };
    env.write_file(path, content.as_bytes(), create_dirs).await?;
    Ok(WriteOutcome {
        path: path.to_string(),
        created: old.is_none(),
        bytes: content.len(),
        diff: unified_diff(path, old.as_deref().unwrap_or(""), content),
    })
}

/// Replace exactly one occurrence of `old_text` (or every one with `all`).
pub async fn patch(env: &dyn ExecutionEnvironment, path: &str, old_text: &str, new_text: &str, all: bool) -> AppResult<WriteOutcome> {
    if old_text.is_empty() {
        return Err(AppError::InvalidInput("old_text is empty".into()));
    }
    let (bytes, truncated) = env.read_file(path, MAX_SEARCH_FILE_BYTES * 4).await?;
    if truncated {
        return Err(AppError::InvalidInput(format!("{path} is too large to patch")));
    }
    let text = as_text(bytes, path)?;
    let count = text.matches(old_text).count();
    let updated = match (count, all) {
        (0, _) => return Err(AppError::NotFound(format!("old_text was not found in {path}; read the file and copy the exact text, including whitespace"))),
        (1, _) | (_, true) => text.replace(old_text, new_text),
        (n, false) => return Err(AppError::Conflict(format!("old_text occurs {n} times in {path}; include more surrounding text so it is unique, or set all=true"))),
    };
    env.write_file(path, updated.as_bytes(), false).await?;
    Ok(WriteOutcome { path: path.to_string(), created: false, bytes: updated.len(), diff: unified_diff(path, &text, &updated) })
}

pub async fn move_path(env: &dyn ExecutionEnvironment, from: &str, to: &str) -> AppResult<String> {
    env.rename(from, to).await?;
    Ok(format!("moved {from} to {to}"))
}

pub async fn delete(env: &dyn ExecutionEnvironment, path: &str, recursive: bool) -> AppResult<String> {
    let where_ = env.remove(path, recursive).await?;
    Ok(if where_.is_empty() { format!("deleted {path}") } else { format!("moved {path} to the trash ({where_}); it can be restored from there") })
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::local::LocalEnv;
    use crate::paths::RootPolicy;

    fn env() -> (tempfile::TempDir, LocalEnv, String) {
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("work");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src").join("main.rs"), "fn main() {\n    println!(\"hi\");\n}\n").unwrap();
        std::fs::write(root.join("README.md"), "hello world\nhello again\n").unwrap();
        std::fs::write(root.join("blob.bin"), [0u8, 1, 2, 0, 5]).unwrap();
        let r = root.to_string_lossy().into_owned();
        (d, LocalEnv::new(RootPolicy::new(vec![root], vec![]), std::env::temp_dir().join("las-test-trash")), r)
    }

    #[tokio::test]
    async fn list_read_search() {
        let (_d, e, root) = env();
        let l = list(&e, &root).await.unwrap();
        assert!(l.starts_with("src/"), "folders first: {l}");
        assert!(read(&e, "README.md", Some(2), Some(1)).await.unwrap() == "hello again");
        assert!(read(&e, "blob.bin", None, None).await.is_err(), "binary refused");
        let s = search(&e, &root, "println", Some("*.rs")).await.unwrap();
        assert!(s.contains("main.rs:2:"), "{s}");
        assert!(search(&e, &root, "nothing-here", None).await.unwrap().starts_with("no matches"));
        assert!(search(&e, &root, "(", None).await.is_err(), "bad regex");
    }

    #[tokio::test]
    async fn write_patch_with_diffs() {
        let (_d, e, _root) = env();
        let w = write(&e, "new/notes.txt", "a\nb\n", true).await.unwrap();
        assert!(w.created && w.diff.contains("+a"), "{w:?}");
        let p = patch(&e, "new/notes.txt", "b", "B", false).await.unwrap();
        assert!(p.diff.contains("-b") && p.diff.contains("+B"), "{}", p.diff);
        assert!(patch(&e, "README.md", "hello", "bye", false).await.is_err(), "ambiguous");
        assert!(patch(&e, "README.md", "absent", "x", false).await.is_err());
        let all = patch(&e, "README.md", "hello", "bye", true).await.unwrap();
        assert_eq!(read(&e, "README.md", None, None).await.unwrap(), "bye world\nbye again\n", "{all:?}");
        assert!(move_path(&e, "new/notes.txt", "new/renamed.txt").await.is_ok());
        assert!(delete(&e, "new/renamed.txt", false).await.unwrap().contains("trash"));
    }

    #[tokio::test]
    async fn git_data_is_refused_for_writes_moves_and_deletes() {
        let (_d, e, root) = env();
        let git = std::path::Path::new(&root).join(".git");
        std::fs::create_dir_all(git.join("hooks")).unwrap();
        std::fs::write(git.join("config"), "[core]\n").unwrap();
        assert!(write(&e, ".git/hooks/post-checkout", "#!/bin/sh\necho pwned\n", true).await.is_err());
        assert!(!git.join("hooks").join("post-checkout").exists());
        assert!(write(&e, ".GIT./config", "x", false).await.is_err(), "Windows spellings of .git");
        assert!(patch(&e, ".git/config", "[core]", "[core]\n\tfsmonitor = evil.exe", false).await.is_err());
        assert!(move_path(&e, "README.md", ".git/hooks/pre-commit").await.is_err());
        assert!(move_path(&e, ".git", "not-git").await.is_err());
        assert!(delete(&e, ".git/config", false).await.is_err());
        assert!(delete(&e, ".git", true).await.is_err());
        assert_eq!(std::fs::read_to_string(git.join("config")).unwrap(), "[core]\n");
        // Reading stays allowed; the repository folder itself is not moved or deleted.
        assert!(read(&e, ".git/config", None, None).await.is_ok());
        std::fs::create_dir_all(std::path::Path::new(&root).join("sub").join(".git")).unwrap();
        assert!(delete(&e, "sub", true).await.is_err());
        assert!(write(&e, "src/new.rs", "fn x() {}\n", false).await.is_ok(), "normal files are unaffected");
    }
}
