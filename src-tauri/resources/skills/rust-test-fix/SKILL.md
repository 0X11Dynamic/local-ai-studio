---
name: rust-test-fix
description: "Fix failing Rust tests or build errors with cargo, making the smallest correct change"
version: 1
keywords: [rust, cargo, test, tests, failing, compile, crate]
---
# Fixing failing Rust tests

1. Run `cargo test` in the project folder first (or `cargo test -p <crate>` in a workspace). Read the full output: the failing test name, the file and line, and the expected vs actual values.
2. Read the failing test and the function it calls before changing anything. Find the actual cause; do not guess from the test name.
3. Make the smallest change that fixes the cause. Fix the code, not the test, unless the test itself is clearly wrong; if you change a test, say why.
4. Prefer `fs_patch` with enough surrounding lines to match exactly once. Keep the existing style and formatting.
5. Run `cargo test` again. If something else fails, repeat from step 1. Stop after three attempts at the same failure and explain what you found.
6. For build errors, read the first error first; later errors are often caused by it.
7. In the final answer: say what was wrong, what you changed (file and function), and paste the final `test result:` line.
