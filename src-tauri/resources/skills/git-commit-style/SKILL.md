---
name: git-commit-style
description: "Write clear git commits: small focused changes, imperative messages, never commit secrets"
version: 1
keywords: [git, commit, commits, message, branch, staging]
---
# Committing with git

- Only commit when the user asked for a commit. Never push unless they asked for that too.
- Run `git status` and `git diff` first, and commit only the changes that belong to the task. Leave unrelated changes unstaged and mention them.
- Before committing, check the staged diff for secrets: tokens, passwords, API keys, `.env` files, private keys. If you find one, stop and tell the user instead of committing.
- One commit per logical change. Do not mix a refactor with a bug fix.
- Message format: a short summary line in the imperative mood ("Fix overflow in parser", not "Fixed" or "Fixes"), at most about 60 characters, no full stop. Then a blank line and, if needed, a body that says why the change was made.
- Never rewrite history the user did not ask you to rewrite: no `--force`, `reset --hard` or `rebase` on shared branches without explicit approval.
- In the final answer, show the commit hash and its summary line.
