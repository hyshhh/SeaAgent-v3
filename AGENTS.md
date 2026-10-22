# Repository instructions

## Active project

- The maintained application is `SeaAgent-v3.2-based_DA`. Treat `SeaAgent-v3.1-based_LG` and archived documents as historical references; do not update them unless the user asks.
- Keep edits scoped to the requested project and task. Do not include unrelated workspace changes in a commit.

## Model-facing prompts and tools

- When changing model behavior, inspect all relevant sources: role prompts, task prompts, skill descriptions and bodies, tool/argument schemas, VLM prompts, middleware reminders, and summary prompts.
- Keep instructions concise and role-specific. State each invariant once where possible; align every prompt and tool description with the current schema and implementation.
- Preserve complete tool capability, argument dependencies, evidence provenance, uncertainty behavior, and safety/confirmation requirements when shortening descriptions.
- Validate prompt/config changes against the actual tool and shared-state schemas. Keep regression tests for critical semantics such as time scope, pagination, and gallery input shape.

## Checks

- After changes, run relevant focused tests, then the full project test suite, Ruff, and Python compile checks from `SeaAgent-v3.2-based_DA` using its `.venv`.
- Review `git diff` and `git status` before committing; include only files related to the requested work.

## GitHub synchronization

- Standing user preference: after an authorized project improvement is implemented and checks pass, commit the task-specific changes directly to `main` and push to `origin/main` without asking again.
- Before committing, fetch and synchronize with `origin/main`; preserve remote and local work. Do not force-push, rewrite history, or commit unrelated changes.
- If the user explicitly asks to pause or skip pushing, follow that request. If sync or push fails, keep the local changes and report the exact blocker.
