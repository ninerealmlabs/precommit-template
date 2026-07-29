---
name: precommit-template
description: >
  Apply and maintain the ninerealmlabs pre-commit Copier template in a repository.
  Covers generating a config with `copier copy`, answering the survey, updating with `copier update`,
  resolving `.rej` conflicts, and the tool configs the template emits (ruff, biome, prettier, mdformat,
  rumdl, yamllint, typos, shellcheck, hadolint, editorconfig, commitizen).
  Use when adding, updating, or debugging pre-commit hooks in a repo that is (or should be) managed by
  this template.
license: CC0-1.0
compatibility: Requires copier >=9 and either prek or pre-commit. The prettier hook needs prettier on PATH.
metadata:
  author: ninerealmlabs
  repository: https://github.com/ninerealmlabs/precommit-template
  tags:
    - pre-commit
    - copier
    - linting
    - formatting
---

# Pre-commit Template

A Copier template that generates a `.pre-commit-config.yaml` plus matching tool configuration files for a repository.
Hooks are declared once and read by both [prek](https://github.com/j178/prek) (the default runner) and
[pre-commit](https://pre-commit.com/).

## Quick start

```bash
# One-time tool install
uv tool install copier --with copier-templates-extensions --with jinja2-time
uv tool install prek

# Generate into an existing repo (run from the repo root)
copier copy --trust "gh:ninerealmlabs/precommit-template" "$(git rev-parse --show-toplevel)"

prek run --all-files   # fix whatever the new rules flag
prek install           # run the checks on every commit
```

`--trust` is mandatory.
`copier.yaml` loads `extensions/detect.py` as a Jinja extension, and Copier refuses to execute extension code without it.

## Skill directory structure

```text
skills/precommit-template/
├── SKILL.md                     ← This file
└── references/
    ├── survey-questions.md      ← Every question and what it generates
    ├── generated-files.md       ← What lands in the target repo
    └── troubleshooting.md       ← Error patterns and fixes
```

## When to use what

| Need                                     | Use                                                         |
| ---------------------------------------- | ----------------------------------------------------------- |
| Add hooks to a repo for the first time   | `copier copy --trust gh:ninerealmlabs/precommit-template .` |
| Pull in newer template revisions         | `copier update --trust --answers-file .copier-answers.yaml` |
| Re-render with the same answers          | `copier recopy --trust --answers-file .copier-answers.yaml` |
| Change an answer (e.g. biome → prettier) | `copier update --trust` and re-answer at the prompt         |
| Run every hook now                       | `prek run --all-files`                                      |
| Run one hook                             | `prek run --all-files <hook-id>`                            |
| Install the git hooks                    | `prek install`                                              |
| Skip a hook for one commit               | `SKIP=<hook-id> git commit ...`                             |

## Core concepts

### The answers file

`copier copy` writes `.copier-answers.yaml` recording the template version and every answer.
`copier update` reads it to replay your choices, so **never edit it by hand** — a stale or hand-tweaked answers file makes the next update produce conflicts that look like template bugs.

### Detection instead of questions

Two values are worked out from the target repo rather than asked:

| Value             | Detected from                                                                                    | Fallback |
| ----------------- | ------------------------------------------------------------------------------------------------ | -------- |
| hook runner       | installed git hook shims, a `prek.toml`, how CI invokes the runner, or which runner is on `PATH` | `prek`   |
| `web_format_tool` | an existing `.pre-commit-config.yaml`, biome/prettier config files, or `package.json` deps       | `biome`  |

The hook runner is **not** a survey question and is **not** recorded in `.copier-answers.yaml`.
Both runners read the same `.pre-commit-config.yaml`, so the detected value only decides which runner the generated comments and post-copy messages name.
It is recomputed on every run, so it follows the repo if you switch runners.

`web_format_tool` **is** a question, because biome and prettier need genuinely different config files.
Detection only pre-selects the prompt's default; the answer is recorded and reused on every update.

### Conditional file generation

Every optional file is named with a Jinja conditional, so answering `no` means the file is never written:

```text
template/{% if python %}.ruff.toml{% endif %}.jinja
template/{% if web_format and web_format_tool == "prettier" %}.prettierrc.yaml{% endif %}.jinja
```

Full list in [references/generated-files.md](references/generated-files.md).

### Externally-installed tools

The hook runner manages its own environments for most hooks, but one tool must already be on `PATH`:

| Tool       | Required when               | Install                   |
| ---------- | --------------------------- | ------------------------- |
| `prettier` | `web_format_tool: prettier` | `npm install -g prettier` |

Without them the corresponding hooks fail with "command not found", not with a config error.

## Workflows

### Adopting the template in an existing repo

1. Commit or stash everything — Copier refuses to run against a dirty working tree.
2. Run `copier copy --trust "gh:ninerealmlabs/precommit-template" "$(git rev-parse --show-toplevel)"`.
3. Answer the survey (see [references/survey-questions.md](references/survey-questions.md)).
4. Review the generated files and the auto-created commit.
5. Run `prek run --all-files`.
   Expect a large diff on first run: the formatters rewrite the whole repo.
6. Commit the formatting churn separately from the config so review stays readable.
7. Run `prek install`.

### Updating to a newer template revision

1. Check out a feature branch — updates can touch many files.
2. Commit or stash current work.
3. Run `copier update --trust --answers-file .copier-answers.yaml`.
4. Resolve any `*.rej` files, then delete them.
   A `forbid-rej` hook blocks commits while they exist, which is deliberate: an unresolved `.rej` means template changes were silently dropped.
5. Run `prek run --all-files` and commit.

### Changing an answer

Re-run `copier update --trust` and give a different answer at the prompt.
Files gated on the old answer are deleted and files gated on the new one are created.
Local edits to a file that gets deleted are lost, so move anything worth keeping out of the way first.

## Reference files

- [references/survey-questions.md](references/survey-questions.md) — every question, its default, and the
  files and hooks it controls.
- [references/generated-files.md](references/generated-files.md) — the full map from answer to generated
  file, and which hooks read each config.
- [references/troubleshooting.md](references/troubleshooting.md) — error patterns and fixes for failed
  copies, updates, and hook runs.

## Gotchas

1. **`--trust` is not optional.**
   Without it Copier skips `extensions/detect.py` and the survey aborts on an undefined `detect_hook_runner`.
2. **Never hand-edit `.copier-answers.yaml`.**
   It is Copier's record of what was rendered; editing it desynchronizes the next update.
3. **Clean working tree required.**
   Both `copier copy` and `copier update` refuse to run with uncommitted changes.
4. **`.rej` files block commits.**
   That is the `forbid-rej` hook doing its job — resolve the rejected hunk rather than deleting the file unread.
5. **YAML files must use `.yaml`.**
   A `forbid-yml` hook fails on any `.yml` extension.
   The only carve-outs are `.copier-answers.yml` and the generator's own `great-docs.yml`.
6. **Commit messages must not carry a `Co-authored-by:` trailer.**
   Use an `AI-assistant:` trailer instead; a `commit-msg` hook enforces this.
7. **First run reformats everything.**
   Land the formatting churn in its own commit so the config change stays reviewable.
8. **`.typos.toml` may need to be committed before the `typos` hook can read it.**
   If the hook errors on a fresh copy, commit the config first, then re-run.

## Capabilities and boundaries

**What agents can do:**

- Run `copier copy`, `copier update`, and `copier recopy` in a repo
- Answer or re-answer the survey
- Resolve `.rej` conflicts
- Tune the generated tool configs (`.ruff.toml`, `.rumdl.toml`, `.typos.toml`, and friends)
- Add repo-local hooks to the generated `.pre-commit-config.yaml`

**Requires human setup:**

- Installing `copier`, a hook runner, and `prettier` if selected as the web formatter
- Deciding whether to accept a breaking template revision
- Reviewing the first bulk-formatting commit

## Resources

- [Documentation](https://ninerealmlabs.github.io/precommit-template/)
- [llms.txt](https://ninerealmlabs.github.io/precommit-template/llms.txt) — indexed overview for agents
- [llms-full.txt](https://ninerealmlabs.github.io/precommit-template/llms-full.txt) — full context for agents
- [GitHub repository](https://github.com/ninerealmlabs/precommit-template)
- [Copier documentation](https://copier.readthedocs.io/)
- [prek](https://github.com/j178/prek) · [pre-commit](https://pre-commit.com/)
