# Troubleshooting — precommit-template

## Contents

- Copier errors
- Update conflicts
- Hook failures
- Commit rejections

## Copier errors

### `'detect_hook_runner' is undefined`

**Cause**: `--trust` was omitted, so Copier skipped `extensions/detect.py`
and the Jinja globals it registers were never defined.

**Fix**: re-run with `--trust`.
It is required on `copy`, `update`, and `recopy`.

### `Destination repository is dirty`

**Cause**: Copier refuses to run against a working tree with uncommitted changes, so it can compute a clean diff.

**Fix**: commit or stash first.

### Survey aborts partway through

**Cause**: a detection helper raised.
`detect_hook_runner()` and `detect_web_format_tool()` are called while rendering question defaults,
and an exception there kills the whole survey.

**Fix**: this is a template bug, not a usage error — report it.
As a workaround, pass the value on the command line: `copier copy --trust --data web_format_tool=biome ...`.

### Nothing was generated for a tool you wanted

**Cause**: the corresponding question was answered `no`, so the conditional filename rendered empty.

**Fix**: `copier update --trust` and answer `yes`.

## Update conflicts

### `*.rej` files after `copier update`

**Cause**: Copier could not merge a template change into a file you had edited locally.

**Fix**: open each `.rej`, apply the rejected hunk by hand, then delete the `.rej` file.
The `forbid-rej` hook blocks commits until they are gone, deliberately:
deleting a `.rej` unread silently drops a template change.

### Local edits vanished after switching an answer

**Cause**: changing an answer deletes files gated on the old value.
Copier renders a rename as delete-then-create, which discards local edits with no conflict marker.

**Fix**: recover from git history.
To avoid it, copy customizations out before re-answering.

### Update pulled in far more than expected

**Cause**: `copier update` renders against the _latest_ template release, not the next one,
so several revisions can land at once.

**Fix**: pin with `--vcs-ref <tag>` to step forward one release at a time.

## Hook failures

### `hadolint: command not found` / `shellcheck: ...` / `prettier: ...`

**Cause**: these three hooks use `language: system` — the runner does not install them.

**Fix**: install the tool (`brew install hadolint`, `brew install shellcheck`, `npm install -g prettier`),
or answer `no` to the corresponding question.

### `typos` cannot find its config

**Cause**: the `typos` hook may need `.typos.toml` to already be committed before it will read it.

**Fix**: commit `.typos.toml` first, then re-run.
If that is not possible, comment out the `typos` block until the config is committed.

### `shfmt` fails to build

**Cause**: shfmt-py's sdist only builds on CPython ≤ 3.13, which is why the hook pins `language_version: python3.13`.

**Fix**: make a 3.13 interpreter available to the runner.

### The whole repo was rewritten on the first run

**Cause**: expected.
`mdformat`, `rumdl-fmt`, `ruff-format`, `shfmt`, and biome or prettier all reformat on first contact.

**Fix**: commit the formatting churn on its own so the config change stays reviewable.

### A formatter and a linter disagree

**Cause**: `mdformat` and `rumdl` both touch markdown and can fight over the same construct.

**Fix**: disable the offending rule in `.rumdl.toml`.
`MD060` (table spacing) is already disabled in this template for exactly that reason.

### Python code blocks in markdown get reformatted

**Cause**: expected.
`ruff-format` owns python code blocks inside markdown,
so a single pinned ruff formats both `.py` files and the snippets in your docs.

**Fix**: none needed.
To exempt one block, wrap it in `<!-- fmt:off -->` / `<!-- fmt:on -->`.
To exempt markdown entirely, add `"*.md"` to `extend-exclude` in `.ruff.toml`
and drop `markdown` from the `ruff-format` hook's `types_or`.

## Commit rejections

### `YAML file extensions must be .yaml`

**Cause**: the `forbid-yml` hook rejects any `.yml` file.

**Fix**: rename it to `.yaml`.
If a tool hard-codes `.yml` and cannot be changed, add it to the hook's `exclude` pattern in `.pre-commit-config.yaml`.

### `Commit messages must not include a Co-authored-by trailer`

**Cause**: a repo-local `commit-msg` hook requires an `AI-assistant:` trailer instead.

**Fix**: replace the trailer.

### commitizen rejects the message

**Cause**: `conventional_commits` is enabled, so messages must match the Conventional Commits format —
`type(scope): subject`.

**Fix**: reword, e.g. `fix(hooks): correct shfmt args`.
