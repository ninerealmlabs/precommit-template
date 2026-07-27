# Generated files — precommit-template

What the template writes into a target repository, and which hook reads each file.

## Contents

- Always generated
- Conditionally generated
- Naming convention in `template/`

## Always generated

| File                      | Purpose                                                                |
| ------------------------- | ---------------------------------------------------------------------- |
| `.pre-commit-config.yaml` | The hook declarations, read by both `prek` and `pre-commit`            |
| `.copier-answers.yaml`    | Copier's record of the template version and every answer — do not edit |
| `.gitattributes`          | Line-ending and diff settings                                          |

### Baseline hooks

These appear no matter how the survey is answered:

- `forbid-yml` — fails on any `.yml` extension, excluding `.copier-answers.yml`
- `forbid-rej` — fails while `*.rej` files from a conflicted update remain
- `pre-commit/pre-commit-hooks` — large files, merge conflicts, private keys, case conflicts, Python
  AST, JSON/TOML/YAML syntax, end-of-file, mixed line endings, trailing whitespace
- `remove-crlf`
- `fix-smartquotes`, `fix-ligatures`
- `gitleaks`
- `strip-exif`

The config also sets `default_install_hook_types` to `pre-commit`, `post-checkout`, `post-merge`, and `post-rewrite`,
plus `pre-push` and `commit-msg` when `conventional_commits` is enabled.

`exclude` skips `*copier-answers.ya?ml` and `*.rej` globally.

## Conditionally generated

| Answer                    | File                                                                   | Read by                         |
| ------------------------- | ---------------------------------------------------------------------- | ------------------------------- |
| `ai`                      | `AGENTS.md`                                                            | AI coding agents                |
| `editorconfig`            | `.editorconfig`                                                        | `editorconfig-checker`, editors |
| `markdown`                | `.mdformat.toml`                                                       | `mdformat`                      |
| `markdown`                | `.rumdl.toml`                                                          | `rumdl-fmt`                     |
| `python`                  | `.ruff.toml`                                                           | `ruff-check`, `ruff-format`[^1] |
| `python`                  | `tests/test_pypi_security_audit.py`, `tests/test_uv_security_audit.py` | your test runner                |
| `docker`                  | `.hadolint.yaml`                                                       | `hadolint`                      |
| `shell`                   | `.shellcheckrc`                                                        | `shellcheck`                    |
| `web_format` + `biome`    | `.biome.jsonc`                                                         | `biome-check`                   |
| `web_format` + `prettier` | `.prettierrc.yaml`                                                     | `prettier`                      |
| `web_format` + `prettier` | `.prettierignore`                                                      | `prettier`                      |
| `yaml`                    | `.yamllint.yaml`                                                       | `yamllint`                      |
| `typos`                   | `.typos.toml`                                                          | `typos`                         |

### System-installed hooks

Three hooks use `language: system`, meaning the runner will not install the tool for you:

| Hook         | Requires               | Install                   |
| ------------ | ---------------------- | ------------------------- |
| `hadolint`   | `hadolint` on `PATH`   | `brew install hadolint`   |
| `shellcheck` | `shellcheck` on `PATH` | `brew install shellcheck` |
| `prettier`   | `prettier` on `PATH`   | `npm install -g prettier` |

`shellcheck` skips `*.zsh` files, since ShellCheck has no zsh dialect.

## Naming convention in `template/`

Optional files are gated by a Jinja conditional embedded in the filename.
When the condition is false the filename renders empty and Copier writes nothing.

```text
template/{% if python %}.ruff.toml{% endif %}.jinja
template/{% if web_format and web_format_tool == "prettier" %}.prettierrc.yaml{% endif %}.jinja
template/{% if python %}tests{% endif %}/test_uv_security_audit.py
```

The `.jinja` suffix is stripped on render and is configured by `_templates_suffix` in `copier.yaml`.
Files without the suffix — such as the generated test files — are copied verbatim.

[^1]: When `markdown` is also enabled, `ruff-format` gains `types_or: [python, pyi, jupyter, markdown]` and
    formats python code blocks inside markdown. `ruff-check` deliberately stays on python files only, so
    illustrative snippets in docs are not held to import and unused-name rules.
