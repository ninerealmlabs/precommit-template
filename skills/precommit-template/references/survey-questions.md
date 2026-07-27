# Survey questions — precommit-template

Every question is defined in `copier.yaml`.
Answers are recorded in `.copier-answers.yaml` and replayed on `copier update`.

## Contents

- Asked questions
- Detected values
- Changing an answer later

## Asked questions

All ten questions default to `true`
(or to a detected value), so accepting every default produces the full configuration.

| Question               | Type | Default  | Prompt                                                 |
| ---------------------- | ---- | -------- | ------------------------------------------------------ |
| `ai`                   | bool | `true`   | Prepare AGENTS.md?                                     |
| `conventional_commits` | bool | `true`   | Use conventional commits?                              |
| `editorconfig`         | bool | `true`   | Use editorconfig?                                      |
| `markdown`             | bool | `true`   | Lint and format markdown?                              |
| `python`               | bool | `true`   | Lint and format python?                                |
| `docker`               | bool | `true`   | Lint and check docker files?                           |
| `shell`                | bool | `true`   | Lint and format shell scripts?                         |
| `web_format`           | bool | `true`   | Lint and format JS/TS/JSON/HTML/CSS and related files? |
| `web_format_tool`      | str  | detected | Select the web formatter (`biome` or `prettier`)       |
| `yaml`                 | bool | `true`   | Lint and format YAML?                                  |
| `typos`                | bool | `true`   | Check for typos?                                       |

`web_format_tool` is only asked when `web_format` is `true`.

### What each answer controls

| Answer                    | Config files written                           | Hooks added                                               |
| ------------------------- | ---------------------------------------------- | --------------------------------------------------------- |
| `ai`                      | `AGENTS.md`                                    | —                                                         |
| `conventional_commits`    | —                                              | `commitizen`; adds `pre-push` and `commit-msg` hook types |
| `editorconfig`            | `.editorconfig`                                | `editorconfig-checker`                                    |
| `markdown`                | `.mdformat.toml`, `.rumdl.toml`                | `mdformat`, `rumdl-fmt`                                   |
| `python`                  | `.ruff.toml`, `tests/test_*_security_audit.py` | `ruff-check`, `ruff-format`, `nbstripout`                 |
| `docker`                  | `.hadolint.yaml`                               | `hadolint` (system)                                       |
| `shell`                   | `.shellcheckrc`                                | `shellcheck` (system), `shfmt`                            |
| `web_format` + `biome`    | `.biome.jsonc`                                 | `biome-check`                                             |
| `web_format` + `prettier` | `.prettierrc.yaml`, `.prettierignore`          | `prettier` (system)                                       |
| `yaml`                    | `.yamllint.yaml`                               | `yamllint`                                                |
| `typos`                   | `.typos.toml`                                  | `typos`                                                   |

Hooks that are always present regardless of answers: `forbid-yml`, `forbid-rej`, the `pre-commit/pre-commit-hooks` set
(large files, merge conflicts, private keys, case conflicts, AST, JSON/TOML/YAML syntax, EOF, line endings,
trailing whitespace), `remove-crlf`, `fix-smartquotes`, `fix-ligatures`, `gitleaks`, and `strip-exif`.

## Detected values

Two values come from inspecting the target repo rather than from a prompt.
Detection lives in `extensions/detect.py`, loaded as a Jinja extension — which is why `--trust` is required.

### `hook_runner`

Declared with `when: false`, so it is a computed value, never a question, and never written to `.copier-answers.yaml`.
It is recomputed on every run and follows the repo if the runner changes.

Detection order:

1. Installed git hook shims in `.git/hooks/`
2. A `prek.toml` in the repo
3. How CI pipelines invoke the runner
4. Which runner is on `PATH`
5. Fallback: `prek`

The value only decides which runner the generated comments and the post-copy message name.
Both runners read the same `.pre-commit-config.yaml`, so a wrong guess is cosmetic.

### `web_format_tool`

A real question whose _default_ is detected, because biome and prettier need different config files.
The answer is recorded and reused on every update.

Detection order:

1. An existing `.pre-commit-config.yaml` that already runs one of them
2. Biome or prettier config files in the repo
3. `package.json` dependencies
4. Fallback: `biome`

## Changing an answer later

Run `copier update --trust --answers-file .copier-answers.yaml` and give a different answer at the prompt.

Copier deletes files gated on the old answer and creates files gated on the new one.
Switching `web_format_tool` from `biome` to `prettier` removes `.biome.jsonc` and adds `.prettierrc.yaml`
and `.prettierignore`; any local edits to the removed file are lost, so copy them somewhere safe first.

Answering `no` to a question that was previously `yes` removes that tool's config file and hook block.
