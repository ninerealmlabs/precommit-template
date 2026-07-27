# Pre-commit Template

Provides a set of configuration files to standardize [pre-commit](https://pre-commit.com/) hooks across repos.

Hooks are declared in a `.pre-commit-config.yaml`, which is read by both [prek](https://github.com/j178/prek)
(the default runner) and [pre-commit](https://pre-commit.com/) itself.

[copier](https://copier.readthedocs.io/) is used to render a pre-commit config
and associated tool configurations based on answers to a survey during the setup phase.

## Quick Start

### Prerequisites

We will use [uv](https://docs.astral.sh/uv/) to install and run tools in isolated environments.

Some hooks (`hadolint`, `shellcheck`, and `prettier` if selected), expect to find the tool available in your path.
You may need to install them.

### Install `copier` and a hook runner

```sh
# install copier and its dependencies
uv tool install copier --with copier-templates-extensions --with jinja2-time
# we want to manage git hooks, so ensure a runner is available
uv tool install prek
# ...or, if you prefer pre-commit
# uv tool install pre-commit
```

`prek` is a drop-in reimplementation of `pre-commit` in Rust:
it reads the same `.pre-commit-config.yaml` and supports the same `run` / `install` commands,
but is considerably faster and manages its own tool environments.
Substitute `pre-commit` for `prek` in the commands below if that is the runner you use.

### Generate your custom configuration with `copier` [docs](https://copier.readthedocs.io/en/stable/generating/)

1. Run `copier` in your local repo

   ```sh
   copier copy --trust "gh:ninerealmlabs/precommit-template" "$(git rev-parse --show-toplevel)"
   ```

2. Answer the questionnaire

   `Copier` will render your configuration based on your selection.
   Then it will commit these new changes automatically (but it will not push the commit).
   This allows you to have a clean git status
   before running `prek run --all-files` to ensure your repo is in compliance with your new configuration.

3. Run `prek run --all-files` and fix any errors that the checks have found

4. Run `prek install` so the checks run on every commit

5. Commit

### Detection

Where the template can work out what your repo already uses, it does, rather than making you say so:

| Value             | Detected from                                                                                                  | Fallback |
| ----------------- | -------------------------------------------------------------------------------------------------------------- | -------- |
| hook runner       | installed git hook shims, a `prek.toml`, how CI pipelines invoke the runner, or which runner is on your `PATH` | `prek`   |
| `web_format_tool` | an existing `.pre-commit-config.yaml`, biome or prettier config files, or `package.json` dependencies          | `biome`  |

The hook runner is **not** a survey question.
Both runners read the same `.pre-commit-config.yaml`,
so the detected value only decides which one the generated comments and post-copy instructions point at.
It is deliberately not recorded in `.copier-answers.yaml`,
so it is recomputed on every run and follows your repo if you switch runners.

`web_format_tool` **is** a survey question, since the two tools produce genuinely different configs.
Detection only pre-selects the prompt's default;
your answer is recorded in `.copier-answers.yaml` and reused on every `copier update`.

## Features

(opinionated) configuration of formatting and linting tools, including:

- [prek](https://github.com/j178/prek) - A fast Rust reimplementation of pre-commit, and the default hook runner
- [pre-commit](https://pre-commit.com/) - The original git hook framework, selectable as an alternative
- [EditorConfig](https://editorconfig.org/) - Maintains consistent coding styles across various editors and IDEs
- [Biome](https://biomejs.dev/) - A fast formatter and linter for JS, TS, JSON, CSS, and HTML; the default web
  formatter
- [hadolint](https://github.com/hadolint/hadolint) - A smarter Dockerfile linter that ensures best practice Docker
  images
- [mdformat](https://github.com/hukkin/mdformat) - A markdown formatter
- [Prettier](https://github.com/prettier/prettier) - Opinionated code formatter (JS, TS, JSON, CSS, HTML, Markdown,
  YAML), selectable as an alternative to Biome
- [ruff](https://github.com/astral-sh/ruff) - An extremely fast Python linter and code formatter; also formats
  python code blocks inside markdown when both `python` and `markdown` are enabled
- [rumdl](https://github.com/rvben/rumdl-pre-commit?tab=readme-ov-file) - A markdown linter and formatter
- [shellcheck](https://github.com/koalaman/shellcheck) - A static analysis tool for shell scripts (sh, bash)
- [typos](https://github.com/crate-ci/typos) - A source code spell checker
- [yamllint](https://github.com/adrienverge/yamllint) - A linter for YAML files

### Dependencies and Gotchas

Some hooks rely on tools that must be installed separately (they are not managed by the hook runner):

| Tool                                                 | Required when               | Install                                                                                    |
| ---------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------ |
| [hadolint](https://github.com/hadolint/hadolint)     | `docker: true`              | `brew install hadolint` or [binary release](https://github.com/hadolint/hadolint/releases) |
| [shellcheck](https://github.com/koalaman/shellcheck) | `shell: true`               | `brew install shellcheck` or `apt install shellcheck`                                      |
| [prettier](https://prettier.io/)                     | `web_format_tool: prettier` | `npm install -g prettier`                                                                  |

If these tools are not available in your `$PATH`, the corresponding hooks will fail.

### Other (unrelated) project setup tools

- [gitignore.io - Create Useful .gitignore Files For Your Project](https://www.toptal.com/developers/gitignore)

## Update your custom configuration with `copier` [docs](https://copier.readthedocs.io/en/stable/updating/)

> **!! DO NOT MANUALLY UPDATE `copier-answers` file!!**

1. Navigate to project directory: `cd <git project dir>`

2. Ensure a `feature` branch is checked out.

3. Commit (or stash) current work.
   Copier will not work with "unclean" file statuses.

4. Run `copier update`.
   This will try to render files based on the _latest_ release of `common`:

   ```sh
   copier update --trust . --answers-file .copier-answers.yaml
   ```

> If `copier` is unable to resolve the diff between current and latest revisions, it will create `*.rej` files that
> contain the unresolved differences. These must be reviewed (and resolved/implemented) prior to commit (this is
> enforced by a hook)

### What does `copier update` do?

`copier` documentation provides a
[good overview of how the update process works](https://copier.readthedocs.io/en/latest/updating/#how-the-update-works) --
but TLDR:

- It renders a fresh project from the _latest_ template version
- Then it compares current vs new to get the diffs
- Next it updates the current project with the latest template changes (asking confirmation)
- Finally, it re-applies the previously obtained diff, and then run the post-migrations

## For AI coding agents

The documentation site publishes machine-readable context so an agent can apply
and maintain this template without being walked through it:

| Artifact                                                                          | Contents                                                  |
| --------------------------------------------------------------------------------- | --------------------------------------------------------- |
| [Agent skill](https://ninerealmlabs.github.io/precommit-template/skills.html)     | Workflows, gotchas, and references for using the template |
| [llms.txt](https://ninerealmlabs.github.io/precommit-template/llms.txt)           | Indexed overview: commands, questions, detection behavior |
| [llms-full.txt](https://ninerealmlabs.github.io/precommit-template/llms-full.txt) | Full context in one file, including troubleshooting       |

Install the skill into an agent with:

```sh
great-docs skill install https://ninerealmlabs.github.io/precommit-template/
```

## Local development

1. Install development dependencies

   ```sh
   uv sync
   ```

2. Build the documentation site

   The site is generated by [Great Docs](https://posit-dev.github.io/great-docs/),
   which renders through [Quarto](https://quarto.org/).
   Both come from `uv sync`, so no separate Quarto install is needed —
   the `quarto-cli` package brings the pandoc build Quarto expects, which a system Quarto does not always match.

   ```sh
   uv run great-docs build     # output in great-docs/_site/
   uv run great-docs preview   # local server on port 3000
   ```

   `great-docs/` is regenerated on every build and is gitignored; edit the sources instead.
   The homepage comes from this `README.md`, the license page from `LICENSE`,
   and the changelog from published GitHub Releases.
   `llms.txt` and `llms-full.txt` are hand-written under `site_root/`,
   which Great Docs copies to the root of the built site.

3. Test updates

   You can run `precommit-template` to update itself using:

   ```sh
   # use current branch's committed files ("HEAD") to run precommit-template on itself
   copier recopy --trust --vcs-ref "HEAD" /path/to/precommit-template  --answers-file .copier-answers.yaml
   ```

   `--trust` is required: `copier.yaml` loads `extensions/detect.py`, which inspects the
   target repo to pre-select the `hook_runner` and `web_format_tool` defaults.
