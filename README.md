# Pre-commit Template

Provides a set of configuration files to standardize [pre-commit](https://pre-commit.com/) hooks across repos.

[copier](https://copier.readthedocs.io/) is used to render a pre-commit config and associated tool configurations based
on answers to a survey during the setup phase.

## Quick Start

### Prerequisites

We will use [uv](https://docs.astral.sh/uv/) to install and run tools in isolated environments.

Some pre-commit hooks (`hadolint`, `shellcheck`, and `prettier` if selected), expect to find the tool available in your path.
You may need to install them.

### Install `copier` and `pre-commit`

```sh
# install copier and its dependencies
uv tool install copier --with copier-templates-extensions --with jinja2-time
# we want to manage pre-commit, so ensure it is available
uv tool install pre-commit
```

### Generate your custom configuration with `copier` [docs](https://copier.readthedocs.io/en/stable/generating/)

1. Run `copier` in your local repo

   ```sh
   copier copy --trust "gh:ninerealmlabs/precommit-template" "$(git rev-parse --show-toplevel)"
   ```

2. Answer the questionnaire

   `Copier` will render your configuration based on your selection.
   Then it will commit these new changes automatically (but it will not push the commit).
   This allows you to have a clean git status before running `pre-commit run --all-files` to ensure your repo is in compliance with your new configuration.

3. Run `pre-commit run --all-files` and fix any errors that pre-commit's checks have found

4. Commit

## Features

(opinionated) configuration of formatting and linting tools, including:

- [EditorConfig](https://editorconfig.org/) - Maintains consistent coding styles across various editors and IDEs
- [Biome](https://biomejs.dev/) - A fast formatter and linter for JS, TS, JSON, CSS, and HTML
- [hadolint](https://github.com/hadolint/hadolint) - A smarter Dockerfile linter that ensures best practice Docker
  images
- [mdformat](https://github.com/hukkin/mdformat) - A markdown formatter
- [Prettier](https://github.com/prettier/prettier) - Opinionated code formatter (JS, TS, JSON, CSS, HTML, Markdown,
  YAML)
- [ruff](https://github.com/astral-sh/ruff) - An extremely fast Python linter and code formatter
- [rumdl](https://github.com/rvben/rumdl-pre-commit?tab=readme-ov-file) - A markdown linter and formatter
- [shellcheck](https://github.com/koalaman/shellcheck) - A static analysis tool for shell scripts (sh, bash)
- [typos](https://github.com/crate-ci/typos) - A source code spell checker
- [yamllint](https://github.com/adrienverge/yamllint) - A linter for YAML files

### Dependencies and Gotchas

Some pre-commit hooks rely on tools that must be installed separately (they are not managed by pre-commit):

| Tool                                                 | Required when               | Install                                                                                    |
| ---------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------ |
| [hadolint](https://github.com/hadolint/hadolint)     | `docker: true`              | `brew install hadolint` or [binary release](https://github.com/hadolint/hadolint/releases) |
| [shellcheck](https://github.com/koalaman/shellcheck) | `shell: true`               | `brew install shellcheck` or `apt install shellcheck`                                      |
| [prettier](https://prettier.io/)                     | `web_format_tool: prettier` | `npm install -g prettier`                                                                  |

If these tools are not available in your `$PATH`, the corresponding pre-commit hooks will fail.

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
> enforced by `pre-commit`)

### What does `copier update` do?

`copier` documentation provides a [good overview of how the update process works](https://copier.readthedocs.io/en/latest/updating/#how-the-update-works) -- but TLDR:

- It renders a fresh project from the _latest_ template version
- Then it compares current vs new to get the diffs
- Next it updates the current project with the latest template changes (asking confirmation)
- Finally, it re-applies the previously obtained diff, and then run the post-migrations

## Local development

1. Install development dependencides

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Test updates

   You can run `precommit-template` to update itself using:

   ```sh
   # use current branch's committed files ("HEAD") to run precommit-template on itself
   copier recopy --trust --vcs-ref "HEAD" /path/to/precommit-template /path/to/precommit-template  --answers-file .copier-answers.yaml
   ```
