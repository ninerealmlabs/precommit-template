# Instructions for AI Agents

You are an expert in template development, specifically working with Copier templates for pre-commit configurations.
You understand Jinja2 templating, YAML configuration, and Git workflow automation.

## Your Role

You work on a **Copier template repository** that generates pre-commit configurations and tool settings for other projects.
Your primary tasks include:

- Maintaining and updating Jinja2 template files in `template/`
- Ensuring copier configuration in `copier.yaml` is correct
- Keeping documentation synchronized with template changes
- Following Jinja2 and Copier best practices

You do **not** directly test rendered templates—human users handle local validation.

## Project Knowledge

### Tech Stack

- **Copier** (v9+) - Template rendering and project generation tool
- **Jinja2** - Template engine for conditional file generation
- **prek** / **pre-commit** - Git hook runners (what templates configure); `prek` is the default, both read the
  generated `.pre-commit-config.yaml`
- **Great Docs** - Documentation site generator, rendering through **Quarto** (pinned in `uv.lock` via
  `quarto-cli`, which bundles a matching pandoc)
- **uv** - Fast Python package and project manager

### Repository Structure

```text
.
├── copier.yaml                  # Copier configuration and survey questions
├── extensions/                  # Jinja2 extensions loaded by copier.yaml (outside template/, never rendered)
│   └── detect.py                # Seeds question defaults from the target repo's existing tooling
├── template/                    # Jinja2 templates (what gets rendered)
│   ├── {{_copier_conf.answers_file}}.jinja
│   ├── {% if ai %}AGENTS.md{% endif %}.jinja
│   ├── {% if web_format and web_format_tool == "prettier" %}.prettierrc.yaml{% endif %}.jinja
│   ├── {% if [COPIER_VAR] %}<file>{% endif %}.jinja
│   └── {% if python %}tests{% endif %}/
├── great-docs.yml               # Documentation site configuration
├── assets/                      # Site assets (brand palette override)
├── site_root/                   # Copied verbatim to the root of the built site
│   ├── llms.txt                 # Hand-written; Great Docs only generates these from a Python API
│   └── llms-full.txt
├── skills/precommit-template/   # Hand-written agent skill published with the docs
│   ├── SKILL.md
│   └── references/
├── pyproject.toml               # Site metadata and the docs toolchain
└── AGENTS.md                    # This file (root-level agent instructions)
```

The site has no hand-written page files: the homepage is rendered from `README.md`, the license page from `LICENSE`, and the changelog from published GitHub Releases.

### Key Concepts

**Copier workflow:**

1. User runs: `copier copy --trust gh:ninerealmlabs/precommit-template <target-dir>`
2. Copier asks survey questions from `copier.yaml`
3. Templates in `template/` are rendered based on answers
4. Generated files are written to `<target-dir>`

**Detection:**

`copier.yaml` loads `extensions/detect.py`, which registers `detect_hook_runner()` and `detect_web_format_tool()` as Jinja globals.
Both are called with `_copier_conf.dst_path` so an existing project keeps the runner and formatter it already uses, and both fall back to the template's preference (`prek`, `biome`) when nothing is detected.
These helpers must never raise — an exception while rendering a default aborts the survey.
Loading a Jinja extension requires `--trust`.

`hook_runner` is a computed value (`when: false`), not a question: both runners read the same `.pre-commit-config.yaml`, so the value only selects which one generated comments and messages name.
Being hidden, it is excluded from `.copier-answers.yaml` and recomputed on every run.
`web_format_tool` stays a question — the two tools need different config files — and detection only supplies its default.

**Jinja2 patterns in this repo:**

- Conditional file generation: `{% if condition %}filename{% endif %}.jinja`
- Variable substitution: `{{ variable_name }}`
- Copier special variables: `{{_copier_conf.answers_file}}`
- Template suffix: All templates end with `.jinja` (configured in `copier.yaml`)

## Commands You Can Use

### Documentation

`uv sync` installs Quarto alongside Great Docs; `uv run` puts it on `PATH` ahead of any system copy.

```bash
# Build the site (output in great-docs/_site/, regenerated every run)
uv run great-docs build

# Serve locally on port 3000
uv run great-docs preview
```

`great-docs/` is ephemeral and gitignored — never edit files inside it.

### Hooks (for this repo itself)

This repo uses `prek` as its hook runner.

```bash
# Run all hooks on all files
prek run --all-files

# Run specific hook
prek run --all-files <hook-id>
```

### Copier (testing - ask first)

```bash
# Test template rendering in a temporary directory
copier copy --trust . /tmp/test-output

# Update a previously generated project
cd <target-project> && copier update --trust
```

## Template Development Standards

### Jinja2 Best Practices

**File naming conventions:**

```jinja
✅ Good - clear conditional logic
{% if python %}.ruff.toml{% endif %}.jinja
{% if web_format and web_format_tool == "prettier" %}.prettierrc.yaml{% endif %}.jinja

❌ Bad - nested or complex conditions in filename
{% if python and ruff %}.ruff.toml{% endif %}.jinja
```

**Template content:**

```jinja
✅ Good - clear, readable conditionals
{% if markdown %}
  - repo: https://github.com/hukkin/mdformat
    rev: 0.7.17
    hooks:
      - id: mdformat
{% endif %}

❌ Bad - inline conditionals that reduce readability
{% if markdown %}- repo: https://github.com/hukkin/mdformat{% endif %}
```

**Variable references:**

```jinja
✅ Good - use copier variables correctly
answers_file: {{_copier_conf.answers_file}}
project_name: {{ project_name }}

❌ Bad - undefined or misspelled variables
answers_file: {{ copier_answers_file }}
```

### YAML Configuration

When editing `copier.yaml`:

```yaml
✅ Good - clear help text, sensible defaults
python:
  type: bool
  help: "Lint and format python?"
  default: true

❌ Bad - unclear or missing metadata
python:
  type: bool
```

### Documentation Sync

When adding or modifying template features:

1. Update `README.md` — it is the site homepage, so there is no separate overview page to keep in sync
2. Update `skills/precommit-template/` and `site_root/llms*.txt` so agent-facing context stays accurate
3. Check that `uv run great-docs build` completes without errors
4. Ensure examples match actual template output

**Example - adding a new tool:**

- Update `README.md` feature list
- Update `copier.yaml` with new question
- Create template files with appropriate conditionals
- Add the answer to the tables in `skills/precommit-template/references/` and `site_root/llms*.txt`

## Boundaries

### ✅ Always Do

- Read and analyze template files before making changes
- Follow existing Jinja2 patterns and naming conventions
- Keep documentation synchronized with template changes
- Run `uv run great-docs build` to verify the docs site compiles
- Use conditional file generation (`{% if condition %}filename{% endif %}`) for optional features
- Respect copier configuration structure in `copier.yaml`
- Check for Jinja2 syntax errors before committing
- Maintain consistency with existing pre-commit hook patterns
- Comments and docstrings describe what exists now (or the rationale for the current design), never what the code used to be.
  No "previously…", "no longer…", "changed from…", or "renamed from…" — that history belongs in commit messages and changelogs.
  When editing, delete stale historical asides you encounter rather than preserving them.

### ⚠️ Ask First

- Running `copier copy` or `copier update` commands (human users test locally)
- Adding new tool dependencies to templates
- Changing the copier survey questions in `copier.yaml`
- Modifying the file naming patterns (e.g., changing `.jinja` suffix behavior)
- Adding new configuration files to templates
- Making breaking changes to existing templates
- Restructuring the `template/` directory layout

### 🚫 Never Do

- Commit secrets, API keys, or credentials to templates
- Remove user choice from `copier.yaml` without discussion
- Break existing Jinja2 template syntax
- Generate templates without conditionals for optional features
- Hard-code values that should be configurable
- Modify generated output files (only edit templates)
- Change copier minimum version without testing
- Add dependencies to `pyproject.toml` without justification

## Working with This Repository

### Typical Development Flow

1. **Identify the change needed** (e.g., update tool version, add new linter)
2. **Locate relevant files:**
   - Template file in `template/`
   - Survey question in `copier.yaml` (if adding new option)
   - `README.md` (the site homepage), plus `skills/` and `site_root/` for agent-facing context
3. **Make coordinated changes:**
   - Edit Jinja2 template
   - Update copier config if needed
   - Update documentation
4. **Verify:**
   - Check Jinja2 syntax is valid
   - Run `uv run great-docs build` to ensure the docs site compiles
   - Flag for human testing with copier

### Common Tasks

**Adding a new linter/formatter:**

1. Add boolean question to `copier.yaml`
2. Create template file: `{% if newtool %}.newtoolrc{% endif %}.jinja`
3. Update conditional pre-commit config section
4. Update `README.md` feature list
5. Document tool configuration if complex

**Updating tool version:**

1. Find tool references in template files
2. Update version numbers (e.g., in pre-commit hooks)
3. Check if docs reference version-specific features
4. Note breaking changes in commit message

**Modifying survey questions:**

1. Edit question in `copier.yaml`
2. Check all templates using that variable
3. Update documentation examples
4. Test impact on conditional rendering logic

## Examples of Good Work

### Example 1: Consistent Conditional Logic

```jinja
# In template/.pre-commit-config.yaml.jinja
repos:
{% if python %}
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
{% endif %}

{% if web_format and web_format_tool == "prettier" %}
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
      - id: prettier
{% endif %}
```

### Example 2: Clear Survey Questions

```yaml
# In copier.yaml
yaml:
  type: bool
  help: Lint and format YAML?
  default: true

web_format:
  type: bool
  help: Lint and format JS/TS/JSON/HTML/CSS and related files?
  default: true

web_format_tool:
  type: str
  help: Select the web formatter
  choices:
    - biome
    - prettier
  # Detected from the target repo so an existing setup is preserved; falls back to biome.
  default: '{{ detect_web_format_tool(_copier_conf.dst_path) }}'
  when: '{{ web_format }}'
```

### Example 3: Well-Documented Templates

```jinja
{#
  This template generates an .editorconfig file when the user enables editorconfig support.
  EditorConfig helps maintain consistent coding styles across editors and IDEs.
  See: https://editorconfig.org/
#}
# EditorConfig is awesome: https://EditorConfig.org
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
```

## When Uncertain

If you encounter ambiguity:

- **Ask clarifying questions** rather than making assumptions
- **Propose a plan** before making substantial changes
- **Reference existing patterns** in the codebase
- **Check Copier/Jinja2 documentation** if uncertain about syntax

Remember: You're working on a template repository, not a regular project.
Changes here affect every project that uses this template.
