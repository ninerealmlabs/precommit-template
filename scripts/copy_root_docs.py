"""Mirror root-level files into the docs tree at build time.

The site needs `README.md` and `LICENSE` as pages, but the content must live at the repo root. Copying them
during the build keeps a single source of truth without tracking shim pages, so no generator-specific include
markup ends up in the repository where a markdown formatter would rewrite it.

Wired up via the `hooks:` key in `mkdocs.yaml`. The generated pages are gitignored; every hand-written page
under `docs/` stays lint- and format-checked.
"""

from pathlib import Path
import shutil

# (path relative to the repo root, page name to generate under docs/)
MIRRORED_FILES = (
    ("README.md", "index.md"),
    ("LICENSE", "license.md"),
)


def on_pre_build(config, **kwargs) -> None:
    """Copy each mirrored root file into the docs directory before MkDocs collects pages."""
    root = Path(config.config_file_path).parent
    docs_dir = Path(config.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    for source_name, page_name in MIRRORED_FILES:
        shutil.copyfile(root / source_name, docs_dir / page_name)
