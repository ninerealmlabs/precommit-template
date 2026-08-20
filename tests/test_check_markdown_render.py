#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "beautifulsoup4>=4.13",
#   "pytest>=8.0",
#   "pyyaml>=6.0",
# ]
# ///
"""Tests for `scripts/check_markdown_render.py`.

The table below is the specification: each case is a pair of markdown documents
and a claim about whether a reader would see a difference.  `DAMAGE` cases must
be reported; `NEUTRAL` cases are the source churn formatters legitimately
produce and must not be.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

# The template holds the canonical copy; anything under `scripts/` in this repo is
# rendered output.  Globbed rather than spelled out, because the directory it lives
# in is named for the copier conditional that decides whether it ships at all.
SCRIPT = next(Path(__file__).resolve().parents[1].glob("template/*/check_markdown_render.py"))
_spec = importlib.util.spec_from_file_location("check_markdown_render", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
check_markdown_render = importlib.util.module_from_spec(_spec)
# `@dataclass` resolves annotations through sys.modules, so register before exec.
sys.modules["check_markdown_render"] = check_markdown_render
_spec.loader.exec_module(check_markdown_render)

pytestmark = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc is required")


def md(text: str) -> str:
    """Dedent an inline markdown fixture and give it a trailing newline."""
    return textwrap.dedent(text).lstrip("\n")


# (id, before, after) -- a reader sees these differently, so they must be reported.
DAMAGE = [
    (
        "paragraph-split",
        "One sentence. Two sentence.\n",
        "One sentence.\n\nTwo sentence.\n",
    ),
    (
        "paragraph-merged",
        "One sentence.\n\nTwo sentence.\n",
        "One sentence. Two sentence.\n",
    ),
    (
        "table-cell-collapsed",
        md("""
            | a | b |
            |---|---|
            | 1 | 2 |
            """),
        md("""
            | a | b |
            |---|---|
            | 1 2 |
            """),
    ),
    (
        "code-block-reindented",
        md("""
            ```python
            def f():
                return 1
            ```
            """),
        md("""
            ```python
            def f():
                    return 1
            ```
            """),
    ),
    (
        "code-block-blank-line-lost",
        "```\na\n\nb\n```\n",
        "```\na\nb\n```\n",
    ),
    (
        "frontmatter-value-changed",
        "---\nname: s\ndescription: Use when doing X.\n---\n\n# T\n",
        "---\nname: s\ndescription: Use when doing Z.\n---\n\n# T\n",
    ),
    (
        "frontmatter-key-dropped",
        "---\nname: s\ndescription: D\n---\n\n# T\n",
        "---\nname: s\n---\n\n# T\n",
    ),
    (
        "link-stopped-parsing",
        "See [the guide](https://example.com/g) here.\n",
        "See \\[the guide\\](https://example.com/g) here.\n",
    ),
    (
        "link-url-changed",
        "See [g](https://example.com/a).\n",
        "See [g](https://example.com/b).\n",
    ),
    (
        "duplicate-link-lost",
        "[a](u) and [a](u).\n",
        "[a](u) and a.\n",
    ),
    (
        "list-nesting-flattened",
        "- one\n  - nested\n- two\n",
        "- one\n- nested\n- two\n",
    ),
    (
        "ordered-list-start-changed",
        "3. three\n4. four\n",
        "1. three\n1. four\n",
    ),
    (
        "sentence-deleted",
        "Alpha. Beta. Gamma.\n",
        "Alpha. Gamma.\n",
    ),
    (
        "sentence-reordered",
        "Alpha. Beta. Gamma.\n",
        "Alpha. Gamma. Beta.\n",
    ),
    (
        "code-span-lost",
        "Run `make` now.\n",
        "Run make now.\n",
    ),
    (
        "hard-break-lost",
        "line one  \nline two\n",
        "line one\nline two\n",
    ),
    (
        "blockquote-lost",
        "> quoted\n",
        "quoted\n",
    ),
    (
        "heading-level-changed",
        "## Title\n\nbody\n",
        "### Title\n\nbody\n",
    ),
    (
        "stray-space-before-punctuation",
        "It was _absurd._).\n",
        "It was _absurd._ ).\n",
    ),
    (
        "table-column-added",
        "| a |\n|---|\n| 1 |\n",
        "| a | b |\n|---|---|\n| 1 | 2 |\n",
    ),
    (
        "image-alt-lost",
        "![a picture](p.png)\n",
        "![](p.png)\n",
    ),
    # The cases below were produced by fuzzing the repo's real mdformat + rumdl
    # chain.  Every one is a rendered difference the chain actually causes.
    (
        # rumdl deletes the text before an emphasis span containing a quoted clause.
        "emphasis-quote-text-deleted",
        'A *"q?" Then b* c. Then d.\n',
        "_Then b_ c.\nThen d.\n",
    ),
    (
        # An emphasis run closed and reopened around a sentence split.  The visible
        # text is byte-identical, so only element structure reveals it.
        "emphasis-run-torn-in-two",
        "This is _first. Second_ here.\n",
        "This is _first._\n_Second_ here.\n",
    ),
    (
        "strikethrough-run-torn-in-two",
        "This is ~~first. Second~~ here.\n",
        "This is ~~first.~~\n~~Second~~ here.\n",
    ),
    (
        # Nested emphasis torn apart, leaking its delimiters as visible characters.
        "emphasis-delimiter-leaked-as-text",
        "A **_b. C_** d.\n",
        "A **_b.**\n**C_** d.\n",
    ),
    (
        "intraword-emphasis-gains-space",
        "The token foo*bar*baz is special.\n",
        "The token foo *bar*baz is special.\n",
    ),
    (
        "heading-trailing-punctuation-stripped",
        "## Configuration:\n\nBody.\n",
        "## Configuration\n\nBody.\n",
    ),
    (
        # A block scalar holding a `---` line truncates the frontmatter; the
        # remainder is re-emitted into the body.
        "frontmatter-scalar-truncated",
        "---\ntitle: T\nbody: |\n  line one\n  ---\n  line three\n---\n\nContent.\n",
        "---\ntitle: T\nbody: |-\n  line one\n---\n\n## line three\n\nContent.\n",
    ),
    (
        # `...` is a YAML document-end marker but not a recognized fence, so the
        # whole block becomes visible body text.
        "frontmatter-docend-demoted-to-body",
        "---\ntitle: t\n...\n\nContent.\n",
        "---\n\ntitle: t ...\n\nContent.\n",
    ),
    (
        "paragraph-became-setext-heading",
        "---\n\nText after a break.\n\n---\n\nMore.\n",
        "---\n\nText after a break.\n---\n\nMore.\n",
    ),
    (
        "sibling-lists-merged",
        "- alpha\n- beta\n\n* gamma\n* delta\n",
        "- alpha\n- beta\n- gamma\n- delta\n",
    ),
    (
        "lazy-table-became-real-table",
        "Some paragraph text.\n| A | B |\n| --- | --- |\n| 1 | 2 |\n",
        "Some paragraph text.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n",
    ),
    (
        # mdformat-ruff reformats the interior of python fences.
        "python-fence-reformatted",
        "```python\nx=1\ndef  f( a,b ):\n    return a+b\n```\n",
        "```python\nx = 1\n\n\ndef f(a, b):\n    return a + b\n```\n",
    ),
    (
        # The code is untouched; only the fence language changes, so no
        # content-based projection would see it.
        "fence-language-changed",
        "```{python}\nx=1\n```\n",
        "```text {python}\nx=1\n```\n",
    ),
    (
        "code-block-trailing-blank-line-lost",
        "```\nx = 1\n\n```\n",
        "```\nx = 1\n```\n",
    ),
    (
        # An angle-bracket destination holding a literal space gets encoded.
        "link-destination-percent-encoded",
        "See [d](<https://ex.com/a b>) now.\n",
        "See [d](https://ex.com/a%20b) now.\n",
    ),
    (
        # CJK has no inter-word spacing, so a sentence split renders as a gap.
        "cjk-sentence-split-adds-space",
        "これはテストです。これもテストです。\n",
        "これはテストです。\nこれもテストです。\n",
    ),
    (
        "h1-demoted-to-h2",
        "---\ntitle: Doc\n---\n\n# Doc\n\nBody.\n",
        "---\ntitle: Doc\n---\n\n## Doc\n\nBody.\n",
    ),
    (
        # A marker with trailing text is not a valid GFM alert; moving the text
        # promotes a plain blockquote into a callout.
        "blockquote-promoted-to-alert",
        "> [!NOTE] inline title text\n> body\n",
        "> [!NOTE]\n> inline title text\n> body\n",
    ),
    # The cases below defeat a projection set built from whitespace-collapsed
    # text, a bare link list, and per-tag element counts: each pair has identical
    # text, identical hrefs, and identical tag counts.
    (
        # Two words run together; a separator-joined text projection invents the
        # very space whose loss is the defect.
        "inline-boundary-space-lost",
        "The value is **not** safe to reuse.\n",
        "The value is **not**safe to reuse.\n",
    ),
    (
        # The same words are present, but a different phrase is hyperlinked.
        "link-anchor-text-moved",
        "See [the deprecated helper](api.md) for details.\n",
        "See the deprecated [helper](api.md) for details.\n",
    ),
    (
        "emphasis-moved-to-different-word",
        "Do **not** delete the backup file.\n",
        "Do not delete the **backup** file.\n",
    ),
    (
        # The dangerous arguments fall out of the code span into prose.
        "code-span-shrunk",
        "Run `rm -rf build` from the repo root.\n",
        "Run `rm` -rf build from the repo root.\n",
    ),
    (
        # rumdl's MD031 turns a tight list loose, padding every item.
        "list-became-loose",
        "- alpha\n- beta\n  ```py\n  x = 1\n  ```\n- gamma\n",
        "- alpha\n- beta\n\n  ```py\n  x = 1\n  ```\n\n- gamma\n",
    ),
    (
        "table-alignment-flipped",
        "| Rule | Cost |\n| :--- | :--- |\n| MD013 | 4 |\n",
        "| Rule | Cost |\n| ---: | ---: |\n| MD013 | 4 |\n",
    ),
    (
        "ordered-list-start-offset",
        "1. Install the tool.\n1. Run the check.\n",
        "7. Install the tool.\n8. Run the check.\n",
    ),
    (
        "task-checkbox-state-flipped",
        "- [ ] Ship the release\n- [x] Run the tests\n",
        "- [x] Ship the release\n- [ ] Run the tests\n",
    ),
    (
        # An image's entire accessible content lives in an attribute.
        "image-alt-text-contradicted",
        "![Latency dropped 40 percent](chart.png)\n",
        "![Latency rose 40 percent](chart.png)\n",
    ),
    (
        "link-title-tooltip-added",
        "Read the [style guide](guide.md).\n",
        'Read the [style guide](guide.md "DEPRECATED - do not follow").\n',
    ),
    (
        # Same tag counts, inverted outline.
        "heading-levels-swapped",
        "# Setup\n\nbody\n\n## Danger\n\nbody\n",
        "## Setup\n\nbody\n\n# Danger\n\nbody\n",
    ),
    (
        "list-nesting-depth-changed",
        "- alpha\n  - beta\n- gamma\n  - delta\n",
        "- alpha\n  - beta\n    - gamma\n- delta\n",
    ),
    (
        # A second independent quote becomes a nested reply inside the first.
        "blockquote-nesting-changed",
        "> The drug is safe.\n\n> The agency disagrees.\n",
        "> The drug is safe.\n>\n> > The agency disagrees.\n",
    ),
    (
        "thematic-break-relocated",
        "intro\n\n---\n\nbody\n\ntail\n",
        "intro\n\nbody\n\n---\n\ntail\n",
    ),
    (
        "hard-break-relocated",
        "alpha\\\nbeta gamma\n",
        "alpha beta\\\ngamma\n",
    ),
    (
        # Raw HTML passes both formatters untouched, so nothing else constrains it.
        "details-forced-open",
        "<details>\n<summary>Terms</summary>\n\nYou agree to arbitration.\n\n</details>\n",
        "<details open>\n<summary>Terms</summary>\n\nYou agree to arbitration.\n\n</details>\n",
    ),
    (
        "content-hidden-by-style",
        "<div>\n\nPayment is required before access.\n\n</div>\n",
        '<div style="display:none">\n\nPayment is required before access.\n\n</div>\n',
    ),
    (
        # The code is untouched, only the highlighting language is downgraded.
        "fence-attribute-downgraded",
        "```{.python}\nx = 1\n```\n",
        "```text {.python}\nx = 1\n```\n",
    ),
]

# (id, before, after) -- source churn that renders identically; must NOT be reported.
NEUTRAL = [
    (
        "prose-rewrapped",
        "Alpha beta gamma delta epsilon zeta eta theta.\n",
        "Alpha beta gamma\ndelta epsilon zeta\neta theta.\n",
    ),
    (
        "sentence-per-line-reflow",
        "First sentence here. Second sentence here.\n",
        "First sentence here.\nSecond sentence here.\n",
    ),
    (
        "bullet-marker-normalized",
        "* one\n* two\n",
        "- one\n- two\n",
    ),
    (
        "emphasis-marker-normalized",
        "This is *emphatic* text.\n",
        "This is _emphatic_ text.\n",
    ),
    (
        "strong-marker-normalized",
        "This is __strong__ text.\n",
        "This is **strong** text.\n",
    ),
    (
        "emphasis-boundary-space-moved",
        "foo <em>bar</em> baz\n",
        "foo<em> bar </em>baz\n",
    ),
    (
        "link-split-across-lines",
        "See [the reference guide](refs/g.md) for details.\n",
        "See [the reference\nguide](refs/g.md) for details.\n",
    ),
    (
        "thematic-break-normalized",
        "a\n\n***\n\nb\n",
        "a\n\n---\n\nb\n",
    ),
    (
        "trailing-whitespace-stripped",
        "some text   \n\nmore text\n",
        "some text\n\nmore text\n",
    ),
    (
        "atx-closing-hashes-removed",
        "## Title ##\n\nbody\n",
        "## Title\n\nbody\n",
    ),
    (
        "indented-code-fenced",
        "text\n\n    literal\n\nmore\n",
        "text\n\n```\nliteral\n```\n\nmore\n",
    ),
    (
        "blank-lines-collapsed",
        "a\n\n\n\nb\n",
        "a\n\nb\n",
    ),
    (
        "frontmatter-quoting-normalized",
        '---\nname: "s"\ndescription: D\n---\n\n# T\n',
        "---\nname: s\ndescription: D\n---\n\n# T\n",
    ),
    (
        "table-pipes-padded",
        "|a|b|\n|-|-|\n|1|2|\n",
        "| a | b |\n| --- | --- |\n| 1   | 2   |\n",
    ),
    # Also from fuzzing the real chain: source churn it produces that must stay
    # quiet, so the projections above do not turn into noise generators.
    (
        # An unlabelled fence spelled `text` is the same fence.
        "fence-labelled-text",
        "```\nliteral\n```\n",
        "```text\nliteral\n```\n",
    ),
    (
        "tilde-fence-converted-to-backticks",
        "~~~\ncontent\n~~~\n",
        "````text\ncontent\n````\n",
    ),
    (
        "hard-break-spaces-become-backslash",
        "line one  \nline two\n",
        "line one\\\nline two\n",
    ),
    (
        "ordered-list-renumbered-consistently",
        "1. one\n1. two\n1. three\n",
        "1. one\n2. two\n3. three\n",
    ),
    (
        "long-sentence-left-unwrapped",
        "Alpha beta gamma delta. Epsilon zeta eta theta.\n",
        "Alpha beta gamma delta.\nEpsilon zeta eta theta.\n",
    ),
    (
        "table-cell-whitespace-padded",
        "| a | b |\n|---|---|\n|1|2|\n",
        "| a   | b   |\n| --- | --- |\n| 1   | 2   |\n",
    ),
    (
        "list-indent-normalized",
        "- one\n    - nested\n- two\n",
        "- one\n  - nested\n- two\n",
    ),
]


def compare(before: str, after: str, tmp_path: Path) -> check_markdown_render.Finding:
    """Render both versions of a document and diff every projection."""
    path = "doc.md"
    (tmp_path / path).write_text(before, encoding="utf-8")
    old = check_markdown_render.render(before, path, tmp_path)
    new = check_markdown_render.render(after, path, tmp_path)
    return check_markdown_render.compare_renders(path, old, new)


@pytest.mark.parametrize(("before", "after"), [c[1:] for c in DAMAGE], ids=[c[0] for c in DAMAGE])
def test_damage_is_reported(before: str, after: str, tmp_path: Path):
    assert compare(before, after, tmp_path), "render damage went undetected"


@pytest.mark.parametrize(("before", "after"), [c[1:] for c in NEUTRAL], ids=[c[0] for c in NEUTRAL])
def test_neutral_churn_is_ignored(before: str, after: str, tmp_path: Path):
    finding = compare(before, after, tmp_path)
    assert not finding, f"false positive: {finding}"


def test_identical_input_is_clean(tmp_path: Path):
    assert not compare("# T\n\nbody\n", "# T\n\nbody\n", tmp_path)


def test_link_multiset_notices_duplicate_count(tmp_path: Path):
    """A lost duplicate must not be masked by the surviving copy."""
    finding = compare("[a](u) [a](u)\n", "[a](u)\n", tmp_path)
    assert finding.links_lost == ["u"]


def test_code_block_anchors_stay_out_of_link_inventory(tmp_path: Path):
    """`--no-highlight` keeps pandoc's per-line code anchors from polluting links."""
    doc = "```python\ndef f():\n    return 1\n```\n"
    assert check_markdown_render.render(doc, "doc.md", tmp_path).links == {}


def test_broken_relative_link_is_reported(tmp_path: Path):
    (tmp_path / "target.md").write_text("# H\n", encoding="utf-8")
    finding = compare("[x](target.md)\n", "[x](gone.md)\n", tmp_path)
    assert finding.targets_broken == ["gone.md"]


def test_surviving_relative_link_is_not_reported(tmp_path: Path):
    (tmp_path / "target.md").write_text("# H\n", encoding="utf-8")
    finding = compare("[x](target.md)\n", "[x](target.md) and more.\n", tmp_path)
    assert not finding.targets_broken


def test_preexisting_broken_link_is_not_a_regression(tmp_path: Path):
    """Only links that used to resolve count; the tool reports regressions."""
    finding = compare("[x](gone.md)\n", "[x](gone.md) and more.\n", tmp_path)
    assert not finding.targets_broken


def test_broken_anchor_is_reported(tmp_path: Path):
    finding = compare("# Some Heading\n\n[x](#some-heading)\n", "# Other Heading\n\n[x](#some-heading)\n", tmp_path)
    assert finding.targets_broken == ["#some-heading"]


def test_percent_encoded_link_resolves_to_the_file_it_names(tmp_path: Path):
    """An href is URL syntax: `%20` names a space, so the file is there."""
    (tmp_path / "my file.md").write_text("# H\n", encoding="utf-8")
    assert check_markdown_render.render("[x](my%20file.md)\n", "doc.md", tmp_path).targets == {"my%20file.md": True}


def test_query_string_is_not_part_of_the_path(tmp_path: Path):
    (tmp_path / "target.md").write_text("# H\n", encoding="utf-8")
    assert check_markdown_render.render("[x](target.md?raw=1)\n", "doc.md", tmp_path).targets == {
        "target.md?raw=1": True
    }


def test_percent_encoded_link_to_a_missing_file_is_still_broken(tmp_path: Path):
    """Decoding must not turn the check into a rubber stamp."""
    assert check_markdown_render.render("[x](no%20such.md)\n", "doc.md", tmp_path).targets == {"no%20such.md": False}


def test_html_comment_churn_is_not_a_rendering_change(tmp_path: Path):
    """Comments are not painted, so editing one changes nothing a reader sees."""
    assert not compare("<!-- note: alpha -->\n\ntext\n", "<!-- note: beta -->\n\ntext\n", tmp_path)
    assert not compare("text\n", "<!-- added -->\n\ntext\n", tmp_path)


def test_text_turned_into_a_comment_is_reported(tmp_path: Path):
    """Dropping comments must not hide prose that vanished into one."""
    assert compare("visible words\n", "<!-- visible words -->\n", tmp_path)


def test_inline_code_span_keeps_its_interior_spacing(tmp_path: Path):
    """Spacing inside a code span is content, unlike the prose around it."""
    assert compare("Run `a  b` now.\n", "Run `a b` now.\n", tmp_path)


def test_code_span_rewrapped_across_a_newline_is_not_reported(tmp_path: Path):
    """CommonMark folds a newline in a code span to a space, so this is neutral churn."""
    assert not compare("Run `a\nb` now.\n", "Run `a b` now.\n", tmp_path)


UNRENDERABLE = '---\nname: s\ndescription: Triggers: "why is this failing"\n---\n\n# Target\n'
"""A document pandoc rejects outright: the inner `: ` makes the frontmatter invalid YAML."""


def test_anchor_into_an_unrenderable_target_is_not_blamed_on_this_file(tmp_path: Path):
    """A target pandoc cannot render says nothing about its anchors, so this file stays clean.

    The href has to change for the check to bite: `compare_renders` discounts links
    that were already broken, and an unchanged href is broken on both sides.
    """
    (tmp_path / "target.md").write_text(UNRENDERABLE, encoding="utf-8")
    finding = compare("[x](./target.md#target)\n", "[x](target.md#target)\n", tmp_path)
    assert not finding.targets_broken


def test_anchor_into_a_renderable_target_is_still_checked(tmp_path: Path):
    """The forgiving path above must not extend to targets that render fine."""
    (tmp_path / "target.md").write_text("# Target\n", encoding="utf-8")
    finding = compare("[x](./target.md#target)\n", "[x](target.md#gone)\n", tmp_path)
    assert finding.targets_broken == ["target.md#gone"]


def test_heading_ids_distinguishes_unrenderable_from_headingless(tmp_path: Path):
    unrenderable = tmp_path / "broken.md"
    unrenderable.write_text(UNRENDERABLE, encoding="utf-8")
    headingless = tmp_path / "prose.md"
    headingless.write_text("Just a paragraph.\n", encoding="utf-8")
    assert check_markdown_render.heading_ids(unrenderable) is None
    assert check_markdown_render.heading_ids(headingless) == set()
    assert check_markdown_render.heading_ids(tmp_path / "absent.md") is None


def test_unparsable_frontmatter_is_reported(tmp_path: Path):
    finding = compare("---\nname: s\n---\n\n# T\n", "---\nname: [unclosed\n---\n\n# T\n", tmp_path)
    assert finding.frontmatter_diff


def test_document_pandoc_refuses_is_reported_not_crashed(tmp_path: Path):
    """An unrenderable file must not read as a clean bill of health."""
    finding = compare(UNRENDERABLE, UNRENDERABLE + "\nA new paragraph.\n", tmp_path)
    assert finding.unrenderable
    assert finding


def test_pandoc_timeout_is_reported_not_crashed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A wedged pandoc has to reach the report as a difference, like any other refusal."""

    def wedged(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pandoc", timeout=check_markdown_render.PANDOC_TIMEOUT)

    # Written first, so `heading_ids` gets as far as pandoc instead of stopping
    # at the missing file, which returns None for a different reason.
    (tmp_path / "doc.md").write_text("# T\n", encoding="utf-8")
    monkeypatch.setattr(check_markdown_render.subprocess, "run", wedged)
    with pytest.raises(check_markdown_render.PandocError):
        check_markdown_render.to_html("# T\n")
    assert check_markdown_render.render("# T\n", "doc.md", tmp_path).error
    assert check_markdown_render.heading_ids(tmp_path / "doc.md") is None


def test_external_urls_are_not_resolved(tmp_path: Path):
    """Reachability is a separate concern; only repo-relative targets are checked."""
    assert check_markdown_render.render("[x](https://example.invalid/nope)\n", "doc.md", tmp_path).targets == {}


def test_pre_content_is_not_whitespace_collapsed(tmp_path: Path):
    doc = "```\n  a\n\n  b\n```\n"
    document = check_markdown_render.render(doc, "doc.md", tmp_path).document
    assert document == ["<pre>", "  |  a", "  |", "  |  b", "</pre>"]


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repo with committer identity and signing forced off."""
    for name, value in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
    }.items():
        monkeypatch.setenv(name, value)
    git(tmp_path, "init", "-q", str(tmp_path))
    return tmp_path


def git(repo: Path, *args: str) -> None:
    """Run a git command in `repo`."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)  # noqa: S603


def run_script(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the script under test inside `repo`."""
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args], cwd=repo, capture_output=True, text=True, check=False
    )


def commit_all(repo: Path) -> None:
    """Stage and commit everything in `repo`."""
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "x")


def test_script_reports_damage_end_to_end(git_repo: Path):
    """The script exits 1 and names the damaged file."""
    doc = git_repo / "doc.md"
    doc.write_text("| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
    commit_all(git_repo)
    doc.write_text("| a | b |\n|---|---|\n| 1 2 |\n", encoding="utf-8")

    result = run_script(git_repo)
    assert result.returncode == 1, result.stderr
    # The report is the output of the run, so it must survive a redirect.
    assert "doc.md" in result.stdout


def test_script_is_clean_on_neutral_churn(git_repo: Path):
    """Rewrapping alone must exit 0."""
    doc = git_repo / "doc.md"
    doc.write_text("Alpha beta gamma. Delta epsilon zeta.\n", encoding="utf-8")
    commit_all(git_repo)
    doc.write_text("Alpha beta gamma.\nDelta epsilon zeta.\n", encoding="utf-8")

    result = run_script(git_repo)
    assert result.returncode == 0, result.stderr


# A hook that rewrites markdown with a sed program kept in a tracked file.  The
# program is deliberately *not* a markdown config file: a mirror that copied only
# a curated list of config would leave sed with nothing to read, the hook would
# rewrite nothing, and every assertion below would fail.
HOOK_CONFIG = """\
repos:
  - repo: local
    hooks:
      - id: rewrite
        name: rewrite
        language: system
        entry: sed -i.bak -f rules.sed
        files: \\.md$
"""

RUNNER = next((name for name in ("prek", "pre-commit") if shutil.which(name)), None)
requires_runner = pytest.mark.skipif(RUNNER is None, reason="a hook runner is required")


def setup_hook_repo(repo: Path, sed_program: str, doc: str) -> None:
    """Commit a repo whose only hook rewrites markdown through `sed_program`."""
    (repo / ".pre-commit-config.yaml").write_text(HOOK_CONFIG, encoding="utf-8")
    (repo / "rules.sed").write_text(sed_program, encoding="utf-8")
    (repo / "doc.md").write_text(doc, encoding="utf-8")
    commit_all(repo)


@requires_runner
def test_hook_mode_leaves_the_worktree_untouched(git_repo: Path):
    """The hooks run on a mirror; the real files must not be rewritten."""
    original = "* one\n* two\n"
    setup_hook_repo(git_repo, "s/one/ONE/\n", original)

    result = run_script(git_repo, "--run-hooks")
    assert (git_repo / "doc.md").read_text(encoding="utf-8") == original
    assert result.returncode == 1, result.stderr
    assert "ONE" in result.stdout


@requires_runner
def test_hook_mode_passes_neutral_rewrites(git_repo: Path):
    """A hook that rewrites without changing the rendering must exit 0."""
    setup_hook_repo(git_repo, "s/^[*] /- /\n", "* one\n* two\n")

    result = run_script(git_repo, "--run-hooks")
    # The rewrite has to have happened, or the clean exit proves nothing.
    assert "rewrote 1 file(s)" in result.stderr
    assert result.returncode == 0, result.stdout
    assert "no rendering differences found" in result.stdout


@requires_runner
def test_hook_id_selects_a_single_hook(git_repo: Path):
    """`--hook ID` implies --run-hooks and reaches the runner as a hook, not a path."""
    setup_hook_repo(git_repo, "s/one/ONE/\n", "* one\n* two\n")

    result = run_script(git_repo, "--hook", "rewrite")
    assert result.returncode == 1, result.stderr
    assert "ONE" in result.stdout


@requires_runner
def test_unknown_hook_id_is_surfaced(git_repo: Path):
    """A hook that never ran would otherwise report a clean run by doing nothing."""
    setup_hook_repo(git_repo, "s/one/ONE/\n", "* one\n* two\n")

    result = run_script(git_repo, "--hook", "nosuchhook")
    assert result.returncode == 0
    assert "rewrote nothing" in result.stderr


def test_hook_mode_needs_a_hook_config(git_repo: Path):
    """Without a config there are no hooks to run; say so instead of comparing nothing."""
    (git_repo / "doc.md").write_text("x\n", encoding="utf-8")
    commit_all(git_repo)
    result = run_script(git_repo, "--run-hooks")
    assert result.returncode == 2
    assert ".pre-commit-config.yaml" in result.stderr


@pytest.mark.parametrize("revision_arg", [("--base", "HEAD~1"), ("--head", "HEAD")])
def test_hook_mode_rejects_revision_args(git_repo: Path, revision_arg: tuple[str, str]):
    """The two modes are exclusive; combining them must not silently drop one."""
    (git_repo / "doc.md").write_text("x\n", encoding="utf-8")
    commit_all(git_repo)
    result = run_script(git_repo, "--run-hooks", *revision_arg)
    assert result.returncode == 2
    assert "do not apply" in result.stderr


def test_diagnostics_stay_off_stdout(git_repo: Path):
    """Only the report goes to stdout, so a redirect captures findings and nothing else."""
    doc = git_repo / "doc.md"
    doc.write_text("| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
    commit_all(git_repo)
    doc.write_text("| a | b |\n|---|---|\n| 1 2 |\n", encoding="utf-8")

    result = run_script(git_repo, "-v")
    assert result.stdout.startswith("compared 1 markdown file(s)")
    assert "DOCUMENT CHANGED" in result.stdout


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [__file__, "-v", "--rootdir", str(Path(__file__).parent), "--confcutdir", str(Path(__file__).parent)]
        )
    )
