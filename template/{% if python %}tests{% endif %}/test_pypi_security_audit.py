"""
Security audit tests using pip-audit to detect known vulnerabilities.

This test runs pip-audit against the installed packages and fails if any
vulnerabilities are detected, ensuring continuous security monitoring.

To ignore a CVE, add an entry to ``IGNORED_VULNERABILITIES`` below with a
justification. Entries are passed to pip-audit via ``--ignore-vuln``.

Ref: https://gist.github.com/mikeckennedy/de70ce13231b407a8dccea758f83a5cd
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

# Map of CVE id -> reason for ignoring. Revisit periodically; remove entries
# once an upstream fix is released or the risk assessment changes.
IGNORED_VULNERABILITIES: dict[str, str] = {
    # "CVE-2025-53000": "nbconvert Windows-only vulnerability (no risk on Linux/macOS)",
    "CVE-2025-69872": "#nofix / #wontfix per https://github.com/grantjenks/python-diskcache/issues/357",
}


def _ignore_vuln_args() -> list[str]:
    args: list[str] = []
    for cve in IGNORED_VULNERABILITIES:
        args.extend(["--ignore-vuln", cve])
    return args


def _format_fix_versions(fix_versions: list[str]) -> str:
    if not fix_versions:
        return "no fix published"
    return ", ".join(fix_versions)


def _summarize_pip_audit_json(raw_output: str) -> str:
    payload = json.loads(raw_output)
    dependencies = payload.get("dependencies", [])

    vulnerable_dependencies = []
    skipped_dependencies = []
    for dependency in dependencies:
        vulns = dependency.get("vulns", [])
        if vulns:
            vulnerable_dependencies.append(dependency)
        elif dependency.get("skip_reason"):
            skipped_dependencies.append(dependency)

    lines: list[str] = []
    lines.append(f"Vulnerable packages: {len(vulnerable_dependencies)}")
    for dependency in vulnerable_dependencies:
        name = dependency["name"]
        version = dependency.get("version", "unknown")
        lines.append(f"- {name} {version}")
        for vulnerability in dependency.get("vulns", []):
            vuln_id = vulnerability.get("id", "unknown-id")
            fix_versions = _format_fix_versions(vulnerability.get("fix_versions", []))
            lines.append(f"  - {vuln_id}; fix: {fix_versions}")

    if skipped_dependencies:
        lines.append("")
        lines.append(f"Skipped dependencies: {len(skipped_dependencies)}")
        for dependency in skipped_dependencies:
            lines.append(f"- {dependency['name']}: {dependency['skip_reason']}")

    return "\n".join(lines)


def test_pip_audit_no_vulnerabilities():
    """
    Run pip-audit to check for known security vulnerabilities.

    This test will fail if any vulnerabilities are detected in the installed packages.

    To run this test specifically:
        pytest tests/test_pypi_security_audit.py -v
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent

    # Run pip-audit with JSON output for easier parsing
    try:
        result = subprocess.run(  # NOQA: S603
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--format=json",
                "--progress-spinner=off",
                "--skip-editable",
                *_ignore_vuln_args(),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
    except subprocess.TimeoutExpired:
        pytest.fail("pip-audit command timed out after 120 seconds")
    except FileNotFoundError:
        pytest.fail("pip-audit not installed or not accessible")

    # Check if pip-audit found any vulnerabilities
    if result.returncode != 0:
        # pip-audit returns non-zero when vulnerabilities are found
        error_output = result.stdout + "\n" + result.stderr

        # Check if it's an actual vulnerability vs an error
        if "vulnerabilities found" in error_output.lower() or '"dependencies"' in result.stdout:
            try:
                summarized_output = _summarize_pip_audit_json(result.stdout)
            except json.JSONDecodeError:
                summarized_output = result.stdout
            pytest.fail(
                f"pip-audit detected security vulnerabilities!\n\n"
                f"Output:\n{summarized_output}\n\n"
                f"Please review and update vulnerable packages.\n"
                f"Run manually with: python -m pip_audit --skip-editable"
            )
        else:
            # Some other error occurred
            pytest.fail(
                f"pip-audit failed to run properly:\n\nReturn code: {result.returncode}\nOutput: {error_output}\n"
            )

    # Success - no vulnerabilities found
    if result.returncode != 0:
        pytest.fail("pip-audit should return 0 when no vulnerabilities are found")


def test_pip_audit_runs_successfully():
    """
    Verify that pip-audit can run successfully (even if vulnerabilities are found).

    This is a smoke test to ensure pip-audit is properly installed and functional.
    """
    try:
        result = subprocess.run(  # NOQA: S603
            [sys.executable, "-m", "pip_audit", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            pytest.fail(f"pip-audit --version failed: {result.stderr}")

        if "pip-audit" not in result.stdout.lower():
            pytest.fail("pip-audit version output unexpected")
    except FileNotFoundError:
        pytest.fail("pip-audit not installed")
    except subprocess.TimeoutExpired:
        pytest.fail("pip-audit --version timed out")
