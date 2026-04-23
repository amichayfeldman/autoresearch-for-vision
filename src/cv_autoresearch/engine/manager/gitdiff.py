"""Git diff helpers used by the manager."""

from __future__ import annotations

import subprocess
from pathlib import Path


def changed_files(repo_root: str | Path = ".") -> list[str]:
    """Return modified tracked/untracked files relative to repo root."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    files: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        files.append(line[3:].strip())
    return files


def diff_for_files(files: list[str], repo_root: str | Path = ".") -> str:
    """Return git diff text for changed files."""
    if not files:
        return ""
    result = subprocess.run(
        ["git", "diff", "--", *files],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def new_changes(before: list[str], after: list[str]) -> list[str]:
    """Return files newly changed after an agent run."""
    return sorted(set(after) - set(before))
