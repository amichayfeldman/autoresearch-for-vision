"""Editable-boundary validation for agent changes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundaryValidation:
    """Result of checking changed paths."""

    valid: bool
    reasons: list[str]


def normalize_path(path: str | Path) -> str:
    """Normalize a path to POSIX style without leading ./."""
    return Path(path).as_posix().lstrip("./")


def path_in_prefixes(path: str, prefixes: list[str]) -> bool:
    normalized = normalize_path(path)
    return any(normalized == normalize_path(prefix).rstrip("/") or normalized.startswith(normalize_path(prefix).rstrip("/") + "/") for prefix in prefixes)


def validate_changed_files(
    changed_files: list[str],
    *,
    editable_paths: list[str],
    forbidden_paths: list[str],
) -> BoundaryValidation:
    """Reject forbidden-path edits and files outside the editable surface."""
    reasons: list[str] = []
    for changed in changed_files:
        if path_in_prefixes(changed, forbidden_paths):
            reasons.append(f"forbidden path changed: {changed}")
        if not path_in_prefixes(changed, editable_paths):
            reasons.append(f"path is outside editable surface: {changed}")
    return BoundaryValidation(valid=not reasons, reasons=reasons)


def detect_multiple_unrelated_changes(changed_files: list[str]) -> bool:
    """Heuristic guardrail for the exactly-one-change contract."""
    if len(changed_files) <= 1:
        return False
    roots = {"/".join(normalize_path(path).split("/")[:5]) for path in changed_files}
    return len(roots) > 1
