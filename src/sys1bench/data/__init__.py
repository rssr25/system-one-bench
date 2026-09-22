"""Packaged data: paraphrase/criteria framing sets per generator and the fixed drift canary.

    from sys1bench.data import framings_path, canary_path
    framings_path("support_tickets")   -> Path to the packaged YAML (or a user path if one is given)
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_PKG = "sys1bench.data"


def _pkg_file(sub: str, name: str) -> Path:
    return Path(str(resources.files(_PKG).joinpath(sub, name)))


def framings_path(name_or_path: str | Path) -> Path:
    """Resolve a framing set: an existing file path is returned as is; otherwise `name` maps to the packaged
    `framings/<name>.yaml` (e.g. "support_tickets", "phishing_email")."""
    p = Path(name_or_path)
    if p.exists():
        return p
    stem = p.stem if p.suffix else str(name_or_path)
    cand = _pkg_file("framings", f"{stem}.yaml")
    if cand.exists():
        return cand
    raise FileNotFoundError(f"no framing set {name_or_path!r}; packaged: {sorted(list_framings())}")


def list_framings() -> list[str]:
    return [p.stem for p in Path(str(resources.files(_PKG).joinpath("framings"))).glob("*.yaml")]


def canary_path() -> Path:
    return _pkg_file("canary", "canary.jsonl")
