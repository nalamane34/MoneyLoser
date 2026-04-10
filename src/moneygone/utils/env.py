"""Small helpers for loading repo-local environment files."""

from __future__ import annotations

import os
from pathlib import Path


def load_repo_env(repo_root: Path, *, override: bool = False) -> list[str]:
    """Load simple KEY=VALUE entries from ``repo_root/.env`` into ``os.environ``.

    Supports blank lines, comments, and optional ``export`` prefixes.
    Returns the list of keys that were set during this call.
    """
    env_path = repo_root / ".env"
    if not env_path.exists():
        return []

    loaded: list[str] = []
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)

    return loaded
