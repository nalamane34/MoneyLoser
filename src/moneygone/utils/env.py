"""Small helpers for loading repo-local environment files."""

from __future__ import annotations

import os
from pathlib import Path


def _load_env_file(env_path: Path, *, override: bool = False) -> list[str]:
    """Load a single env file into ``os.environ``."""
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


def load_repo_env(repo_root: Path, *, override: bool = False) -> list[str]:
    """Load common repo-local env files into ``os.environ``.

    We load ``.env`` first, then runtime/deploy overlays if present, while
    preserving already-set keys by default. Supports blank lines, comments,
    and optional ``export`` prefixes. Returns the list of keys set during
    this call.
    """
    loaded: list[str] = []
    for filename in (".env", ".env.runtime", ".deploy.env"):
        loaded.extend(_load_env_file(repo_root / filename, override=override))
    return loaded
