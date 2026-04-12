from __future__ import annotations

from pathlib import Path

from moneygone.utils.env import load_repo_env


def test_load_repo_env_reads_runtime_and_deploy_overlays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BASE_ONLY", raising=False)
    monkeypatch.delenv("RUNTIME_ONLY", raising=False)
    monkeypatch.delenv("DEPLOY_ONLY", raising=False)

    (tmp_path / ".env").write_text("BASE_ONLY=base\n")
    (tmp_path / ".env.runtime").write_text("export RUNTIME_ONLY=runtime\n")
    (tmp_path / ".deploy.env").write_text("DEPLOY_ONLY=deploy\n")

    loaded = load_repo_env(tmp_path)

    assert "BASE_ONLY" in loaded
    assert "RUNTIME_ONLY" in loaded
    assert "DEPLOY_ONLY" in loaded
