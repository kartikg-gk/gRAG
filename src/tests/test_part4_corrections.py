from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException


def _repository():
    from src.models.control_plane import Repository

    return Repository(
        repo_id="repo_1",
        org_id="org_1",
        provider="github",
        provider_repo_id="acme/repo",
        name="acme/repo",
        status="active",
        created_at=1,
    )


def test_vault_present_but_malformed_key_is_not_configured(monkeypatch):
    from src import vault

    monkeypatch.setenv(vault.MASTER_KEY_VARIABLE, "present-but-invalid")

    assert vault.is_configured() is False


def test_repository_token_helper_sets_gets_and_clears_without_plaintext(monkeypatch):
    from src import vault

    monkeypatch.setenv(vault.MASTER_KEY_VARIABLE, Fernet.generate_key().decode())
    repository = _repository()

    repository.set_github_token("plain-token")
    assert repository.github_token != "plain-token"
    assert "plain-token" not in repository.github_token
    assert repository.get_github_token() == "plain-token"

    repository.set_github_token(None)
    assert repository.github_token is None
    assert repository.get_github_token() is None

    repository.set_github_token("plain-token")
    repository.set_github_token("")
    assert repository.github_token is None
    assert repository.get_github_token() is None


def test_oauth_state_with_extra_colon_is_invalid_not_malformed(monkeypatch):
    from src.api.github_oauth import _validated_org

    monkeypatch.setenv("GRAPHRAG_GITHUB_OAUTH_STATE_SECRET", "state-secret")

    with pytest.raises(HTTPException) as raised:
        _validated_org("org_1:signature:extra")

    assert raised.value.status_code == 400
    assert raised.value.detail == "Invalid OAuth state."


@pytest.mark.parametrize("version", ["1", "v1"])
def test_artifact_and_cache_paths_are_canonical_lbug(tmp_path, version):
    from src.artifacts import artifact_key, pod_cache_path

    assert artifact_key("org_a", version) == "artifacts/org_a/v1.lbug"
    assert pod_cache_path("pod-a", "org_a", version, root=tmp_path) == (
        tmp_path / "pod-a" / "org_a" / "v1.lbug"
    )


def test_artifact_object_prefix_uses_only_the_s3_prefix_setting():
    script = (
        "import importlib; "
        "from src.common import config; "
        "from src import artifacts; "
        "importlib.reload(config); "
        "importlib.reload(artifacts); "
        "print(artifacts.artifact_key('org_a', '1'))"
    )

    def key_with(**settings):
        environment = os.environ.copy()
        environment.pop("GRAPHRAG_ARTIFACT_S3_PREFIX", None)
        environment.pop("GRAPHRAG_ARTIFACT_PREFIX", None)
        environment.update(settings)
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return result.stdout.strip()

    assert key_with() == "artifacts/org_a/v1.lbug"
    assert key_with(GRAPHRAG_ARTIFACT_S3_PREFIX="custom") == (
        "custom/org_a/v1.lbug"
    )
    assert key_with(GRAPHRAG_ARTIFACT_PREFIX="must-be-ignored") == (
        "artifacts/org_a/v1.lbug"
    )


def test_boot_hydration_does_not_gate_or_rehash_recorded_artifact(
    monkeypatch, tmp_path
):
    from src import pod as pod_module

    assignment = SimpleNamespace(org_id="org_a", artifact_id="artifact_1")
    artifact = SimpleNamespace(
        artifact_id="artifact_1",
        version=1,
        s3_uri="local://artifacts/org_a/v1.lbug",
        checksum_sha256=None,
    )

    class Rows:
        @staticmethod
        def all():
            return [assignment]

    class Session:
        @staticmethod
        def exec(_statement):
            return Rows()

        @staticmethod
        def get(_model, key):
            return artifact if key == artifact.artifact_id else None

    class Registry:
        def __init__(self):
            self.replaced = []

        def replace(self, org_id, *, path, version):
            self.replaced.append((org_id, Path(path), version))

    def download(_uri, destination, *, root=None):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"compiled graph")
        return destination

    def forbidden_rehash(*_args, **_kwargs):
        raise AssertionError("hydrate must not re-hash an already-served artifact")

    registry = Registry()
    monkeypatch.setattr(pod_module, "get_artifact", download)
    monkeypatch.setattr(
        pod_module, "_arrived_intact", forbidden_rehash, raising=False
    )

    hydrated = pod_module.hydrate(
        Session(), pod_id="pod-a", registry=registry, cache_root=tmp_path
    )

    expected = tmp_path / "pod-a" / "org_a" / "v1.lbug"
    assert hydrated == [pod_module.Hydrated(org_id="org_a", version=1)]
    assert registry.replaced == [("org_a", expected, "1")]
