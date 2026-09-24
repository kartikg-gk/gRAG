from __future__ import annotations

import hmac
import inspect

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError


def test_reconcile_task_and_lock_contract():
    from src.worker import compile as compile_module
    from src.worker.app import COMPILE_TASK
    from src.worker.locks import lock_key

    task = compile_module.reconcile_org_to_head
    assert COMPILE_TASK == "worker.tasks.reconcile_org_to_head"
    assert task.name == COMPILE_TASK
    assert task.max_retries == 5
    assert inspect.ismethod(task.run)
    assert lock_key("org_1") == "reconcile:lock:org_1"


def test_control_plane_uses_exact_table_and_field_names():
    from src.models.control_plane import (
        ApiKey,
        GraphArtifact,
        IngestJob,
        Organization,
        Pod,
        PodAssignment,
        Repository,
    )

    assert {
        Organization.__tablename__,
        Repository.__tablename__,
        GraphArtifact.__tablename__,
        IngestJob.__tablename__,
        Pod.__tablename__,
        PodAssignment.__tablename__,
        ApiKey.__tablename__,
    } == {
        "organization",
        "repository",
        "graphartifact",
        "ingestjob",
        "pod",
        "podassignment",
        "apikey",
    }
    assert "desired_artifact_id" in Organization.model_fields
    assert {"last_synced_cursor", "last_synced_at", "github_token"} <= set(
        Repository.model_fields
    )
    assert {"s3_uri", "checksum_sha256", "built_by_job_id"} <= set(
        GraphArtifact.model_fields
    )
    assert {"produced_artifact_id", "cursor_to"} <= set(IngestJob.model_fields)

    targets = {
        foreign_key.target_fullname
        for model in (
            Organization,
            Repository,
            GraphArtifact,
            IngestJob,
            PodAssignment,
            ApiKey,
        )
        for foreign_key in model.__table__.foreign_keys
    }
    assert targets == {
        "organization.org_id",
        "repository.repo_id",
        "graphartifact.artifact_id",
        "ingestjob.job_id",
        "pod.pod_id",
    }


def test_part4_setting_defaults(monkeypatch):
    import importlib

    from src.common import config as common_config
    from src.worker import config as worker_config

    for name in (
        "GRAPHRAG_POD_ID",
        "GRAPHRAG_RECONCILE_INTERVAL",
        "GRAPHRAG_POD_HEARTBEAT_WINDOW",
        "GRAPHRAG_DEBOUNCE_WINDOW",
        "GRAPHRAG_DEBOUNCE_MAX_WAIT",
        "GRAPHRAG_SWEEP_INTERVAL",
        "GRAPHRAG_COMPILE_LOCK_TTL",
    ):
        monkeypatch.delenv(name, raising=False)

    common = importlib.reload(common_config)
    worker = importlib.reload(worker_config)
    assert common.POD_ID == "pod-local"
    assert common.RECONCILE_INTERVAL_SECONDS == 5
    assert common.POD_HEARTBEAT_WINDOW_SECONDS == 120
    assert worker.DEBOUNCE_WINDOW_SECONDS == 120
    assert worker.DEBOUNCE_MAX_WAIT_SECONDS == 600
    assert worker.SWEEP_INTERVAL_SECONDS == 30
    assert worker.COMPILE_LOCK_TTL == 1800


def test_vault_exact_errors_and_round_trip(monkeypatch):
    from src import vault

    monkeypatch.delenv("GRAPHRAG_ENCRYPTION_MASTER_KEY", raising=False)
    assert vault.is_configured() is False
    with pytest.raises(vault.VaultError, match="^GRAPHRAG_ENCRYPTION_MASTER_KEY is not set; refusing to encrypt/decrypt secrets\\.$"):
        vault.encrypt("token")

    monkeypatch.setenv("GRAPHRAG_ENCRYPTION_MASTER_KEY", "bad")
    with pytest.raises(vault.VaultError, match="^GRAPHRAG_ENCRYPTION_MASTER_KEY is invalid — expected a urlsafe-base64 32-byte Fernet key\\.$"):
        vault.encrypt("token")

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("GRAPHRAG_ENCRYPTION_MASTER_KEY", key)
    assert vault.is_configured() is True
    encrypted = vault.encrypt("secret-token")
    assert "secret-token" not in encrypted
    assert vault.decrypt(encrypted) == "secret-token"
    with pytest.raises(vault.VaultError):
        vault.encrypt("")
    with pytest.raises(vault.VaultError):
        vault.decrypt("")

    monkeypatch.setenv(
        "GRAPHRAG_ENCRYPTION_MASTER_KEY", Fernet.generate_key().decode()
    )
    with pytest.raises(vault.VaultError, match="^could not decrypt token \\(wrong key or corrupt data\\)\\.$"):
        vault.decrypt(encrypted)


def test_repository_token_is_encrypted(monkeypatch):
    from src.models.control_plane import Repository

    monkeypatch.setenv(
        "GRAPHRAG_ENCRYPTION_MASTER_KEY", Fernet.generate_key().decode()
    )
    repository = Repository(
        repo_id="repo_1",
        org_id="org_1",
        provider="github",
        provider_repo_id="acme/repo",
        name="acme/repo",
        status="active",
        created_at=1,
    )
    repository.set_github_token("plain-token")
    assert repository.github_token != "plain-token"
    assert "plain-token" not in repository.github_token
    assert repository.get_github_token() == "plain-token"


def test_onboarding_rejects_github_token():
    from src.api.onboarding import ProvisionRequest

    with pytest.raises(ValidationError):
        ProvisionRequest.model_validate(
            {
                "tenant_name": "Acme",
                "repo_name": "acme/repo",
                "github_token": "must-not-be-accepted",
            }
        )


def test_api_key_verification_is_constant_time_and_rejects_revoked(monkeypatch):
    from datetime import datetime, timezone

    from src.control_plane import ApiKeyRecord, verify_key

    seen = []

    def compare(left, right):
        seen.append((left, right))
        return left == right

    monkeypatch.setattr(hmac, "compare_digest", compare)
    record = ApiKeyRecord(
        key_id="key_1",
        hashed_key="digest",
        org_id="org_1",
        scopes="read",
        prefix="12345678",
        created_at=datetime.now(timezone.utc),
        revoked_at=datetime.now(timezone.utc),
    )
    assert verify_key(record, "digest") is None
    assert seen == [("digest", "digest")]
