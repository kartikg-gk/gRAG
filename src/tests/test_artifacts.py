"""Tests for moving an artifact somewhere durable and getting it back.

Everything here runs on the local backend, on real files in ``tmp_path``,
because that backend is the default and is meant to work with nothing
installed and nothing running. The cloud backend is exercised only where it
can be without a service: that its client is not imported until it is used,
and that missing configuration is reported clearly at first use.

The dispatch test is the one that matters most. Reading with "whatever is
configured now" works until the configuration changes, and then fails on
exactly the artifacts that already existed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.artifacts import (
    CHUNK_BYTES,
    ArtifactError,
    BackendNotConfigured,
    artifact_key,
    checksum,
    get_artifact,
    local_path_for,
    pod_cache_path,
    put_artifact,
)


@pytest.fixture
def built(tmp_path):
    """A file standing in for a built graph."""
    path = tmp_path / "built.lbug"
    path.write_bytes(b"a graph, of sorts" * 100)
    return path


# ==========================================================================
# round trip
# ==========================================================================


def test_an_artifact_survives_a_round_trip_byte_for_byte(tmp_path, built):
    root = tmp_path / "store"

    uri = put_artifact(built, artifact_key("org_a", "v1"), backend="local", root=root)
    fetched = get_artifact(uri, tmp_path / "back" / "graph.lbug", root=root)

    assert fetched.read_bytes() == built.read_bytes()
    assert checksum(fetched) == checksum(built)


def test_the_returned_uri_carries_the_scheme_of_the_backend_that_wrote_it(
    tmp_path, built
):
    uri = put_artifact(
        built, artifact_key("org_a", "v1"), backend="local", root=tmp_path / "store"
    )

    assert uri.startswith("local://")
    assert "artifacts/org_a/v1.lbug" in uri


def test_directories_are_created_on_demand(tmp_path, built):
    """No administrative step before first use."""
    root = tmp_path / "not" / "yet" / "there"

    uri = put_artifact(built, artifact_key("org_a", "v1"), backend="local", root=root)
    get_artifact(uri, tmp_path / "deep" / "nested" / "graph.lbug", root=root)

    assert (tmp_path / "deep" / "nested" / "graph.lbug").is_file()


def test_storing_something_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(ArtifactError):
        put_artifact(tmp_path / "absent.lbug", artifact_key("org_a", "v1"), backend="local")


def test_fetching_something_never_stored_is_an_error_not_an_empty_file(tmp_path):
    with pytest.raises(ArtifactError):
        get_artifact(
            "local://artifacts/org_a/v9.lbug",
            tmp_path / "out.lbug",
            root=tmp_path / "store",
        )

    assert not (tmp_path / "out.lbug").exists()


# ==========================================================================
# the dispatch rule
# ==========================================================================


def test_get_dispatches_on_the_uri_scheme_not_the_current_default(
    tmp_path, built, monkeypatch
):
    """The rule this module exists to hold.

    Written under the local backend, then the process default is changed to
    the cloud one. The fetch must still take the local path — if it consulted
    configuration it would try the cloud and fail, which is exactly what
    happens to every artifact that predates a configuration change.
    """
    root = tmp_path / "store"
    uri = put_artifact(built, artifact_key("org_a", "v1"), backend="local", root=root)

    # The default is now the other backend, with nothing configured for it.
    monkeypatch.setattr("src.artifacts.ARTIFACT_BACKEND", "cloud")
    monkeypatch.setattr("src.artifacts.ARTIFACT_BUCKET", "")

    fetched = get_artifact(uri, tmp_path / "back.lbug", root=root)

    assert fetched.read_bytes() == built.read_bytes()


def test_a_cloud_uri_is_never_served_from_the_local_backend(tmp_path, monkeypatch):
    """The reverse: a cloud address does not quietly become a local read."""
    monkeypatch.setattr("src.artifacts.ARTIFACT_BACKEND", "local")
    monkeypatch.setattr("src.artifacts.ARTIFACT_BUCKET", "")

    with pytest.raises(BackendNotConfigured):
        get_artifact("s3://some-bucket/artifacts/org_a/v1.lbug", tmp_path / "out.lbug")


def test_an_unknown_scheme_is_refused_rather_than_guessed(tmp_path):
    with pytest.raises(ArtifactError, match="unknown artifact scheme"):
        get_artifact("ftp://elsewhere/thing.lbug", tmp_path / "out.lbug")


def test_an_unknown_backend_is_refused(tmp_path, built):
    with pytest.raises(ArtifactError, match="unknown artifact backend"):
        put_artifact(built, artifact_key("org_a", "v1"), backend="carrier-pigeon")


# ==========================================================================
# keys and paths
# ==========================================================================


def test_the_same_tenant_and_version_always_give_the_same_key():
    """A writer and a reader that never speak still agree on the location."""
    assert artifact_key("org_a", "v1") == artifact_key("org_a", "v1")
    assert artifact_key("org_a", "v1") == "artifacts/org_a/v1.lbug"
    assert artifact_key("org_a", "1") == "artifacts/org_a/v1.lbug"


def test_different_tenants_or_versions_give_different_keys():
    keys = {
        artifact_key("org_a", "v1"),
        artifact_key("org_a", "v2"),
        artifact_key("org_b", "v1"),
    }

    assert len(keys) == 3


@pytest.mark.parametrize(
    "tenant, version", [("", "v1"), ("org_a", ""), ("", "")]
)
def test_a_key_needs_both_halves(tenant, version):
    with pytest.raises(ArtifactError):
        artifact_key(tenant, version)


def test_the_local_path_is_derivable_from_the_key(tmp_path):
    key = artifact_key("org_a", "v1")

    assert local_path_for(key, root=tmp_path) == tmp_path / key


def test_a_pod_cache_path_separates_pods_holding_the_same_tenant(tmp_path):
    """Two processes on one machine must not write the same file."""
    first = pod_cache_path("pod-1", "org_a", "v1", root=tmp_path)
    second = pod_cache_path("pod-2", "org_a", "v1", root=tmp_path)

    assert first != second
    assert first.parent != second.parent
    assert first.name == second.name == "v1.lbug"


def test_a_pod_cache_path_is_stable_for_the_same_three(tmp_path):
    assert pod_cache_path("pod-1", "org_a", "v1", root=tmp_path) == pod_cache_path(
        "pod-1", "org_a", "v1", root=tmp_path
    )


# ==========================================================================
# integrity
# ==========================================================================


def test_a_checksum_is_stable_across_repeated_computation(built):
    assert checksum(built) == checksum(built)


def test_a_checksum_changes_when_one_byte_changes(tmp_path):
    path = tmp_path / "artifact.lbug"
    path.write_bytes(b"aaaaaaaa")
    before = checksum(path)

    path.write_bytes(b"aaaaaaab")

    assert checksum(path) != before


def test_a_checksum_reads_in_chunks_rather_than_whole(tmp_path, monkeypatch):
    """A built graph can be large enough that this is a memory decision.

    Asserted by counting reads on a file bigger than one chunk: reading it
    whole would be a single call.
    """
    path = tmp_path / "big.lbug"
    path.write_bytes(b"x" * (CHUNK_BYTES * 2 + 17))

    reads = []
    real_open = open

    def counting_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        real_read = handle.read

        def read(size=-1):
            reads.append(size)
            return real_read(size)

        handle.read = read
        return handle

    monkeypatch.setattr("builtins.open", counting_open)
    checksum(path)

    assert len(reads) >= 3
    assert all(size == CHUNK_BYTES for size in reads)


def test_a_checksum_of_an_empty_file_is_still_a_digest(tmp_path):
    path = tmp_path / "empty.lbug"
    path.write_bytes(b"")

    assert len(checksum(path)) == 64


# ==========================================================================
# the cloud backend, without a cloud
# ==========================================================================


def test_the_cloud_backend_with_no_bucket_names_what_is_unset(tmp_path, built, monkeypatch):
    monkeypatch.setattr("src.artifacts.ARTIFACT_BUCKET", "")

    with pytest.raises(BackendNotConfigured) as caught:
        put_artifact(built, artifact_key("org_a", "v1"), backend="cloud")

    assert "GRAPHRAG_ARTIFACT_BUCKET" in str(caught.value)


def test_the_cloud_backend_fails_at_first_use_not_at_import():
    """An import that failed on an unset bucket would break every process.

    A fresh interpreter with nothing configured imports the module, calls the
    functions that need no backend, and only fails when the cloud one is
    actually asked for.
    """
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys;"
            "[os.environ.pop(n) for n in list(os.environ) if n.startswith('GRAPHRAG_ARTIFACT')];"
            "import src.artifacts as a;"
            "print('imported');"
            "print(a.artifact_key('org_a', 'v1'));"
            "\n"
            "try:\n"
            "    a.put_artifact(sys.executable, 'k', backend='cloud')\n"
            "except a.BackendNotConfigured as exc:\n"
            "    print('failed at use:', 'GRAPHRAG_ARTIFACT_BUCKET' in str(exc))\n",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )

    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout
    assert "artifacts/org_a/v1.lbug" in result.stdout
    assert "failed at use: True" in result.stdout


def test_the_cloud_client_is_not_imported_unless_that_backend_is_used():
    """Importing this module must need nothing beyond the runtime set."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;"
            "import src.artifacts;"
            "print('boto3' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_the_module_imports_with_the_cloud_client_blocked(tmp_path):
    """And the local backend keeps working while it is unavailable."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['boto3'] = None;"
            "import src.artifacts as a;"
            "print(a.artifact_key('org_a', 'v1'))",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "artifacts/org_a/v1.lbug"


def test_this_module_imports_nothing_from_the_rest_of_the_project():
    """It is the bottom of the pipeline: everything calls in, it calls out."""
    import ast

    import src.artifacts as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    forbidden = {"engine", "graphdb", "registry", "retrieval", "knowledge", "api"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (forbidden & set((node.module or "").split("."))), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not (forbidden & set(alias.name.split("."))), alias.name
