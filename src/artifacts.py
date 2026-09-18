"""Moving a built graph somewhere durable, and getting it back intact.

A file goes in, a URI comes out; a URI goes in, a file comes back. That is the
whole surface. This module opens no database, holds no handle, and imports
nothing else from this project beyond the configuration it reads — everything
downstream of it eventually calls in, and it calls out to nothing.

Two backends, and the default is the one that needs nothing
-----------------------------------------------------------

The local backend copies files under a configured directory. No service, no
credentials, no optional package. It is what development and the tests use,
and it is a real backend rather than a stand-in for one.

The cloud backend talks to an object store and is chosen only when configured.
**Its client library is imported inside the call that needs it**, so importing
this module needs nothing installed and only actually using that backend does.
Missing configuration is reported at first use, naming what is unset, rather
than at import — an import that fails because a bucket is unnamed would break
every process that merely mentions this module.

A read follows the URI, not the configuration
---------------------------------------------

**``get`` dispatches on the scheme of the URI it was handed.** An artifact
written to the cloud is fetched from the cloud even if the process default has
since become local, and the reverse.

This is the one rule here worth stating twice. Reading with "whatever is
configured now" works perfectly until the configuration changes, and then it
fails on exactly the artifacts that already exist — the ones written before the
change, which is to say the important ones. The scheme is written into the URI
at put time precisely so a reader never has to guess.

Layout
------

A key is ``artifacts/<tenant>/v<version>.lbug`` and the local backend puts it at
``<root>/<key>``, so a key and the root give the path with no lookup. The
cloud backend uses the same key as the object name under its bucket, so the
two layouts read the same in a listing.

Directories are made on demand. Nothing here needs an administrative step
before first use.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlparse

from .common.config import (
    ARTIFACT_BACKEND,
    ARTIFACT_BUCKET,
    ARTIFACT_PREFIX,
    ARTIFACT_REGION,
    ARTIFACT_ROOT,
    POD_CACHE_ROOT,
)

#: The two backends, and the URI scheme each one writes.
LOCAL = "local"
CLOUD = "cloud"
LOCAL_SCHEME = "local"
CLOUD_SCHEME = "s3"

#: Bytes read at a time when checksumming. A built graph can be large enough
#: that reading it whole to hash it is a memory decision rather than a
#: convenience, and this makes the cost independent of the file's size.
CHUNK_BYTES = 1024 * 1024

#: Extension a stored artifact carries. The stores this moves are single
#: files, so the key names one.
ARTIFACT_SUFFIX = ".lbug"


class ArtifactError(RuntimeError):
    """An artifact could not be stored or fetched.

    One type for both directions. A caller can tell them apart by which
    function raised, and every failure here means the same thing to the code
    above it: the file is not where it was supposed to end up.
    """


class BackendNotConfigured(ArtifactError):
    """A backend was selected and something it needs was never set.

    Its own type because it is the one failure a caller can fix without
    looking at logs — the message names the variable.
    """


# --------------------------------------------------------------------------
# addresses
# --------------------------------------------------------------------------


def _canonical_version(version: str) -> str:
    return version if version.startswith("v") else f"v{version}"


def artifact_key(tenant: str, version: str) -> str:
    """Where a tenant's graph at a given version lives, as a storage key.

    Deterministic: the same pair always gives the same key, so a writer and a
    reader that never speak still agree on the location. That is the whole
    point of deriving it rather than recording it somewhere.
    """
    if not tenant or not version:
        raise ArtifactError(
            f"a key needs both a tenant and a version, got {tenant!r} and {version!r}"
        )
    prefix = ARTIFACT_PREFIX.strip("/")
    return f"{prefix}/{tenant}/{_canonical_version(version)}{ARTIFACT_SUFFIX}"


def local_path_for(key: str, *, root: str | Path | None = None) -> Path:
    """The local backend's path for ``key``.

    The key laid under the root unchanged, so the mapping can be worked out
    from either direction without consulting anything.
    """
    base = Path(root if root is not None else ARTIFACT_ROOT)
    return base / key


def pod_cache_path(
    pod: str, tenant: str, version: str, *, root: str | Path | None = None
) -> Path:
    """Where a downloaded artifact lands on the machine that will open it.

    Keyed by pod as well as tenant and version, because two processes on one
    machine holding the same tenant must not write the same file — one would
    be reading it while the other replaced it.

    Separate from the artifact root: that is storage and this is a cache, and
    a cache is safe to delete.
    """
    base = Path(root if root is not None else POD_CACHE_ROOT)
    return base / pod / tenant / f"{_canonical_version(version)}{ARTIFACT_SUFFIX}"


def _cloud_uri(bucket: str, key: str) -> str:
    return f"{CLOUD_SCHEME}://{bucket}/{key}"


def _local_uri(key: str) -> str:
    return f"{LOCAL_SCHEME}://{key}"


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------


def checksum(path: str | Path) -> str:
    """A digest of a file's contents, read in chunks.

    Chunked deliberately. A built graph can be large, and hashing it by
    reading it whole makes memory a function of the largest artifact anyone
    ever stores — a cost that only shows up on the day it matters.

    This answers "did this arrive intact", not "is this the file I think it
    is" against an adversary. Nothing here defends against someone who can
    write to the store and recompute the digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# the local backend
# --------------------------------------------------------------------------


def _put_local(source: Path, key: str, root: str | Path | None) -> str:
    destination = local_path_for(key, root=root)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise ArtifactError(f"could not store {key}: {exc}") from exc
    return _local_uri(key)


def _get_local(key: str, destination: Path, root: str | Path | None) -> Path:
    source = local_path_for(key, root=root)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise ArtifactError(f"could not fetch {key}: {exc}") from exc
    return destination


# --------------------------------------------------------------------------
# the cloud backend
# --------------------------------------------------------------------------


def _cloud_client(bucket: str | None, region: str | None):
    """The object-store client, and the configuration it needs.

    **Imported here rather than at module scope.** This module has to be
    importable with nothing installed, because everything downstream of it
    imports it and most of that never touches this backend.

    Missing configuration is reported here, at first use, naming the variable
    — an import-time failure would break a process that only mentions this
    module in a type annotation.
    """
    name = bucket if bucket is not None else ARTIFACT_BUCKET
    if not name:
        raise BackendNotConfigured(
            "the cloud backend needs a bucket: set GRAPHRAG_ARTIFACT_BUCKET"
        )

    try:
        import boto3
    except ImportError as exc:
        raise BackendNotConfigured(
            "the cloud backend needs its client library, which is an optional "
            "dependency and is not installed"
        ) from exc

    where = region if region is not None else ARTIFACT_REGION
    client = boto3.client("s3", region_name=where or None)
    return client, name


def _put_cloud(
    source: Path, key: str, bucket: str | None, region: str | None
) -> str:
    client, name = _cloud_client(bucket, region)
    try:
        client.upload_file(str(source), name, key)
    except Exception as exc:
        raise ArtifactError(f"could not store {key}: {exc}") from exc
    return _cloud_uri(name, key)


def _get_cloud(
    bucket: str, key: str, destination: Path, region: str | None
) -> Path:
    client, name = _cloud_client(bucket, region)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(name, key, str(destination))
    except Exception as exc:
        raise ArtifactError(f"could not fetch {key}: {exc}") from exc
    return destination


# --------------------------------------------------------------------------
# the two calls everything else uses
# --------------------------------------------------------------------------


def put_artifact(
    source: str | Path,
    key: str,
    *,
    backend: str | None = None,
    root: str | Path | None = None,
    bucket: str | None = None,
    region: str | None = None,
) -> str:
    """Store ``source`` under ``key``, and return where it now lives.

    The returned URI carries the scheme of whichever backend wrote it, so a
    later read needs nothing but the URI.
    """
    source = Path(source)
    if not source.is_file():
        raise ArtifactError(f"nothing to store at {source}")

    chosen = (backend if backend is not None else ARTIFACT_BACKEND).strip().lower()
    if chosen == LOCAL:
        return _put_local(source, key, root)
    if chosen == CLOUD:
        return _put_cloud(source, key, bucket, region)
    raise ArtifactError(
        f"unknown artifact backend {chosen!r}; expected {LOCAL!r} or {CLOUD!r}"
    )


def get_artifact(
    uri: str,
    destination: str | Path,
    *,
    root: str | Path | None = None,
    region: str | None = None,
) -> Path:
    """Fetch the artifact at ``uri`` to ``destination``.

    **Dispatches on the URI's scheme, never on the configured backend.** An
    artifact written to the cloud is fetched from the cloud whatever this
    process is configured to write to now, and the reverse — otherwise
    changing the default would break exactly the artifacts that already
    exist.
    """
    destination = Path(destination)
    parsed = urlparse(uri)

    if parsed.scheme == LOCAL_SCHEME:
        # A local URI carries the key in the part after the scheme. urlparse
        # puts a bare first segment in netloc, so both halves are rejoined
        # rather than trusting either one alone.
        key = (parsed.netloc + parsed.path).lstrip("/")
        return _get_local(key, destination, root)

    if parsed.scheme == CLOUD_SCHEME:
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ArtifactError(f"not a usable artifact address: {uri!r}")
        return _get_cloud(bucket, key, destination, region)

    raise ArtifactError(
        f"unknown artifact scheme {parsed.scheme!r} in {uri!r}; expected "
        f"{LOCAL_SCHEME!r} or {CLOUD_SCHEME!r}"
    )
