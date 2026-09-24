"""Tests for the control plane's tables.

**Every constraint here is asserted against the database, not against a Python
check.** A rule enforced by code that happens to be the only writer today is no
rule at all the moment a second writer exists, which is precisely the situation
the one-build-at-a-time index and the declared references are for. So these
tests write rows and let the database refuse them.

The engine is SQLite on a file, with foreign keys turned on the way the
connection factory turns them on everywhere. The dialect the deployment uses is
a server; what is checked here is what both dialects share — the constraints
themselves — plus, where the two differ, the statements the schema actually
emits for each.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable
from sqlmodel import Field, SQLModel, select

from src.control_plane import hash_api_key, open_control_plane
from src.models.control_plane import (
    ARTIFACT_BUILDING,
    ARTIFACT_JOB_FK,
    ApiKey,
    Base,
    CIRCULAR_FOREIGN_KEYS,
    CONTROL_PLANE_TABLES,
    GraphArtifact,
    IngestJob,
    JOB_ARTIFACT_FK,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_IN_FLIGHT,
    JOB_QUEUED,
    JOB_REGISTERED,
    JOB_SWAPPED,
    LOAD_PULLING,
    ORGANIZATION_ARTIFACT_FK,
    Organization,
    POD_BOOTING,
    Pod,
    PodAssignment,
    Repository,
    create_control_plane_schema,
)
from src.models.database import (
    CONTROL_PLANE_URL_VARIABLE,
    DATABASE_URL_VARIABLE,
    ControlPlaneNotConfigured,
    control_plane_sessions,
    control_plane_url,
    create_control_plane_engine,
)

NOW = 1_700_000_000


@pytest.fixture()
def engine(tmp_path):
    """A real database with the schema up, built the way a process builds it."""
    made = create_control_plane_engine(tmp_path / "control-plane.db")
    create_control_plane_schema(made)
    try:
        yield made
    finally:
        made.dispose()


@pytest.fixture()
def sessions(engine):
    return control_plane_sessions(engine)


@pytest.fixture()
def session(sessions):
    with sessions() as open_session:
        yield open_session


def _organization(session, org_id="org_1"):
    session.add(
        Organization(
            org_id=org_id,
            name="Acme",
            plan="team",
            status="active",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    session.commit()
    return org_id


def _job(session, job_id, org_id="org_1", *, status=JOB_QUEUED):
    session.add(
        IngestJob(
            job_id=job_id,
            org_id=org_id,
            trigger="manual",
            status=status,
            cursor_from="c0",
            cursor_to="c9",
            queued_at=NOW,
        )
    )
    session.commit()
    return job_id


def _provision(plane, *org_ids):
    """Create the tenants a credential is about to name.

    A credential refers to its organisation, so the row comes first — the same
    order a deployment provisions in.
    """
    with control_plane_sessions(plane.engine)() as session:
        for org_id in org_ids:
            session.add(
                Organization(
                    org_id=org_id,
                    name=org_id,
                    plan="team",
                    status="active",
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        session.commit()
    return plane


def _pod(session, pod_id="pod_1", address="10.0.0.4:8000"):
    session.add(Pod(pod_id=pod_id, address=address, created_at=NOW))
    session.commit()
    return pod_id


# --------------------------------------------------------------------------
# the schema call
# --------------------------------------------------------------------------


def test_the_schema_call_is_idempotent(tmp_path):
    """Running it twice on a database that already has it changes nothing.

    The second run must not raise and must not disturb what is there — a row
    written between the two survives. This is the whole of the migration story
    here: one create that is safe to run at every start.
    """
    made = create_control_plane_engine(tmp_path / "twice.db")
    try:
        create_control_plane_schema(made)
        with control_plane_sessions(made)() as writing:
            _organization(writing, "org_1")

        before = set(inspect(made).get_table_names())

        create_control_plane_schema(made)

        assert set(inspect(made).get_table_names()) == before
        with control_plane_sessions(made)() as reading:
            assert reading.scalar(select(Organization.org_id)) == "org_1"
    finally:
        made.dispose()


def test_the_schema_call_creates_only_the_control_plane_tables(tmp_path):
    """A model from another concern must not get a table in this database.

    The schema call names its tables rather than creating everything the ORM
    has been told about. Without that, a module imported for an unrelated
    reason puts its tables here — and the leak is invisible until something
    reads a table it did not expect to find.
    """

    # Declared the way every model here is declared, so it lands on the same
    # shared metadata — which is exactly the case that bites: a second
    # concern's model imported into this process for an unrelated reason.
    class AnotherConcern(SQLModel, table=True):
        __tablename__ = "another_concern"

        id: int = Field(primary_key=True)
        note: str

    made = create_control_plane_engine(tmp_path / "only.db")
    try:
        create_control_plane_schema(made)
        created = set(inspect(made).get_table_names())
    finally:
        made.dispose()
        SQLModel.metadata.remove(AnotherConcern.__table__)

    assert created == {table.name for table in CONTROL_PLANE_TABLES}
    assert "another_concern" not in created


def test_every_table_exists(engine):
    assert set(inspect(engine).get_table_names()) == {
        "apikey",
        "organization",
        "repository",
        "graphartifact",
        "ingestjob",
        "pod",
        "podassignment",
    }


def test_the_credential_table_is_unaffected(engine):
    """The existing table keeps its columns.

    Named columns rather than a count, because a column added to the end would
    pass a count and break every reader that names them.
    """
    columns = [column["name"] for column in inspect(engine).get_columns("apikey")]

    assert columns == [
        "key_id",
        "hashed_key",
        "org_id",
        "scopes",
        "prefix",
        "created_at",
        "revoked_at",
    ]


def test_a_credential_still_round_trips(tmp_path):
    """Issue, look up, revoke — the behaviour the request path depends on."""
    plane = _provision(open_control_plane(tmp_path / "credentials.db"), "org_1")
    try:
        raw, record = plane.issue("org_1")

        found = plane.record_for_hash(hash_api_key(raw))
        assert found is not None
        assert found.org_id == "org_1"
        assert found.key_id == record.key_id
        assert not found.is_revoked

        assert plane.revoke(record.key_id) is True
        assert plane.record_for_hash(hash_api_key(raw)).is_revoked
        # A second revocation changes nothing and says so.
        assert plane.revoke(record.key_id) is False
    finally:
        plane.close()


# --------------------------------------------------------------------------
# the declared references
# --------------------------------------------------------------------------


def test_every_reference_between_these_tables_is_declared(engine):
    """No table refers to another by an undeclared string.

    Every pair, the credential table's included: a key naming a tenant that
    does not exist is not a credential, it is a row nothing can resolve.
    """
    inspector = inspect(engine)
    declared = {
        (table, key["constrained_columns"][0], key["referred_table"])
        for table in inspector.get_table_names()
        for key in inspector.get_foreign_keys(table)
    }

    assert declared == {
        ("apikey", "org_id", "organization"),
        ("organization", "desired_artifact_id", "graphartifact"),
        ("repository", "org_id", "organization"),
        ("graphartifact", "org_id", "organization"),
        ("graphartifact", "built_by_job_id", "ingestjob"),
        ("ingestjob", "org_id", "organization"),
        ("ingestjob", "repo_id", "repository"),
        ("ingestjob", "produced_artifact_id", "graphartifact"),
        ("podassignment", "pod_id", "pod"),
        ("podassignment", "org_id", "organization"),
        ("podassignment", "artifact_id", "graphartifact"),
    }


def test_a_credential_cannot_name_a_tenant_that_does_not_exist(tmp_path):
    """Issuing against an unknown organisation fails, at the database.

    The refusal has to come from the store. A check inside the issuing
    function is skipped by anything that writes the row another way, and what
    it would let through is a working credential for a tenant nobody can find.
    """
    from src.control_plane import ControlPlaneError

    plane = open_control_plane(tmp_path / "unknown.db")
    try:
        with pytest.raises(ControlPlaneError):
            plane.issue("org_that_never_existed")

        _provision(plane, "org_1")
        raw, record = plane.issue("org_1")

        assert record.org_id == "org_1"
        assert plane.record_for_hash(hash_api_key(raw)) is not None
    finally:
        plane.close()


def test_every_declared_reference_is_indexed(engine):
    """A join or a lookup on a reference must not scan.

    The reconcile pass joins assignments to organisations and looks up
    artifacts by id; unindexed, each of those is a table scan that gets slower
    as the deployment gets bigger.
    """
    inspector = inspect(engine)

    for table in inspector.get_table_names():
        keys = inspector.get_foreign_keys(table)
        if not keys:
            continue

        indexed = {
            tuple(index["column_names"])[0]
            for index in inspector.get_indexes(table)
            if index["column_names"]
        }
        primary = set(inspector.get_pk_constraint(table)["constrained_columns"])

        for key in keys:
            column = key["constrained_columns"][0]
            # A primary-key column is already indexed by being one.
            assert column in indexed or column in primary, f"{table}.{column}"


def test_the_three_circular_references_exist_after_creation(engine):
    """The cycle is declared, not dropped to make creation work.

    An organisation names the artifact it intends to serve, an artifact names
    the build that produced it, and that build names the artifact it produced.
    All three survive schema creation, under the names this project chose.
    """
    inspector = inspect(engine)
    found = {
        key["name"]
        for table in ("organization", "graphartifact", "ingestjob")
        for key in inspector.get_foreign_keys(table)
    }

    assert set(CIRCULAR_FOREIGN_KEYS) <= found


def test_the_three_circular_references_are_added_after_the_tables():
    """On a server dialect they arrive as their own statements.

    This is what makes the cycle creatable at all: none of the three is part
    of either table's creation, so no ordering problem exists. Checked by
    compiling the tables for the dialect the deployment runs on — the tables
    themselves must not carry these constraints.
    """
    from sqlalchemy.dialects import postgresql

    dialect = postgresql.dialect()
    rendered = {
        table.name: str(CreateTable(table).compile(dialect=dialect))
        for table in CONTROL_PLANE_TABLES
    }

    for name in CIRCULAR_FOREIGN_KEYS:
        for statement in rendered.values():
            assert name not in statement, f"{name} is part of a table creation"

    # The ordinary references are part of their table, which is what makes
    # them the ordinary case.
    assert "FOREIGN KEY(org_id) REFERENCES organization" in rendered["repository"]


def test_an_assignment_for_an_unknown_organisation_is_rejected(session):
    """The database refuses it, so a join can never see it.

    Undeclared, this row is not an error — it is a row that silently drops out
    of the reconcile pass's join, and a pass that then acts on the result.
    """
    _pod(session, "pod_1")

    session.add(
        PodAssignment(pod_id="pod_1", org_id="org_that_never_existed", assigned_at=NOW)
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_an_assignment_for_an_unknown_pod_is_rejected(session):
    _organization(session, "org_1")

    session.add(
        PodAssignment(pod_id="pod_that_never_existed", org_id="org_1", assigned_at=NOW)
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_an_artifact_for_an_unknown_organisation_is_rejected(session):
    session.add(
        GraphArtifact(
            artifact_id="art_1",
            org_id="org_that_never_existed",
            version=1,
            s3_uri="local://artifacts/ghost/v1.lbug",
            created_at=NOW,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_the_cycle_is_written_in_order(session):
    """Organisation, artifact and build, each naming the next.

    There is no simultaneous write here and none is needed. All three of the
    circular columns are nullable, so the cycle is closed by writing in an
    order that exists: the build first, then the artifact that names it, then
    the organisation pointed at that artifact. Every reference is checked as
    its statement runs, which is what the schema declares — the three are
    ordered at *creation* time only.
    """
    _organization(session, "org_1")
    _job(session, "job_1")

    session.add(
        GraphArtifact(
            artifact_id="art_1",
            org_id="org_1",
            version=1,
            s3_uri="local://artifacts/org_1/v1.lbug",
            checksum_sha256="abc",
            size_bytes=10,
            entity_count=16,
            built_by_job_id="job_1",
            created_at=NOW,
        )
    )
    session.flush()

    session.get(IngestJob, "job_1").produced_artifact_id = "art_1"
    session.get(Organization, "org_1").desired_artifact_id = "art_1"
    session.commit()

    organization = session.get(Organization, "org_1")
    artifact = session.get(GraphArtifact, "art_1")
    job = session.get(IngestJob, "job_1")

    assert organization.desired_artifact_id == "art_1"
    assert artifact.built_by_job_id == "job_1"
    assert job.produced_artifact_id == "art_1"


# --------------------------------------------------------------------------
# each table takes a well-formed row
# --------------------------------------------------------------------------


def test_an_organization_row_round_trips(session):
    _organization(session, "org_1")
    row = session.get(Organization, "org_1")

    assert row.name == "Acme"
    assert row.plan == "team"
    # The intent pointer starts empty. A tenant with nothing built for it is a
    # tenant with nothing to serve, not a tenant pointing at nothing.
    assert row.desired_artifact_id is None


def test_a_repository_row_round_trips(session):
    _organization(session, "org_1")
    session.add(
        Repository(
            repo_id="repo_1",
            org_id="org_1",
            provider="github",
            provider_repo_id="12345",
            name="acme/widgets",
            default_branch="main",
            last_synced_cursor="cur",
            last_synced_at=NOW,
            status="active",
            created_at=NOW,
        )
    )
    session.commit()

    row = session.get(Repository, "repo_1")
    assert row.provider_repo_id == "12345"
    assert row.default_branch == "main"


def test_an_artifact_row_defaults_to_building(session):
    _organization(session, "org_1")
    session.add(
        GraphArtifact(
            artifact_id="art_1",
            org_id="org_1",
            version=1,
            s3_uri="local://artifacts/org_1/v1.lbug",
            created_at=NOW,
        )
    )
    session.commit()
    session.expire_all()

    row = session.get(GraphArtifact, "art_1")
    assert row.status == ARTIFACT_BUILDING
    assert row.built_by_job_id is None


def test_a_job_row_defaults_to_queued(session):
    _organization(session, "org_1")
    session.add(
        IngestJob(job_id="job_1", org_id="org_1", trigger="webhook", queued_at=NOW)
    )
    session.commit()
    session.expire_all()

    row = session.get(IngestJob, "job_1")
    assert row.status == JOB_QUEUED
    assert row.started_at is None
    assert row.finished_at is None


def test_a_pod_row_defaults_to_booting(session):
    _pod(session, "pod_1")
    session.expire_all()

    row = session.get(Pod, "pod_1")
    assert row.status == POD_BOOTING
    assert row.memory_budget_mb is None


def test_an_assignment_row_defaults_to_pulling(session):
    _organization(session, "org_1")
    _pod(session, "pod_1")
    session.add(PodAssignment(pod_id="pod_1", org_id="org_1", assigned_at=NOW))
    session.commit()
    session.expire_all()

    row = session.get(PodAssignment, ("pod_1", "org_1"))
    assert row.load_status == LOAD_PULLING
    assert row.confirmed_at is None


def test_the_same_tenant_on_the_same_pod_is_one_row(session):
    """The composite key is the identity, so the pair cannot repeat."""
    _organization(session, "org_1")
    _pod(session, "pod_1")
    session.add(PodAssignment(pod_id="pod_1", org_id="org_1", assigned_at=NOW))
    session.commit()

    with pytest.raises(IntegrityError):
        session.execute(
            PodAssignment.__table__.insert().values(
                pod_id="pod_1", org_id="org_1", assigned_at=NOW + 1
            )
        )
    session.rollback()

    # The same tenant on a different pod is a different fact and is allowed.
    _pod(session, "pod_2", "10.0.0.5:8000")
    session.add(PodAssignment(pod_id="pod_2", org_id="org_1", assigned_at=NOW))
    session.commit()


# --------------------------------------------------------------------------
# the uniqueness rules
# --------------------------------------------------------------------------


def test_a_repository_cannot_be_registered_twice_for_one_tenant(session):
    _organization(session, "org_1")
    _organization(session, "org_2")

    def register(org_id, repo_id):
        session.add(
            Repository(
                repo_id=repo_id,
                org_id=org_id,
                provider="github",
                provider_repo_id="12345",
                name="acme/widgets",
                status="active",
                created_at=NOW,
            )
        )
        session.commit()

    register("org_1", "repo_1")

    with pytest.raises(IntegrityError):
        register("org_1", "repo_2")
    session.rollback()

    # The same upstream repository under a different tenant is a different
    # registration, with its own token and its own sync cursor.
    register("org_2", "repo_3")

    assert session.scalar(select(Repository).where(Repository.repo_id == "repo_3"))
    assert len(session.scalars(select(Repository)).all()) == 2


def test_a_version_number_means_one_artifact_per_tenant(session):
    _organization(session, "org_1")
    _organization(session, "org_2")

    def store(artifact_id, org_id, version):
        session.add(
            GraphArtifact(
                artifact_id=artifact_id,
                org_id=org_id,
                version=version,
                s3_uri=f"local://artifacts/{org_id}/v{version}.lbug",
                created_at=NOW,
            )
        )
        session.commit()

    store("art_1", "org_1", 1)

    with pytest.raises(IntegrityError):
        store("art_2", "org_1", 1)
    session.rollback()

    # Version numbers are per tenant, so another tenant's version 1 is fine.
    store("art_3", "org_2", 1)
    store("art_4", "org_1", 2)

    assert len(session.scalars(select(GraphArtifact)).all()) == 3


# --------------------------------------------------------------------------
# one build at a time
# --------------------------------------------------------------------------


def test_a_second_in_flight_job_is_rejected_by_the_database(session):
    """The rule the partial index exists for.

    Two workers reading this table at the same moment both see no build
    running. One of them has to be told no by something that saw both, and
    that something is the database.
    """
    _organization(session, "org_1")
    _job(session, "job_1")

    with pytest.raises(IntegrityError):
        _job(session, "job_2")
    session.rollback()

    assert len(session.scalars(select(IngestJob)).all()) == 1


def test_a_finished_job_frees_the_tenant(session):
    """A terminal status leaves the index, so the next build may start."""
    _organization(session, "org_1")
    _job(session, "job_1")

    job = session.get(IngestJob, "job_1")
    job.status = JOB_COMPLETED
    job.finished_at = NOW + 60
    session.commit()

    _job(session, "job_2")
    assert len(session.scalars(select(IngestJob)).all()) == 2


@pytest.mark.parametrize("terminal", [JOB_SWAPPED, JOB_COMPLETED, JOB_FAILED])
def test_every_terminal_status_frees_the_tenant(session, terminal):
    _organization(session, "org_1")
    _job(session, "job_1")

    session.get(IngestJob, "job_1").status = terminal
    session.commit()

    _job(session, "job_2")


@pytest.mark.parametrize("status", list(JOB_IN_FLIGHT))
def test_every_in_flight_status_holds_the_tenant(session, status):
    """Each member of the group blocks, not just the one a build starts in.

    Parameterised over ``JOB_IN_FLIGHT`` itself, so adding a status to the
    group without adding it to the index fails here rather than at the moment
    two builds run.
    """
    _organization(session, "org_1")
    _job(session, "job_1", status=status)

    with pytest.raises(IntegrityError):
        _job(session, "job_2")
    session.rollback()


def test_two_tenants_build_at_the_same_time(session):
    """The rule is per tenant. One busy tenant must not stop another."""
    _organization(session, "org_1")
    _organization(session, "org_2")

    _job(session, "job_1", "org_1")
    _job(session, "job_2", "org_2", status=JOB_REGISTERED)

    assert len(session.scalars(select(IngestJob)).all()) == 2


def test_the_index_is_declared_for_both_dialects_over_exactly_the_group():
    """The condition lists the in-flight group and nothing else, either side.

    Generated from ``JOB_IN_FLIGHT`` so the two cannot drift. Compiled for
    both dialects the project runs on, because an index declared for only one
    of them silently does nothing on the other — and the one it would do
    nothing on is the one that runs in production.
    """
    from sqlalchemy.dialects import postgresql, sqlite
    from sqlalchemy.schema import CreateIndex

    index = next(
        candidate
        for candidate in IngestJob.__table__.indexes
        if candidate.name == "ingestjob_one_in_flight"
    )

    for dialect in (postgresql.dialect(), sqlite.dialect()):
        statement = str(CreateIndex(index).compile(dialect=dialect))

        assert "UNIQUE INDEX" in statement
        assert "WHERE" in statement
        for status in JOB_IN_FLIGHT:
            assert f"'{status}'" in statement
        for status in (JOB_SWAPPED, JOB_COMPLETED, JOB_FAILED):
            assert f"'{status}'" not in statement


# --------------------------------------------------------------------------
# the credential column nothing may write
# --------------------------------------------------------------------------


def test_the_github_token_column_exists_and_starts_empty(session, engine):
    columns = {column["name"] for column in inspect(engine).get_columns("repository")}
    assert "github_token" in columns

    _organization(session, "org_1")
    session.add(
        Repository(
            repo_id="repo_1",
            org_id="org_1",
            provider="github",
            provider_repo_id="12345",
            name="acme/widgets",
            status="active",
            created_at=NOW,
        )
    )
    session.commit()
    session.expire_all()

    assert session.get(Repository, "repo_1").github_token is None


def test_repository_never_persists_a_plaintext_github_token(session, monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv(
        "GRAPHRAG_ENCRYPTION_MASTER_KEY", Fernet.generate_key().decode()
    )
    _organization(session, "org_1")
    repository = Repository(
        repo_id="repo_1",
        org_id="org_1",
        provider="github",
        provider_repo_id="12345",
        name="acme/widgets",
        status="active",
        created_at=NOW,
    )
    repository.set_github_token("plain-token")
    session.add(repository)
    session.commit()
    session.expire_all()

    stored = session.get(Repository, "repo_1")
    assert stored.github_token != "plain-token"
    assert stored.get_github_token() == "plain-token"


# --------------------------------------------------------------------------
# where the database is
# --------------------------------------------------------------------------


def test_the_connection_factory_fails_when_neither_variable_is_set():
    """Unset is an error naming both, not a quiet local file.

    A fallback would work perfectly on the machine that wrote it and would, in
    the place this runs, give every process its own private control plane with
    no error anywhere.
    """
    with pytest.raises(ControlPlaneNotConfigured) as failure:
        control_plane_url({})

    message = str(failure.value)
    assert CONTROL_PLANE_URL_VARIABLE in message
    assert DATABASE_URL_VARIABLE in message


def test_an_empty_variable_counts_as_unset():
    """A variable set to whitespace is a variable nobody set on purpose."""
    with pytest.raises(ControlPlaneNotConfigured):
        control_plane_url({CONTROL_PLANE_URL_VARIABLE: "   ", DATABASE_URL_VARIABLE: ""})


def test_the_general_database_variable_is_the_fallback():
    """One variable is enough for a single-database setup."""
    assert (
        control_plane_url({DATABASE_URL_VARIABLE: "postgresql+psycopg://h/app"})
        == "postgresql+psycopg://h/app"
    )


def test_the_specific_variable_wins():
    """A deployment that has split the two cannot be pulled back together."""
    assert (
        control_plane_url(
            {
                CONTROL_PLANE_URL_VARIABLE: "postgresql+psycopg://h/control",
                DATABASE_URL_VARIABLE: "postgresql+psycopg://h/app",
            }
        )
        == "postgresql+psycopg://h/control"
    )


def test_connections_are_health_checked(tmp_path):
    """A connection idle past the server's timeout is replaced, not returned.

    Without this the process learns about a dead connection as a failed query
    on the request path.
    """
    made = create_control_plane_engine(tmp_path / "pool.db")
    try:
        assert made.pool._pre_ping is True
    finally:
        made.dispose()


def test_a_credential_lookup_needs_no_argument_but_needs_a_variable(monkeypatch):
    """A process nobody told about a database raises rather than inventing one."""
    from src.control_plane import ControlPlane, ControlPlaneError

    monkeypatch.delenv(CONTROL_PLANE_URL_VARIABLE, raising=False)
    monkeypatch.delenv(DATABASE_URL_VARIABLE, raising=False)

    with pytest.raises(ControlPlaneError):
        ControlPlane()


def test_the_key_itself_is_never_stored(tmp_path):
    """Only the digest reaches the database, checked against the file itself."""
    path = tmp_path / "digest.db"
    plane = _provision(open_control_plane(path), "org_1")
    try:
        raw, _ = plane.issue("org_1")
    finally:
        plane.close()

    contents = path.read_bytes()
    assert raw.encode("utf-8") not in contents
    assert hash_api_key(raw).encode("utf-8") in contents


def test_a_stored_credential_is_readable_as_a_model(tmp_path):
    """The credential table is a model like the others, not a special case."""
    path = tmp_path / "model.db"
    plane = _provision(open_control_plane(path), "org_1")
    try:
        raw, record = plane.issue("org_1", scopes="read")
        with control_plane_sessions(plane.engine)() as reading:
            stored = reading.get(ApiKey, record.key_id)

            assert stored.org_id == "org_1"
            assert stored.hashed_key == hash_api_key(raw)
            assert stored.revoked_at is None
    finally:
        plane.close()
