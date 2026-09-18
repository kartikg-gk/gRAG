"""The control plane's tables, as models.

What is here
------------

Who the tenants are, what they have registered, what has been built for them,
which process is serving it, and the credentials that decide which tenant a
request belongs to. Six tables plus the credential one, in one database that
every process shares.

Why models rather than SQL strings
----------------------------------

Everything downstream of these tables does the same four things: fetch a row
by its key, join two tables, change a field, commit. Written as SQL each
consumer carries its own query and its own row-to-object conversion, and they
drift — a column added for one of them is read by none of the others, and a
mistyped name is a runtime error in whichever consumer is least often run.
Declared once here, they share both.

The definitions are the schema. There is no second place where a column's type
or default is written down, so the two cannot disagree.

Table names use SQLModel's class-name defaults. They are part of the control-
plane contract, so foreign keys below name those exact generated tables.

The three references that point in a circle
-------------------------------------------

An organisation names the artifact it intends to serve, an artifact names the
build that produced it, and a build names the artifact it produced. Declared
plainly, no creation order satisfies all three: whichever table is created
first refers to one that does not exist yet.

Each of the three is therefore emitted as its own statement once both tables
exist, rather than as part of either table's creation, and each is given an
explicit name so a later change can refer to it instead of guessing what the
database called it. That is a rule about the order the *schema* is built in
and says nothing about when a row is checked — the three columns are nullable,
and a writer creates the cycle by writing in an order that exists: the build,
then the artifact naming it, then the organisation pointed at the artifact.

The other references are ordinary, and every one of them is indexed. The
reason is the pass that reconciles intent against reality: it joins
assignments to organisations and looks up artifacts by id. Undeclared, an
assignment pointing at an organisation that no longer exists is not an error —
it is a row that silently drops out of a join, and a pass that acts on the
result.

Statuses
--------

Plain strings, grouped by what they describe. Every one crosses into the
database as text and comes back as text, so a type that had to be converted in
both directions would buy nothing the grouping and the names do not already
give.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    Column,
    Engine,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlmodel import Field, SQLModel

#: The metadata these models share, and the thing the schema call is
#: deliberately *not* asked to create in full: it is every model imported into
#: the process, and another concern's tables must not appear in this database
#: as a side effect of an import.
Base = SQLModel


# --------------------------------------------------------------------------
# Status vocabularies
# --------------------------------------------------------------------------

#: An artifact's life. It is built, becomes servable, is put in front of
#: traffic, is replaced by a newer version, or never arrives.
ARTIFACT_BUILDING = "building"
ARTIFACT_READY = "ready"
ARTIFACT_ACTIVE = "active"
ARTIFACT_SUPERSEDED = "superseded"
ARTIFACT_FAILED = "failed"

#: A build's life, in the order it passes through.
JOB_QUEUED = "queued"
JOB_FETCHING = "fetching"
JOB_COMPUTING = "computing"
JOB_COMPILING = "compiling"
JOB_UPLOADING = "uploading"
JOB_REGISTERED = "registered"
JOB_SWAPPED = "swapped"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"

#: The statuses that mean a build is still happening -- queued through
#: registered.
#:
#: Named once because two things need exactly this set: the rule that one
#: organisation gets one build at a time, and any caller asking whether a
#: build is still running. Writing the list out in both places is how the two
#: come to disagree, and the disagreement would show up as a second build
#: starting rather than as an error.
JOB_IN_FLIGHT = (
    JOB_QUEUED,
    JOB_FETCHING,
    JOB_COMPUTING,
    JOB_COMPILING,
    JOB_UPLOADING,
    JOB_REGISTERED,
)

#: A serving process's life.
POD_BOOTING = "booting"
POD_READY = "ready"
POD_DRAINING = "draining"
POD_DEAD = "dead"

#: What one process has done with one tenant's artifact.
LOAD_PULLING = "pulling"
LOAD_READY = "ready"
LOAD_FAILED = "failed"

#: What a credential is allowed to do. Every key issued today is full-access,
#: and nothing reads this yet -- enforcement is separate work. The field is
#: here because the alternative is discovering at enforcement time that no
#: existing key records what it was issued for.
DEFAULT_SCOPES = "read"

#: The in-flight set as a SQL list, built from the tuple above rather than
#: written out again, for the index that enforces one build at a time.
_IN_FLIGHT_SQL = ", ".join(f"'{status}'" for status in JOB_IN_FLIGHT)

#: The names of the three constraints that close the cycle. Explicit, so a
#: later change refers to a name this project chose rather than to whatever
#: the database happened to generate.
ORGANIZATION_ARTIFACT_FK = "fk_organization_desired_artifact"
ARTIFACT_JOB_FK = "fk_graphartifact_job"
JOB_ARTIFACT_FK = "fk_ingestjob_artifact"

#: The three, as one group. Tested against what the schema actually created.
CIRCULAR_FOREIGN_KEYS = (
    ORGANIZATION_ARTIFACT_FK,
    ARTIFACT_JOB_FK,
    JOB_ARTIFACT_FK,
)


# --------------------------------------------------------------------------
# the tables
# --------------------------------------------------------------------------


class ApiKey(SQLModel, table=True):
    """One stored credential. Carries no secret.

    ``org_id`` names the organisation this key belongs to and refers to it.
    The reference costs the request path nothing — the lookup still reads one
    row by digest — and it buys the one thing the column could not otherwise
    have: a credential cannot name a tenant that does not exist. A key issued
    against a typo used to be a working credential for an organisation nobody
    could find.
    """

    key_id: str = Field(primary_key=True)
    #: The SHA-256 digest of the key. **Never the key itself**, so a copy of
    #: this database does not let its holder authenticate as anyone.
    hashed_key: str = Field(unique=True, index=True)
    org_id: str = Field(foreign_key="organization.org_id", index=True)
    scopes: str = Field(
        default=DEFAULT_SCOPES,
        sa_column=Column(String, nullable=False, server_default=DEFAULT_SCOPES),
    )
    #: The first few characters of the raw key: enough to tell two keys apart
    #: in a list, far too few to be a useful fragment of the secret.
    prefix: Optional[str] = Field(default=None)
    created_at: int
    #: Set rather than deleted. A deleted key and a key that never existed are
    #: indistinguishable afterwards, and "was this revoked, and when" is
    #: exactly the question asked after an incident.
    revoked_at: Optional[int] = Field(default=None)


class Organization(SQLModel, table=True):
    """A tenant, and the artifact it is supposed to be serving."""

    org_id: str = Field(primary_key=True)
    name: str
    plan: str
    status: str
    #: Intent, not fact: something sets this, and something else is answerable
    #: for making a process actually hold it. What is really loaded lives on
    #: the assignment, and the two being equal is the definition of a tenant
    #: being served the right graph.
    #:
    #: One of the three that close the cycle, so the reference is spelled out
    #: here rather than as a string: this is where the name it is created
    #: under, and the instruction to add it after both tables exist, live.
    desired_artifact_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String,
            ForeignKey(
                "graphartifact.artifact_id",
                use_alter=True,
                name=ORGANIZATION_ARTIFACT_FK,
            ),
            nullable=True,
            index=True,
        ),
    )
    created_at: int
    updated_at: int


class Repository(SQLModel, table=True):
    """An upstream repository a tenant has registered for ingestion."""

    __table_args__ = (
        # One provider repository registers once per tenant. Two rows for the
        # same repo would each carry their own sync cursor, and ingestion
        # would run twice over the same commits.
        UniqueConstraint(
            "org_id",
            "provider",
            "provider_repo_id",
            name="repository_one_per_provider_repo",
        ),
    )

    repo_id: str = Field(primary_key=True)
    org_id: str = Field(foreign_key="organization.org_id", index=True)
    provider: str
    provider_repo_id: str
    name: str
    default_branch: Optional[str] = Field(default=None)
    last_synced_cursor: Optional[str] = Field(default=None)
    last_synced_at: Optional[int] = Field(default=None)
    status: str
    #: Fernet ciphertext only. Plaintext is accepted and returned exclusively
    #: through the two helpers below and never assigned to this field.
    github_token: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: int

    def set_github_token(self, plain_token: Optional[str]) -> None:
        if not plain_token:
            self.github_token = None
            return

        from ..vault import encrypt

        self.github_token = encrypt(plain_token)

    def get_github_token(self) -> Optional[str]:
        if not self.github_token:
            return None
        from ..vault import decrypt

        return decrypt(self.github_token)


class GraphArtifact(SQLModel, table=True):
    """One built graph, addressable and checksummed."""

    __table_args__ = (
        # A version number names one artifact for one tenant. Without this,
        # two builds racing both call themselves version 4, and a reader
        # asking for version 4 gets whichever one it happens to see.
        UniqueConstraint("org_id", "version", name="graphartifact_one_per_version"),
    )

    artifact_id: str = Field(primary_key=True)
    org_id: str = Field(foreign_key="organization.org_id", index=True)
    version: int
    s3_uri: str
    checksum_sha256: Optional[str] = Field(default=None)
    size_bytes: Optional[int] = Field(default=None)
    entity_count: Optional[int] = Field(default=None)
    status: str = Field(
        default=ARTIFACT_BUILDING,
        sa_column=Column(String, nullable=False, server_default=ARTIFACT_BUILDING),
    )
    #: The build that produced this. Nullable: an artifact can be put in place
    #: by hand, and the first ones were — and being nullable is also what lets
    #: the cycle be written in an order that exists.
    #:
    #: The second of the three that close the cycle.
    built_by_job_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String,
            ForeignKey("ingestjob.job_id", use_alter=True, name=ARTIFACT_JOB_FK),
            nullable=True,
            index=True,
        ),
    )
    created_at: int


class IngestJob(SQLModel, table=True):
    """One build, from queued to whatever became of it."""

    __table_args__ = (
        # One build at a time per tenant, decided by the database.
        #
        # A unique index over the organisation covering only rows that are
        # still in flight: a finished build leaves the index, so the next one
        # is free to start. The check has to live here rather than in a
        # worker, because the case it exists for is two workers reading the
        # same table at the same moment and both seeing no build running. One
        # of them has to be told no by something that saw both.
        #
        # Declared for both dialects the project runs on, from the one status
        # group above rather than a list written out again.
        Index(
            "ingestjob_one_in_flight",
            "org_id",
            unique=True,
            postgresql_where=text(f"status IN ({_IN_FLIGHT_SQL})"),
            sqlite_where=text(f"status IN ({_IN_FLIGHT_SQL})"),
        ),
    )

    job_id: str = Field(primary_key=True)
    org_id: str = Field(foreign_key="organization.org_id", index=True)
    repo_id: Optional[str] = Field(
        default=None, foreign_key="repository.repo_id", index=True
    )
    trigger: str
    status: str = Field(
        default=JOB_QUEUED,
        sa_column=Column(String, nullable=False, server_default=JOB_QUEUED, index=True),
    )
    cursor_from: Optional[str] = Field(default=None)
    cursor_to: Optional[str] = Field(default=None)
    #: What this build produced. The third of the three that close the cycle,
    #: and nullable for the same reason: a build that has not finished has not
    #: produced anything yet, which is what makes the ordered write possible.
    produced_artifact_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String,
            ForeignKey(
                "graphartifact.artifact_id", use_alter=True, name=JOB_ARTIFACT_FK
            ),
            nullable=True,
            index=True,
        ),
    )
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    queued_at: int
    started_at: Optional[int] = Field(default=None)
    finished_at: Optional[int] = Field(default=None)


class Pod(SQLModel, table=True):
    """One serving process, and when it was last heard from."""

    pod_id: str = Field(primary_key=True)
    address: str
    status: str = Field(
        default=POD_BOOTING,
        sa_column=Column(String, nullable=False, server_default=POD_BOOTING),
    )
    memory_budget_mb: Optional[int] = Field(default=None)
    last_heartbeat_at: Optional[int] = Field(default=None)
    created_at: int


class PodAssignment(SQLModel, table=True):
    """A tenant on a pod, and what that pod actually has loaded.

    The pair is the identity: assigning the same tenant to the same pod twice
    is the same fact, not a second one.
    """

    pod_id: str = Field(foreign_key="pod.pod_id", primary_key=True, index=True)
    org_id: str = Field(
        foreign_key="organization.org_id", primary_key=True, index=True
    )
    #: Reality, against the organisation's ``desired_artifact_id``, which is
    #: intent. The two are apart so the question can be asked.
    artifact_id: Optional[str] = Field(
        default=None, foreign_key="graphartifact.artifact_id", index=True
    )
    load_status: str = Field(
        default=LOAD_PULLING,
        sa_column=Column(String, nullable=False, server_default=LOAD_PULLING),
    )
    assigned_at: int
    #: When the loaded state above was last confirmed, not when it was first
    #: set. A row nothing has confirmed recently is a row to distrust.
    confirmed_at: Optional[int] = Field(default=None)


#: Exactly the tables this database holds, in creation order.
#:
#: Named rather than taken from the metadata, because the metadata is every
#: model imported into the process. Another concern's models will share this
#: process and must not have their tables appear here as a side effect of an
#: import.
CONTROL_PLANE_MODELS = (
    ApiKey,
    Organization,
    Repository,
    GraphArtifact,
    IngestJob,
    Pod,
    PodAssignment,
)

CONTROL_PLANE_TABLES = tuple(model.__table__ for model in CONTROL_PLANE_MODELS)


def create_control_plane_schema(engine: Engine) -> None:
    """Bring the control plane's tables up. Safe to call on every start.

    One idempotent create, and the whole of the migration story here: existing
    tables are left alone, and the three references that close the cycle are
    added by their own statements once both ends exist.
    """
    SQLModel.metadata.create_all(
        engine, tables=list(CONTROL_PLANE_TABLES), checkfirst=True
    )
