"""Normalized domain models for the GitHub entities Module 2 ingests.

Field names mirror GitHub's JSON keys, so no aliases are needed and a payload
maps onto a model without a translation table. Nested objects stay nested:
``pr.user.login`` and ``commit.commit.message`` read the way the API reads,
which keeps normalization honest and the models trivial to eyeball against
GitHub's docs.

Every model ignores unknown fields, so GitHub adding keys never breaks a build.
Timestamps parse into timezone-aware ``datetime`` objects.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class _GitHubModel(BaseModel):
    """Shared config: tolerate unknown keys, reject nothing else."""

    model_config = ConfigDict(extra="ignore")


class GitHubUser(_GitHubModel):
    """An account GitHub resolved for an action."""

    login: str
    id: int


class Label(_GitHubModel):
    """A label attached to an issue or pull request."""

    name: str


class GitActor(_GitHubModel):
    """Git's own author/committer record, which has no GitHub account attached."""

    name: str | None = None
    email: str | None = None
    date: datetime | None = None


class CommitDetail(_GitHubModel):
    """The ``commit`` sub-object: what git recorded, not what GitHub inferred."""

    message: str
    author: GitActor | None = None


class CommitParent(_GitHubModel):
    """A parent pointer; more than one means a merge commit."""

    sha: str


class Repository(_GitHubModel):
    id: int
    name: str
    full_name: str
    private: bool
    owner: GitHubUser
    html_url: str
    description: str | None = None
    language: str | None = None
    default_branch: str
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime | None = None


class PullRequest(_GitHubModel):
    id: int
    number: int
    state: str
    title: str
    body: str | None = None
    user: GitHubUser
    html_url: str
    draft: bool = False
    merge_commit_sha: str | None = None
    labels: list[Label] = []
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    merged_at: datetime | None = None


class Issue(_GitHubModel):
    id: int
    number: int
    state: str
    title: str
    body: str | None = None
    user: GitHubUser
    html_url: str
    labels: list[Label] = []
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None


class Commit(_GitHubModel):
    """A commit as the list endpoint returns it.

    ``author`` is the GitHub account, and it is ``None`` when GitHub cannot
    match the git email to one. ``commit.author`` always carries the git-level
    name and date, so that is the field to trust for attribution.
    """

    sha: str
    html_url: str
    commit: CommitDetail
    author: GitHubUser | None = None
    parents: list[CommitParent] = []


class Review(_GitHubModel):
    """A pull request review.

    ``submitted_at`` is absent while a review is still ``PENDING``.
    """

    id: int
    state: str
    body: str | None = None
    user: GitHubUser
    html_url: str
    commit_id: str | None = None
    submitted_at: datetime | None = None


class ChangedFile(_GitHubModel):
    """A file touched by a pull request. Carries no timestamps of its own."""

    sha: str | None = None
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str | None = None
    previous_filename: str | None = None
