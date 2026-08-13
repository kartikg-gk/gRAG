"""Module 2: repository ingestion.

Acquires GitHub data and normalizes it into typed models. Nothing here extracts
entities, parses source code, builds a graph, or persists anything.
"""

from .collect import STAGE_CHANGED_FILES, STAGE_REVIEWS, collect_by_pull_request
from .order import in_ingest_order, ingest_key
from .github import (
    API_ROOT,
    PER_PAGE,
    GitHubAuthError,
    GitHubError,
    GitHubRateLimitError,
    fetch_changed_files,
    fetch_commits,
    fetch_issues,
    fetch_pull_requests,
    fetch_repository,
    fetch_reviews,
    make_session,
)
from .models import (
    ChangedFile,
    Commit,
    CommitDetail,
    CommitParent,
    GitActor,
    GitHubUser,
    Issue,
    Label,
    PullRequest,
    Repository,
    Review,
)

__all__ = [
    "API_ROOT",
    "PER_PAGE",
    "GitHubError",
    "GitHubAuthError",
    "GitHubRateLimitError",
    "make_session",
    "collect_by_pull_request",
    "STAGE_REVIEWS",
    "STAGE_CHANGED_FILES",
    "in_ingest_order",
    "ingest_key",
    "fetch_repository",
    "fetch_pull_requests",
    "fetch_issues",
    "fetch_commits",
    "fetch_reviews",
    "fetch_changed_files",
    "Repository",
    "PullRequest",
    "Issue",
    "Commit",
    "Review",
    "ChangedFile",
    "GitHubUser",
    "Label",
    "GitActor",
    "CommitDetail",
    "CommitParent",
]
