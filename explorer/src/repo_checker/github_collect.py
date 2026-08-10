from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import RepoConfig


@dataclass(frozen=True)
class GitHubItem:
    kind: str
    number: int
    title: str
    state: str
    author: str
    url: str
    created_at: str
    updated_at: str
    body: str
    labels: tuple[str, ...]
    files: tuple[str, ...] = ()
    additions: int | None = None
    deletions: int | None = None


@dataclass(frozen=True)
class GitHubChanges:
    repo: str
    issues: tuple[GitHubItem, ...]
    prs: tuple[GitHubItem, ...]
    errors: tuple[str, ...] = ()


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ptoas-repo-checker",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(url: str) -> Any:
    request = Request(url, headers=_headers())
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _page(url: str, params: dict[str, str | int]) -> list[dict[str, Any]]:
    query = urlencode(params)
    return _get_json(f"{url}?{query}")


def _updated_pages(
    url: str,
    params: dict[str, str | int],
    since: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    latest: str | None = None
    for page in range(1, 11):
        page_rows = _page(url, {**params, "page": page})
        if not page_rows:
            break
        should_continue = False
        for row in page_rows:
            updated_at = row.get("updated_at")
            if updated_at and (latest is None or updated_at > latest):
                latest = updated_at
            if since and updated_at and updated_at <= since:
                continue
            if since:
                rows.append(row)
            should_continue = True
        if since is None:
            break
        if since and not should_continue:
            break
    return rows, latest


def _item(kind: str, raw: dict[str, Any], files: tuple[str, ...] = ()) -> GitHubItem:
    labels = tuple(label.get("name", "") for label in raw.get("labels", []) if label.get("name"))
    return GitHubItem(
        kind=kind,
        number=int(raw["number"]),
        title=raw.get("title", ""),
        state=raw.get("state", ""),
        author=(raw.get("user") or {}).get("login", "unknown"),
        url=raw.get("html_url", ""),
        created_at=raw.get("created_at", ""),
        updated_at=raw.get("updated_at", ""),
        body=(raw.get("body") or "")[:4000],
        labels=labels,
        files=files,
        additions=raw.get("additions"),
        deletions=raw.get("deletions"),
    )


def _pr_files(base: str, number: int) -> tuple[str, ...]:
    rows = _page(f"{base}/pulls/{number}/files", {"per_page": 100})
    files = []
    for row in rows:
        filename = row.get("filename", "")
        status = row.get("status", "")
        additions = row.get("additions", 0)
        deletions = row.get("deletions", 0)
        files.append(f"{status} {filename} (+{additions}/-{deletions})")
    return tuple(files)


def _latest_timestamp(items: tuple[GitHubItem, ...], fallback: str | None) -> str:
    timestamps = [item.updated_at for item in items if item.updated_at]
    if timestamps:
        return max(timestamps)
    return fallback or datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _bootstrap_timestamp(latest: str | None, errors: tuple[str, ...]) -> str | None:
    if latest:
        return latest
    if errors:
        return None
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def collect_github_changes(
    repo: RepoConfig,
    previous_state: dict,
    since_override: str | None = None,
) -> tuple[GitHubChanges | None, dict]:
    github_repo = repo.upstream
    if not github_repo or not (repo.track_github_issues or repo.track_github_prs):
        return None, previous_state

    base = f"https://api.github.com/repos/{github_repo}"
    state_issue_since = previous_state.get("issues_last_seen_updated_at")
    state_pr_since = previous_state.get("prs_last_seen_updated_at")
    previous_issue_since = since_override or state_issue_since
    previous_pr_since = since_override or state_pr_since
    errors: list[str] = []
    issues: list[GitHubItem] = []
    prs: list[GitHubItem] = []

    latest_issues: str | None = None
    latest_prs: str | None = None

    try:
        params: dict[str, str | int] = {
            "state": "all",
            "sort": "updated",
            "direction": "desc",
            "per_page": 100,
        }
        if previous_issue_since:
            params["since"] = previous_issue_since
        issue_rows, latest_issues = _updated_pages(f"{base}/issues", params, previous_issue_since)
        if repo.track_github_issues:
            for row in issue_rows:
                if "pull_request" in row:
                    continue
                issues.append(_item("issue", row))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"issues: {exc}")

    try:
        pr_rows, latest_prs = _updated_pages(
            f"{base}/pulls",
            {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100},
            previous_pr_since,
        )
        if repo.track_github_prs:
            for row in pr_rows:
                files = _pr_files(base, int(row["number"]))
                prs.append(_item("pr", row, files=files))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"prs: {exc}")

    if not since_override and not previous_issue_since and not previous_pr_since:
        # Bootstrap mode records cursors without summarizing historical issue/PR backlog.
        error_tuple = tuple(errors)
        next_state = {
            "issues_last_seen_updated_at": _bootstrap_timestamp(latest_issues, error_tuple),
            "prs_last_seen_updated_at": _bootstrap_timestamp(latest_prs, error_tuple),
        }
        return GitHubChanges(repo=github_repo, issues=(), prs=(), errors=error_tuple), next_state

    next_state = {
        "issues_last_seen_updated_at": _latest_timestamp(
            tuple(issues),
            state_issue_since or previous_issue_since,
        ),
        "prs_last_seen_updated_at": _latest_timestamp(
            tuple(prs),
            state_pr_since or previous_pr_since,
        ),
    }
    return (
        GitHubChanges(repo=github_repo, issues=tuple(issues), prs=tuple(prs), errors=tuple(errors)),
        next_state,
    )
