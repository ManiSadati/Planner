from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from .config import CheckerConfig, RepoConfig
from .git_collect import BranchChange, CommitInfo, FileChangeSummary, RepoChanges
from .github_collect import _get_json, _page


@dataclass(frozen=True)
class ForkRepo:
    full_name: str
    html_url: str
    default_branch: str
    fork_depth: int
    parent_repo: str
    pushed_at: str


@dataclass(frozen=True)
class ForkBranch:
    name: str
    sha: str


def collect_github_fork_changes(
    repo: RepoConfig,
    config: CheckerConfig,
    previous_state: dict,
    since_override: str | None = None,
) -> tuple[RepoChanges | None, dict]:
    if not repo.upstream or not repo.track_github_forks:
        return None, previous_state

    errors: list[str] = []
    forks = _discover_forks(repo.upstream, config, errors)
    previous_forks = previous_state.get("forks", {})
    next_forks: dict[str, Any] = {}
    changes: list[BranchChange] = []
    is_bootstrap = not previous_forks

    for fork in forks:
        if len(changes) >= config.max_fork_branch_changes_per_run:
            errors.append(
                "fork branch change cap reached: "
                f"{config.max_fork_branch_changes_per_run}"
            )
            break

        branches = _branches(fork.full_name, config.max_branches_per_fork, errors)
        previous_branches = previous_forks.get(fork.full_name, {}).get("branches", {})
        next_forks[fork.full_name] = {
            "html_url": fork.html_url,
            "default_branch": fork.default_branch,
            "fork_depth": fork.fork_depth,
            "parent_repo": fork.parent_repo,
            "pushed_at": fork.pushed_at,
            "branches": {
                branch.name: {"sha": branch.sha}
                for branch in branches
            },
        }

        for branch in branches:
            if len(changes) >= config.max_fork_branch_changes_per_run:
                break

            old_sha = (
                _commit_before(fork.full_name, branch.name, since_override, errors)
                if since_override
                else previous_branches.get(branch.name, {}).get("sha")
            )
            if old_sha == branch.sha:
                continue

            is_new_branch = old_sha is None
            if (
                not since_override
                and is_new_branch
                and is_bootstrap
                and not config.initial_backfill
            ):
                continue

            change = _branch_change(fork, branch, old_sha, since_override, config, errors)
            if change is not None:
                changes.append(change)

    next_state = {
        "scanned_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "root": repo.upstream,
        "max_fork_depth": config.max_fork_depth,
        "forks": next_forks,
    }
    return (
        RepoChanges(
            repo_id=f"{repo.id}:github-forks",
            repo_name=f"{repo.name} GitHub fork network",
            path=f"https://github.com/{repo.upstream}/forks",
            branch_changes=tuple(changes),
            errors=tuple(errors),
        ),
        next_state,
    )


def _discover_forks(
    root_repo: str,
    config: CheckerConfig,
    errors: list[str],
) -> tuple[ForkRepo, ...]:
    discovered: dict[str, ForkRepo] = {}
    queue: deque[tuple[str, int]] = deque([(root_repo, 0)])

    while queue and len(discovered) < config.max_forks_to_scan:
        parent, parent_depth = queue.popleft()
        child_depth = parent_depth + 1
        if child_depth > config.max_fork_depth:
            continue

        for raw in _forks(parent, errors):
            full_name = raw.get("full_name", "")
            if not full_name or full_name in discovered or full_name == root_repo:
                continue
            fork = ForkRepo(
                full_name=full_name,
                html_url=raw.get("html_url", ""),
                default_branch=raw.get("default_branch", "main"),
                fork_depth=child_depth,
                parent_repo=parent,
                pushed_at=raw.get("pushed_at", "") or raw.get("updated_at", ""),
            )
            discovered[full_name] = fork
            if len(discovered) >= config.max_forks_to_scan:
                break
            if child_depth < config.max_fork_depth:
                queue.append((full_name, child_depth))

    return tuple(
        sorted(
            discovered.values(),
            key=lambda fork: (fork.pushed_at, fork.full_name),
            reverse=True,
        )
    )


def _forks(full_name: str, errors: list[str]) -> tuple[dict[str, Any], ...]:
    try:
        rows: list[dict[str, Any]] = []
        for page in range(1, 4):
            page_rows = _page(
                f"https://api.github.com/repos/{full_name}/forks",
                {"sort": "newest", "per_page": 100, "page": page},
            )
            if not page_rows:
                break
            rows.extend(page_rows)
        return tuple(rows)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"forks {full_name}: {exc}")
        return ()


def _branches(full_name: str, limit: int, errors: list[str]) -> tuple[ForkBranch, ...]:
    try:
        rows = _page(
            f"https://api.github.com/repos/{full_name}/branches",
            {"per_page": min(100, limit)},
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"branches {full_name}: {exc}")
        return ()

    branches = []
    for row in rows[:limit]:
        commit = row.get("commit") or {}
        name = row.get("name", "")
        sha = commit.get("sha", "")
        if name and sha:
            branches.append(ForkBranch(name=name, sha=sha))
    return tuple(branches)


def _commit_before(
    full_name: str,
    branch: str,
    since: str | None,
    errors: list[str],
) -> str | None:
    if not since:
        return None
    try:
        rows = _page(
            f"https://api.github.com/repos/{full_name}/commits",
            {"sha": branch, "until": since, "per_page": 1},
        )
        if rows:
            return (rows[0].get("sha") or "").strip() or None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"commit before {full_name}:{branch}: {exc}")
    return None


def _branch_change(
    fork: ForkRepo,
    branch: ForkBranch,
    old_sha: str | None,
    since_override: str | None,
    config: CheckerConfig,
    errors: list[str],
) -> BranchChange | None:
    base_ref = "previous-seen-head" if old_sha else fork.default_branch
    base_sha = old_sha or _default_branch_sha(fork.full_name, fork.default_branch, errors)
    compare = _compare(fork.full_name, base_sha, branch.sha, errors) if base_sha else None
    files = tuple((compare or {}).get("files", ()))
    commits = tuple((compare or {}).get("commits", ()))
    commit = _commit(fork.full_name, branch.sha, errors)
    summary = _file_summary(files)
    changed_files = _changed_files(files, config.max_diff_files_per_branch)

    return BranchChange(
        ref=f"{fork.full_name}:{branch.name}",
        base_ref=base_ref,
        base_sha=base_sha,
        old_sha=old_sha,
        new_sha=branch.sha,
        subject=_commit_subject(commit),
        committer_date=_commit_date(commit),
        lookback_since=since_override,
        commits=_commits(commits, config.max_commits_per_branch),
        changed_files=changed_files,
        file_summary=summary,
        diff_stat="\n".join(changed_files[: config.max_diff_files_per_branch]),
        diff_excerpt="[metadata-only: GitHub fork discovery does not fetch raw diffs]",
        is_new_branch=old_sha is None,
        source_repo=fork.full_name,
        source_url=fork.html_url,
        fork_depth=fork.fork_depth,
        parent_repo=fork.parent_repo,
    )


def _default_branch_sha(full_name: str, default_branch: str, errors: list[str]) -> str | None:
    try:
        rows = _branches(full_name, 100, errors)
        for branch in rows:
            if branch.name == default_branch:
                return branch.sha
    except Exception as exc:  # noqa: BLE001
        errors.append(f"default branch {full_name}:{default_branch}: {exc}")
    return None


def _compare(
    full_name: str,
    base_sha: str | None,
    head_sha: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not base_sha or base_sha == head_sha:
        return {"files": (), "commits": ()}
    try:
        base = quote(base_sha, safe="")
        head = quote(head_sha, safe="")
        return _get_json(f"https://api.github.com/repos/{full_name}/compare/{base}...{head}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"compare {full_name}: {exc}")
        return None


def _commit(full_name: str, sha: str, errors: list[str]) -> dict[str, Any]:
    try:
        return _get_json(f"https://api.github.com/repos/{full_name}/commits/{sha}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"commit {full_name}@{sha[:12]}: {exc}")
        return {}


def _commit_subject(raw: dict[str, Any]) -> str:
    message = ((raw.get("commit") or {}).get("message") or "").strip()
    return message.splitlines()[0] if message else ""


def _commit_date(raw: dict[str, Any]) -> str:
    commit = raw.get("commit") or {}
    committer = commit.get("committer") or {}
    author = commit.get("author") or {}
    return committer.get("date") or author.get("date") or ""


def _commits(raw_commits: tuple[dict[str, Any], ...], limit: int) -> tuple[CommitInfo, ...]:
    commits = []
    for raw in raw_commits[:limit]:
        commit = raw.get("commit") or {}
        author = commit.get("author") or {}
        commits.append(
            CommitInfo(
                sha=(raw.get("sha") or "")[:40],
                timestamp=(author.get("date") or ""),
                author=(author.get("name") or ""),
                subject=(commit.get("message") or "").splitlines()[0],
            )
        )
    return tuple(commits)


def _changed_files(files: tuple[dict[str, Any], ...], limit: int) -> tuple[str, ...]:
    lines = []
    for file in files[:limit]:
        status = file.get("status", "changed")
        filename = file.get("filename", "")
        additions = int(file.get("additions") or 0)
        deletions = int(file.get("deletions") or 0)
        lines.append(f"{status} {filename} (+{additions}/-{deletions})")
    return tuple(lines)


def _file_summary(files: tuple[dict[str, Any], ...]) -> FileChangeSummary:
    paths = tuple(str(file.get("filename", "")) for file in files if file.get("filename"))
    markdown_files = tuple(path for path in paths if path.lower().endswith((".md", ".markdown")))
    additions = sum(int(file.get("additions") or 0) for file in files)
    deletions = sum(int(file.get("deletions") or 0) for file in files)
    markdown_additions = sum(
        int(file.get("additions") or 0)
        for file in files
        if str(file.get("filename", "")).lower().endswith((".md", ".markdown"))
    )
    markdown_deletions = sum(
        int(file.get("deletions") or 0)
        for file in files
        if str(file.get("filename", "")).lower().endswith((".md", ".markdown"))
    )
    top_locations = tuple(
        f"{location}: {count}"
        for location, count in Counter(_location(path) for path in paths).most_common(12)
    )

    return FileChangeSummary(
        changed_file_count=len(paths),
        added_lines=additions,
        deleted_lines=deletions,
        markdown_file_count=len(markdown_files),
        markdown_added_lines=markdown_additions,
        markdown_deleted_lines=markdown_deletions,
        binary_file_count=0,
        top_locations=top_locations,
        markdown_files_sample=markdown_files[:8],
        diff_excerpt_policy="metadata-only: GitHub fork discovery does not fetch raw diffs",
    )


def _location(path: str) -> str:
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    if len(parts) <= 1:
        return "."
    return "/".join(parts[:2])
