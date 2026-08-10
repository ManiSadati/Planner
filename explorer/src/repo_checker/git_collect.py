from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import subprocess

from .config import CheckerConfig, RepoConfig


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    timestamp: str
    author: str
    subject: str


@dataclass(frozen=True)
class FileChangeSummary:
    changed_file_count: int
    added_lines: int
    deleted_lines: int
    markdown_file_count: int
    markdown_added_lines: int
    markdown_deleted_lines: int
    binary_file_count: int
    top_locations: tuple[str, ...]
    markdown_files_sample: tuple[str, ...]
    diff_excerpt_policy: str


@dataclass(frozen=True)
class NumstatEntry:
    additions: int | None
    deletions: int | None
    path: str


@dataclass(frozen=True)
class BranchChange:
    ref: str
    base_ref: str | None
    base_sha: str | None
    old_sha: str | None
    new_sha: str
    subject: str
    committer_date: str
    lookback_since: str | None
    commits: tuple[CommitInfo, ...]
    changed_files: tuple[str, ...]
    file_summary: FileChangeSummary
    diff_stat: str
    diff_excerpt: str
    is_new_branch: bool


@dataclass(frozen=True)
class RepoChanges:
    repo_id: str
    repo_name: str
    path: str
    branch_changes: tuple[BranchChange, ...]
    errors: tuple[str, ...] = ()


def _git(repo: RepoConfig, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo.path), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _fetch(repo: RepoConfig) -> tuple[str, ...]:
    errors = []
    for remote in repo.fetch_remotes:
        result = subprocess.run(
            ["git", "-C", str(repo.path), "fetch", "--prune", remote],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            errors.append(f"fetch {remote}: {result.stderr.strip() or result.stdout.strip()}")
    return tuple(errors)


def _branch_heads(repo: RepoConfig) -> dict[str, dict[str, str]]:
    fmt = "%(refname:short)%00%(objectname)%00%(committerdate:iso-strict)%00%(subject)"
    out = _git(repo, "for-each-ref", f"--format={fmt}", "refs/heads", "refs/remotes")
    heads: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        if not line:
            continue
        ref, sha, date, subject = line.split("\x00", 3)
        if ref.endswith("/HEAD"):
            continue
        heads[ref] = {"sha": sha, "date": date, "subject": subject}
    return heads


def _commits(repo: RepoConfig, revision_range: str, limit: int) -> tuple[CommitInfo, ...]:
    fmt = "%H%x00%cI%x00%an%x00%s"
    out = _git(repo, "log", f"--max-count={limit}", f"--format={fmt}", revision_range, check=False)
    commits = []
    for line in out.splitlines():
        if not line:
            continue
        sha, timestamp, author, subject = line.split("\x00", 3)
        commits.append(CommitInfo(sha=sha, timestamp=timestamp, author=author, subject=subject))
    return tuple(commits)


def _base_for_ref(
    ref: str,
    current_heads: dict[str, dict[str, str]],
) -> tuple[str | None, str | None]:
    candidates: list[str] = []
    if "/" in ref:
        remote = ref.split("/", 1)[0]
        candidates.extend((f"{remote}/main", f"{remote}/master"))

    candidates.extend(
        ("upstream/main", "origin/main", "main", "master", "upstream/master", "origin/master")
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen or candidate == ref:
            continue
        seen.add(candidate)
        head = current_heads.get(candidate)
        if head:
            return candidate, head["sha"]

    return None, None


def _merge_base(repo: RepoConfig, base_sha: str, new_sha: str) -> str | None:
    out = _git(repo, "merge-base", base_sha, new_sha, check=False).strip()
    return out or None


def _commit_before(repo: RepoConfig, ref: str, since: str) -> str | None:
    out = _git(repo, "rev-list", "-1", f"--before={since}", ref, check=False).strip()
    return out or None


def _revision_range(
    repo: RepoConfig,
    old_sha: str | None,
    new_sha: str,
    base_sha: str | None,
) -> tuple[str, str | None]:
    if old_sha:
        return f"{old_sha}..{new_sha}", old_sha

    if base_sha:
        merge_base = _merge_base(repo, base_sha, new_sha)
        if merge_base:
            return f"{merge_base}..{new_sha}", merge_base

    parent = _git(repo, "rev-parse", f"{new_sha}^", check=False).strip()
    if parent:
        return f"{parent}..{new_sha}", parent

    return new_sha, None


def _changed_files(repo: RepoConfig, revision_range: str, limit: int) -> tuple[str, ...]:
    out = _git(repo, "diff", "--name-status", revision_range, check=False)
    return tuple(out.splitlines()[:limit])


def _numstat(repo: RepoConfig, revision_range: str) -> tuple[NumstatEntry, ...]:
    out = _git(repo, "diff", "--numstat", "--find-renames", revision_range, check=False)
    entries = []
    for line in out.splitlines():
        if not line:
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            continue
        added, deleted, path = fields
        entries.append(
            NumstatEntry(
                additions=None if added == "-" else int(added),
                deletions=None if deleted == "-" else int(deleted),
                path=path,
            )
        )
    return tuple(entries)


def _is_markdown_path(path: str) -> bool:
    return path.lower().endswith((".md", ".markdown"))


def _location(path: str) -> str:
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    if len(parts) <= 1:
        return "."
    return "/".join(parts[:2])


def _file_change_summary(
    entries: tuple[NumstatEntry, ...],
    config: CheckerConfig,
) -> FileChangeSummary:
    markdown_entries = [entry for entry in entries if _is_markdown_path(entry.path)]
    added_lines = sum(entry.additions or 0 for entry in entries)
    deleted_lines = sum(entry.deletions or 0 for entry in entries)
    markdown_added_lines = sum(entry.additions or 0 for entry in markdown_entries)
    markdown_deleted_lines = sum(entry.deletions or 0 for entry in markdown_entries)
    markdown_changed_lines = markdown_added_lines + markdown_deleted_lines
    top_locations = tuple(
        f"{location}: {count}"
        for location, count in Counter(_location(entry.path) for entry in entries).most_common(12)
    )
    markdown_files_sample = tuple(
        entry.path for entry in markdown_entries[: config.max_markdown_files_for_diff_excerpt]
    )

    if not markdown_entries:
        diff_excerpt_policy = "metadata-only: no Markdown files changed"
    elif len(markdown_entries) > config.max_markdown_files_for_diff_excerpt:
        diff_excerpt_policy = (
            "metadata-only: "
            f"{len(markdown_entries)} Markdown files changed, above limit "
            f"{config.max_markdown_files_for_diff_excerpt}"
        )
    elif markdown_changed_lines > config.max_markdown_changed_lines_for_diff:
        diff_excerpt_policy = (
            "metadata-only: "
            f"{markdown_changed_lines} Markdown changed lines, above limit "
            f"{config.max_markdown_changed_lines_for_diff}"
        )
    else:
        diff_excerpt_policy = (
            "Markdown excerpt included for "
            f"{len(markdown_files_sample)} file(s), capped at "
            f"{config.max_diff_chars_per_branch} chars"
        )

    return FileChangeSummary(
        changed_file_count=len(entries),
        added_lines=added_lines,
        deleted_lines=deleted_lines,
        markdown_file_count=len(markdown_entries),
        markdown_added_lines=markdown_added_lines,
        markdown_deleted_lines=markdown_deleted_lines,
        binary_file_count=sum(
            1 for entry in entries if entry.additions is None or entry.deletions is None
        ),
        top_locations=top_locations,
        markdown_files_sample=markdown_files_sample,
        diff_excerpt_policy=diff_excerpt_policy,
    )


def _diff_stat(repo: RepoConfig, revision_range: str, limit: int) -> str:
    out = _git(repo, "diff", "--stat", revision_range, check=False)
    lines = out.splitlines()
    if len(lines) <= limit:
        return out
    return "\n".join(lines[:limit]) + f"\n[diff stat truncated after {limit} lines]"


def _diff_excerpt(
    repo: RepoConfig,
    revision_range: str,
    summary: FileChangeSummary,
    limit: int,
) -> str:
    if not summary.diff_excerpt_policy.startswith("Markdown excerpt included"):
        return f"[{summary.diff_excerpt_policy}]"

    out = _git(
        repo,
        "diff",
        "--unified=1",
        "--no-ext-diff",
        revision_range,
        "--",
        *summary.markdown_files_sample,
        check=False,
    )
    if len(out) <= limit:
        return out
    return out[:limit] + "\n\n[markdown diff excerpt truncated]"


def collect_git_changes(
    repo: RepoConfig,
    config: CheckerConfig,
    previous_repo_state: dict,
    since_override: str | None = None,
) -> tuple[RepoChanges, dict]:
    errors = list(_fetch(repo))
    current_heads = _branch_heads(repo)
    previous_heads = previous_repo_state.get("branches", {})
    next_state = {"branches": current_heads}
    changes: list[BranchChange] = []
    is_bootstrap = not previous_heads

    for ref, head in sorted(current_heads.items()):
        old = previous_heads.get(ref)
        old_sha = _commit_before(repo, ref, since_override) if since_override else None
        if not since_override:
            old_sha = old.get("sha") if old else None
        new_sha = head["sha"]
        if old_sha == new_sha:
            continue
        is_new_branch = old_sha is None
        if not since_override and is_new_branch and is_bootstrap and not config.initial_backfill:
            continue

        base_ref, configured_base_sha = _base_for_ref(ref, current_heads)
        compare_ref = "previous-seen-head" if old_sha else base_ref
        revision_range, base_sha = _revision_range(repo, old_sha, new_sha, configured_base_sha)
        numstat = _numstat(repo, revision_range)
        file_summary = _file_change_summary(numstat, config)
        changes.append(
            BranchChange(
                ref=ref,
                base_ref=compare_ref,
                base_sha=base_sha,
                old_sha=old_sha,
                new_sha=new_sha,
                subject=head["subject"],
                committer_date=head["date"],
                lookback_since=since_override,
                commits=_commits(repo, revision_range, config.max_commits_per_branch),
                changed_files=_changed_files(
                    repo,
                    revision_range,
                    config.max_diff_files_per_branch,
                ),
                file_summary=file_summary,
                diff_stat=_diff_stat(
                    repo,
                    revision_range,
                    config.max_diff_files_per_branch,
                ),
                diff_excerpt=_diff_excerpt(
                    repo,
                    revision_range,
                    file_summary,
                    config.max_diff_chars_per_branch,
                ),
                is_new_branch=is_new_branch,
            )
        )

    return (
        RepoChanges(
            repo_id=repo.id,
            repo_name=repo.name,
            path=str(repo.path),
            branch_changes=tuple(changes),
            errors=tuple(errors),
        ),
        next_state,
    )
