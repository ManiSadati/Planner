from __future__ import annotations

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
class BranchChange:
    ref: str
    old_sha: str | None
    new_sha: str
    subject: str
    committer_date: str
    commits: tuple[CommitInfo, ...]
    changed_files: tuple[str, ...]
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


def _changed_files(repo: RepoConfig, revision_range: str, limit: int) -> tuple[str, ...]:
    out = _git(repo, "diff", "--name-status", revision_range, check=False)
    return tuple(out.splitlines()[:limit])


def _diff_stat(repo: RepoConfig, revision_range: str) -> str:
    return _git(repo, "diff", "--stat", revision_range, check=False)


def _diff_excerpt(repo: RepoConfig, revision_range: str, limit: int) -> str:
    out = _git(repo, "diff", "--unified=2", "--no-ext-diff", revision_range, check=False)
    if len(out) <= limit:
        return out
    return out[:limit] + "\n\n[diff excerpt truncated]"


def collect_git_changes(
    repo: RepoConfig,
    config: CheckerConfig,
    previous_repo_state: dict,
) -> tuple[RepoChanges, dict]:
    errors = list(_fetch(repo))
    current_heads = _branch_heads(repo)
    previous_heads = previous_repo_state.get("branches", {})
    next_state = {"branches": current_heads}
    changes: list[BranchChange] = []

    for ref, head in sorted(current_heads.items()):
        old = previous_heads.get(ref)
        old_sha = old.get("sha") if old else None
        new_sha = head["sha"]
        if old_sha == new_sha:
            continue
        is_new_branch = old_sha is None
        if is_new_branch and not config.initial_backfill:
            continue

        revision_range = f"{old_sha}..{new_sha}" if old_sha else new_sha
        changes.append(
            BranchChange(
                ref=ref,
                old_sha=old_sha,
                new_sha=new_sha,
                subject=head["subject"],
                committer_date=head["date"],
                commits=_commits(repo, revision_range, config.max_commits_per_branch),
                changed_files=_changed_files(repo, revision_range, config.max_diff_files_per_branch),
                diff_stat=_diff_stat(repo, revision_range),
                diff_excerpt=_diff_excerpt(repo, revision_range, config.max_diff_chars_per_branch),
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
