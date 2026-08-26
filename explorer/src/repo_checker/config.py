from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GitHubForkWatch:
    full_name: str
    branches: tuple[str, ...]


@dataclass(frozen=True)
class RepoConfig:
    id: str
    name: str
    kind: str
    path: Path
    fetch_remotes: tuple[str, ...]
    track_branches: bool
    track_github_issues: bool
    track_github_prs: bool
    track_github_forks: bool
    priority_forks: tuple[GitHubForkWatch, ...]
    upstream: str | None = None
    fork: str | None = None
    upstream_url: str | None = None
    fork_url: str | None = None


@dataclass(frozen=True)
class CheckerConfig:
    root: Path
    state_path: Path
    reports_dir: Path
    daily_report_dir: Path
    generated_readme: Path
    timezone: str
    model: str
    big_change_threshold: int
    max_fork_depth: int
    max_commits_per_branch: int
    max_diff_files_per_branch: int
    max_diff_chars_per_branch: int
    max_markdown_files_for_diff_excerpt: int
    max_markdown_changed_lines_for_diff: int
    max_forks_to_scan: int
    max_branches_per_fork: int
    max_fork_branch_changes_per_run: int
    initial_backfill: bool
    env_files: tuple[Path, ...]
    repos: tuple[RepoConfig, ...]


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _resolve(root: Path, value: str) -> Path:
    path = _expand_path(value)
    return path if path.is_absolute() else root / path


def load_config(config_path: Path) -> CheckerConfig:
    config_path = _expand_path(str(config_path)).resolve()
    root = config_path.parent
    data: dict[str, Any] = json.loads(config_path.read_text())

    repos = []
    for item in data["repos"]:
        repos.append(
            RepoConfig(
                id=item["id"],
                name=item["name"],
                kind=item["kind"],
                path=_expand_path(item["path"]),
                fetch_remotes=tuple(item.get("fetch_remotes", ())),
                track_branches=bool(item.get("track_branches", True)),
                track_github_issues=bool(item.get("track_github_issues", False)),
                track_github_prs=bool(item.get("track_github_prs", False)),
                track_github_forks=bool(item.get("track_github_forks", False)),
                priority_forks=tuple(
                    GitHubForkWatch(
                        full_name=watch["repo"],
                        branches=tuple(watch.get("branches", ())),
                    )
                    for watch in item.get("priority_forks", ())
                ),
                upstream=item.get("upstream"),
                fork=item.get("fork"),
                upstream_url=item.get("upstream_url"),
                fork_url=item.get("fork_url"),
            )
        )

    return CheckerConfig(
        root=root,
        state_path=_resolve(root, data["state_path"]),
        reports_dir=_resolve(root, data["reports_dir"]),
        daily_report_dir=_resolve(root, data["daily_report_dir"]),
        generated_readme=_resolve(root, data["generated_readme"]),
        timezone=data.get("timezone", "UTC"),
        model=data.get("model", "gpt-5"),
        big_change_threshold=int(data.get("big_change_threshold", 7)),
        max_fork_depth=int(data.get("max_fork_depth", 2)),
        max_commits_per_branch=int(data.get("max_commits_per_branch", 40)),
        max_diff_files_per_branch=int(data.get("max_diff_files_per_branch", 80)),
        max_diff_chars_per_branch=int(data.get("max_diff_chars_per_branch", 12000)),
        max_markdown_files_for_diff_excerpt=int(
            data.get("max_markdown_files_for_diff_excerpt", 8)
        ),
        max_markdown_changed_lines_for_diff=int(
            data.get("max_markdown_changed_lines_for_diff", 1200)
        ),
        max_forks_to_scan=int(data.get("max_forks_to_scan", 120)),
        max_branches_per_fork=int(data.get("max_branches_per_fork", 30)),
        max_fork_branch_changes_per_run=int(data.get("max_fork_branch_changes_per_run", 60)),
        initial_backfill=bool(data.get("initial_backfill", False)),
        env_files=tuple(_resolve(root, value) for value in data.get("env_files", ())),
        repos=tuple(repos),
    )
