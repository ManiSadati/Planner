from __future__ import annotations

from pathlib import Path

from .config import load_config
from .env import load_env_files
from .git_collect import RepoChanges, collect_git_changes
from .github_collect import GitHubChanges, collect_github_changes
from .report import has_changes, write_reports
from .state import CheckerState, utc_now_iso
from .summarize import DailySummary, summarize_changes


def run(config_path: Path, dry_run: bool = False) -> None:
    config = load_config(config_path)
    load_env_files(config.env_files)

    state = CheckerState.load(config.state_path)
    next_repo_states = dict(state.repos)
    next_github_states = dict(state.github)

    repo_changes: list[RepoChanges] = []
    github_changes: list[GitHubChanges] = []

    for repo in config.repos:
        previous_repo_state = state.repos.get(repo.id, {})
        if repo.track_branches:
            changes, repo_state = collect_git_changes(repo, config, previous_repo_state)
            repo_changes.append(changes)
            next_repo_states[repo.id] = repo_state

        previous_github_state = state.github.get(repo.id, {})
        github, github_state = collect_github_changes(repo, previous_github_state)
        if github is not None:
            github_changes.append(github)
            next_github_states[repo.id] = github_state

    repo_changes_tuple = tuple(repo_changes)
    github_changes_tuple = tuple(github_changes)

    if has_changes(repo_changes_tuple, github_changes_tuple):
        summary = summarize_changes(config.model, repo_changes_tuple, github_changes_tuple)
    else:
        summary = DailySummary(
            title="No new tracked PTOAS changes",
            importance=0,
            ptoas_state="No branch, issue, or PR changes were found since the last scan.",
            daily_markdown="No branch, issue, or PR changes were found since the last scan.",
            should_write_daily_report=False,
        )

    if dry_run:
        print(summary.ptoas_state)
        return

    write_reports(config, summary, repo_changes_tuple, github_changes_tuple)
    state.repos = next_repo_states
    state.github = next_github_states
    state.last_run_at = utc_now_iso()
    state.save(config.state_path)

