from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import CheckerConfig
from .git_collect import RepoChanges
from .github_collect import GitHubChanges
from .summarize import DailySummary


_PERSISTENT_CONTEXT_HEADING = "## Persistent Watch Context"


def has_changes(repo_changes: tuple[RepoChanges, ...], github_changes: tuple[GitHubChanges, ...]) -> bool:
    return any(change.branch_changes for change in repo_changes) or any(
        change.issues or change.prs for change in github_changes
    )


def write_reports(
    config: CheckerConfig,
    summary: DailySummary,
    repo_changes: tuple[RepoChanges, ...],
    github_changes: tuple[GitHubChanges, ...],
) -> None:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.daily_report_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    persistent_context = _persistent_context(config.generated_readme)
    config.generated_readme.write_text(
        _readme(
            now.isoformat(),
            summary,
            repo_changes,
            github_changes,
            persistent_context,
        )
    )

    if summary.should_write_daily_report and summary.importance >= config.big_change_threshold:
        daily_path = config.daily_report_dir / f"{now.date().isoformat()}.md"
        daily_path.write_text(_daily(now.isoformat(), summary))


def _readme(
    now: str,
    summary: DailySummary,
    repo_changes: tuple[RepoChanges, ...],
    github_changes: tuple[GitHubChanges, ...],
    persistent_context: str = "",
) -> str:
    repo_lines = []
    for repo in repo_changes:
        repo_lines.append(f"- {repo.repo_name}: {len(repo.branch_changes)} changed branches")
        for error in repo.errors:
            repo_lines.append(f"  - warning: {error}")
    for github in github_changes:
        repo_lines.append(
            f"- {github.repo}: {len(github.issues)} updated issues, {len(github.prs)} updated PRs"
        )
        for error in github.errors:
            repo_lines.append(f"  - warning: {error}")

    if not repo_lines:
        repo_lines.append("- No new tracked changes.")

    lines = [
        "# PTOAS State",
        "",
        f"Last updated: {now}",
        "",
        f"## {summary.title}",
        "",
        summary.ptoas_state.strip() or "No new tracked changes.",
        "",
        "## Scan Coverage",
        "",
        *repo_lines,
        "",
    ]
    if persistent_context:
        lines.extend((persistent_context.strip(), ""))
    return "\n".join(lines)


def _persistent_context(path: Path) -> str:
    if not path.exists():
        return ""
    current = path.read_text()
    marker = current.find(_PERSISTENT_CONTEXT_HEADING)
    if marker < 0:
        return ""
    return current[marker:].strip()


def _daily(now: str, summary: DailySummary) -> str:
    return "\n".join(
        [
            f"# {summary.title}",
            "",
            f"Generated: {now}",
            f"Importance: {summary.importance}/10",
            "",
            summary.daily_markdown.strip(),
            "",
        ]
    )
