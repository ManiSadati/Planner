from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from .git_collect import RepoChanges
from .github_collect import GitHubChanges


@dataclass(frozen=True)
class DailySummary:
    title: str
    importance: int
    ptoas_state: str
    daily_markdown: str
    should_write_daily_report: bool


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"


def _compact_payload(repo_changes: tuple[RepoChanges, ...], github_changes: tuple[GitHubChanges, ...]) -> dict:
    compact_repo_changes = []
    for repo in repo_changes:
        compact_branches = []
        for branch in repo.branch_changes:
            compact_branches.append(
                {
                    "ref": branch.ref,
                    "base_ref": branch.base_ref,
                    "base_sha": branch.base_sha,
                    "old_sha": branch.old_sha,
                    "new_sha": branch.new_sha,
                    "subject": branch.subject,
                    "committer_date": branch.committer_date,
                    "lookback_since": branch.lookback_since,
                    "is_new_branch": branch.is_new_branch,
                    "commits": [
                        {
                            "sha": commit.sha[:12],
                            "timestamp": commit.timestamp,
                            "author": commit.author,
                            "subject": commit.subject,
                        }
                        for commit in branch.commits[:12]
                    ],
                    "changed_files_sample": branch.changed_files[:40],
                    "file_summary": asdict(branch.file_summary),
                    "diff_stat_excerpt": "\n".join(branch.diff_stat.splitlines()[:20]),
                    "diff_excerpt": _truncate(branch.diff_excerpt, 4000),
                }
            )
        compact_repo_changes.append(
            {
                "repo_id": repo.repo_id,
                "repo_name": repo.repo_name,
                "branch_changes": compact_branches,
                "errors": repo.errors,
            }
        )

    compact_github_changes = []
    for github in github_changes:
        compact_github_changes.append(
            {
                "repo": github.repo,
                "issues": [
                    {
                        "number": issue.number,
                        "title": issue.title,
                        "state": issue.state,
                        "author": issue.author,
                        "url": issue.url,
                        "created_at": issue.created_at,
                        "updated_at": issue.updated_at,
                        "labels": issue.labels,
                        "body_excerpt": _truncate(issue.body, 400),
                    }
                    for issue in github.issues
                ],
                "prs": [
                    {
                        "number": pr.number,
                        "title": pr.title,
                        "state": pr.state,
                        "author": pr.author,
                        "url": pr.url,
                        "created_at": pr.created_at,
                        "updated_at": pr.updated_at,
                        "labels": pr.labels,
                        "body_excerpt": _truncate(pr.body, 400),
                        "file_count": len(pr.files),
                        "files_sample": pr.files[:30],
                        "additions": pr.additions,
                        "deletions": pr.deletions,
                    }
                    for pr in github.prs
                ],
                "errors": github.errors,
            }
        )

    return {
        "repo_changes": compact_repo_changes,
        "github_changes": compact_github_changes,
    }


def _json_from_text(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    return json.loads(candidate)


def _agent_contract() -> str:
    path = Path(__file__).resolve().parents[2] / "AGENT.md"
    if not path.exists():
        return ""
    return path.read_text()


def summarize_changes(
    model: str,
    repo_changes: tuple[RepoChanges, ...],
    github_changes: tuple[GitHubChanges, ...],
) -> DailySummary:
    payload = _compact_payload(repo_changes, github_changes)
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    if not has_openai_key:
        return _fallback_summary(payload, "OPENAI_API_KEY is not set; wrote a deterministic summary.")

    try:
        from openai import OpenAI

        client = OpenAI()
        agent_contract = _agent_contract()
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful repo intelligence agent. Summarize only new changes "
                        "in the payload. Focus on PTOAS state, branch movement, issues, PRs, "
                        "risk, and what changed technically. Follow the explorer agent contract "
                        "below. Return strict JSON only.\n\n"
                        f"{agent_contract}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Return JSON with keys: title, importance, ptoas_state, "
                        "daily_markdown, should_write_daily_report. importance is 0-10. "
                        "should_write_daily_report is true only for big or meaningful changes.\n\n"
                        f"Payload:\n{json.dumps(payload, indent=2)}"
                    ),
                },
            ],
        )
        data = _json_from_text(response.output_text)
        return DailySummary(
            title=str(data.get("title", "Daily PTOAS update")),
            importance=max(0, min(10, int(data.get("importance", 0)))),
            ptoas_state=str(data.get("ptoas_state", "")),
            daily_markdown=str(data.get("daily_markdown", "")),
            should_write_daily_report=bool(data.get("should_write_daily_report", False)),
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback_summary(payload, f"OpenAI summarization failed: {exc}")


def _fallback_summary(payload: dict, note: str) -> DailySummary:
    branch_count = sum(len(repo["branch_changes"]) for repo in payload["repo_changes"])
    issue_count = sum(len(repo["issues"]) for repo in payload["github_changes"])
    pr_count = sum(len(repo["prs"]) for repo in payload["github_changes"])
    importance = min(10, branch_count + issue_count + pr_count)
    title = f"{branch_count} branch updates, {issue_count} issues, {pr_count} PRs"
    body = [
        f"# {title}",
        "",
        f"> {note}",
        "",
        f"- Branch updates: {branch_count}",
        f"- Issue updates: {issue_count}",
        f"- PR updates: {pr_count}",
    ]
    return DailySummary(
        title=title,
        importance=importance,
        ptoas_state="\n".join(body),
        daily_markdown="\n".join(body),
        should_write_daily_report=importance >= 7,
    )
