from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from .git_collect import RepoChanges
from .github_collect import GitHubChanges
from .triage import score_branch, score_github_item


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


def _triage_dict(result: object) -> dict:
    return asdict(result)


def _include_in_summary(triage: object) -> bool:
    return getattr(triage, "classification") in {"Investigate", "Watch"}


def _triage_score(item: dict) -> int:
    return int(item.get("triage", {}).get("score", 0))


def _file_count(item: dict) -> int:
    if "file_count" in item:
        return int(item["file_count"])
    return int(item.get("file_summary", {}).get("changed_file_count", 0))


def _sort_by_triage(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda item: (_triage_score(item), _file_count(item)), reverse=True)


def _skipped_summary(items: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        classification = item["triage"]["classification"]
        counts[classification] = counts.get(classification, 0) + 1
    sorted_items = _sort_by_triage(items)
    return {
        "counts": counts,
        "samples": sorted_items[:20],
    }


def _highlight_candidates(repo_changes: list[dict], github_changes: list[dict]) -> list[dict]:
    candidates = []
    for repo in repo_changes:
        repo_name = repo["repo_name"]
        for branch in repo["branch_changes"]:
            if _is_very_important(branch):
                candidates.append(
                    {
                        "kind": "branch",
                        "repo": repo_name,
                        "id": branch["ref"],
                        "title": branch["subject"],
                        "score": _triage_score(branch),
                        "classification": branch["triage"]["classification"],
                        "needs_codex_backfill": branch["triage"]["needs_codex_backfill"],
                        "positive_signals": branch["triage"]["positive_signals"][:6],
                    }
                )

    for github in github_changes:
        repo_name = github["repo"]
        for issue in github["issues"]:
            if _is_very_important(issue):
                candidates.append(
                    {
                        "kind": "issue",
                        "repo": repo_name,
                        "id": f"#{issue['number']}",
                        "title": issue["title"],
                        "url": issue["url"],
                        "score": _triage_score(issue),
                        "classification": issue["triage"]["classification"],
                        "needs_codex_backfill": issue["triage"]["needs_codex_backfill"],
                        "positive_signals": issue["triage"]["positive_signals"][:6],
                    }
                )
        for pr in github["prs"]:
            if _is_very_important(pr):
                candidates.append(
                    {
                        "kind": "pr",
                        "repo": repo_name,
                        "id": f"#{pr['number']}",
                        "title": pr["title"],
                        "url": pr["url"],
                        "score": _triage_score(pr),
                        "classification": pr["triage"]["classification"],
                        "needs_codex_backfill": pr["triage"]["needs_codex_backfill"],
                        "positive_signals": pr["triage"]["positive_signals"][:6],
                    }
                )

    return _sort_by_triage(candidates)[:15]


def _is_very_important(item: dict) -> bool:
    triage = item["triage"]
    return int(triage["score"]) >= 10 or bool(triage["needs_codex_backfill"])


def _compact_payload(
    repo_changes: tuple[RepoChanges, ...],
    github_changes: tuple[GitHubChanges, ...],
) -> dict:
    compact_repo_changes = []
    for repo in repo_changes:
        compact_branches = []
        skipped_branches = []
        for branch in repo.branch_changes:
            triage = score_branch(branch)
            entry = {
                "ref": branch.ref,
                "base_ref": branch.base_ref,
                "base_sha": branch.base_sha,
                "old_sha": branch.old_sha,
                "new_sha": branch.new_sha,
                "subject": branch.subject,
                "committer_date": branch.committer_date,
                "lookback_since": branch.lookback_since,
                "is_new_branch": branch.is_new_branch,
                "source_repo": branch.source_repo,
                "source_url": branch.source_url,
                "fork_depth": branch.fork_depth,
                "parent_repo": branch.parent_repo,
                "triage": _triage_dict(triage),
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
            if _include_in_summary(triage):
                compact_branches.append(entry)
            else:
                skipped_branches.append(
                    {
                        "ref": branch.ref,
                        "subject": branch.subject,
                        "triage": _triage_dict(triage),
                        "file_summary": asdict(branch.file_summary),
                    }
                )
        compact_repo_changes.append(
            {
                "repo_id": repo.repo_id,
                "repo_name": repo.repo_name,
                "branch_changes": _sort_by_triage(compact_branches),
                "skipped_branch_changes": _skipped_summary(skipped_branches),
                "errors": repo.errors,
            }
        )

    compact_github_changes = []
    for github in github_changes:
        compact_issues = []
        skipped_issues = []
        compact_prs = []
        skipped_prs = []
        for issue in github.issues:
            triage = score_github_item(issue)
            entry = {
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "author": issue.author,
                "url": issue.url,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
                "labels": issue.labels,
                "body_excerpt": _truncate(issue.body, 400),
                "triage": _triage_dict(triage),
            }
            if _include_in_summary(triage):
                compact_issues.append(entry)
            else:
                skipped_issues.append(entry)

        for pr in github.prs:
            triage = score_github_item(pr)
            entry = {
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
                "triage": _triage_dict(triage),
            }
            if _include_in_summary(triage):
                compact_prs.append(entry)
            else:
                skipped_prs.append(
                    {
                        "number": pr.number,
                        "title": pr.title,
                        "url": pr.url,
                        "file_count": len(pr.files),
                        "triage": _triage_dict(triage),
                    }
                )

        compact_github_changes.append(
            {
                "repo": github.repo,
                "issues": _sort_by_triage(compact_issues),
                "skipped_issues": _skipped_summary(skipped_issues),
                "prs": _sort_by_triage(compact_prs),
                "skipped_prs": _skipped_summary(skipped_prs),
                "errors": github.errors,
            }
        )

    return {
        "repo_changes": compact_repo_changes,
        "github_changes": compact_github_changes,
        "very_important_candidates": _highlight_candidates(
            compact_repo_changes,
            compact_github_changes,
        ),
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
        return _fallback_summary(
            payload,
            "OPENAI_API_KEY is not set; wrote a deterministic summary.",
        )

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
                        "should_write_daily_report is true only for big or meaningful changes. "
                        "Use deterministic triage fields to prioritize Investigate and Watch "
                        "items; do not expand Low priority or Noise samples unless the skip "
                        "pattern itself is important. Preserve score-sorted priority in the "
                        "report. End daily_markdown with an 'Agent Highlights' section that "
                        "lists the few candidates you judge most important, using "
                        "very_important_candidates as the starting point.\n\n"
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
