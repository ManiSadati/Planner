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


def _compact_payload(repo_changes: tuple[RepoChanges, ...], github_changes: tuple[GitHubChanges, ...]) -> dict:
    return {
        "repo_changes": [asdict(change) for change in repo_changes],
        "github_changes": [asdict(change) for change in github_changes],
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
