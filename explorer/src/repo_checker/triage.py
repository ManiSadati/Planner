from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class TriageResult:
    score: int
    classification: str
    positive_signals: tuple[str, ...]
    negative_signals: tuple[str, ...]
    skip_reason: str | None
    needs_codex_backfill: bool
    large_change: bool


KEYWORD_SCORES: tuple[tuple[str, int], ...] = (
    ("vmi", 4),
    ("vpto", 3),
    ("tilelib", 3),
    ("sync", 3),
    ("memory planning", 3),
    ("memory-plan", 3),
    ("memplan", 3),
    ("fp4", 3),
    ("scheduler", 3),
    ("softlib", 3),
    ("ptodsl", 2),
    ("migrat", 2),
    ("lowering", 2),
    ("pipeline", 2),
    ("vminormalizesignlessinttounsigned", 4),
    ("vsmins", 4),
    ("vsmaxs", 4),
    ("vmins", 4),
    ("vmaxs", 4),
    ("vector-scalar", 3),
    ("vector scalar", 3),
    ("reduction", 2),
    ("result shape", 2),
    ("vcadd", 4),
    ("vcmax", 4),
    ("reduce_addf", 4),
    ("reduce_maxf", 4),
    ("brc_b32", 4),
    ("brc b32", 4),
    ("broadcast load", 3),
    ("one-lane", 2),
    ("onept_b32", 4),
    ("one-point", 3),
    ("rank-zero", 3),
)

AI_PREFIXES = ("codex/", "claude/", "copilot/", "agent/", "ai/")
LOW_VALUE_PATH_PREFIXES = (".github/", ".claude/", ".cursor/", "openspec/", "adr/")
LOW_VALUE_FILE_PREFIXES = ("agent", "claude")


def score_branch(branch: object) -> TriageResult:
    files = tuple(getattr(branch, "changed_files", ()))
    file_count = getattr(getattr(branch, "file_summary", None), "changed_file_count", len(files))
    text = " ".join(
        [
            str(getattr(branch, "ref", "")),
            str(getattr(branch, "subject", "")),
            *_paths(files),
        ]
    )
    return _score_candidate(
        identifier=str(getattr(branch, "ref", "")),
        text=text,
        files=files,
        file_count=file_count,
        is_branch=True,
    )


def score_github_item(item: object) -> TriageResult:
    files = tuple(getattr(item, "files", ()))
    file_count = len(files)
    text = " ".join(
        [
            str(getattr(item, "title", "")),
            str(getattr(item, "body", "")),
            *_paths(files),
        ]
    )
    return _score_candidate(
        identifier=f"#{getattr(item, 'number', 'unknown')}",
        text=text,
        files=files,
        file_count=file_count,
        is_branch=False,
    )


def _score_candidate(
    identifier: str,
    text: str,
    files: tuple[str, ...],
    file_count: int,
    is_branch: bool,
) -> TriageResult:
    score = 0
    positive: list[str] = []
    negative: list[str] = []
    normalized_text = text.lower()
    normalized_paths = tuple(path.lower() for path in _paths(files))

    for keyword, weight in KEYWORD_SCORES:
        if keyword in normalized_text:
            score += weight
            positive.append(f"contains `{keyword}` (+{weight})")

    if any("lib/pto/transforms" in path for path in normalized_paths):
        score += 2
        positive.append("touches `lib/PTO/Transforms` (+2)")

    if any("docs/designs" in path for path in normalized_paths):
        score += 5
        positive.append("touches `docs/designs` (+5)")

    large_change = file_count > 100
    if large_change:
        score += 2
        positive.append("changed files > 100 (+2)")

    if is_branch and _has_ai_prefix(identifier):
        score -= 3
        negative.append("AI branch prefix (-3)")

    if files and all(_is_low_value_path(path) for path in normalized_paths):
        score -= 4
        negative.append("only low-value workflow/agent paths (-4)")

    classification = _classification(score)
    skip_reason = None
    if classification == "Noise":
        skip_reason = "score <= 0 after deterministic triage"
    elif classification == "Low priority":
        skip_reason = "score between 1 and 3 after deterministic triage"

    needs_codex_backfill = classification == "Investigate" and large_change
    return TriageResult(
        score=score,
        classification=classification,
        positive_signals=tuple(positive),
        negative_signals=tuple(negative),
        skip_reason=skip_reason,
        needs_codex_backfill=needs_codex_backfill,
        large_change=large_change,
    )


def _classification(score: int) -> str:
    if score >= 8:
        return "Investigate"
    if score >= 4:
        return "Watch"
    if score >= 1:
        return "Low priority"
    return "Noise"


def _paths(files: Iterable[str]) -> tuple[str, ...]:
    return tuple(_path_from_file_line(file) for file in files if _path_from_file_line(file))


def _path_from_file_line(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "\t" in value:
        return value.split("\t")[-1].strip()
    match = re.match(r"^(?:added|modified|removed|renamed|changed)\s+(.+?)\s+\(\+", value)
    if match:
        return match.group(1).strip()
    return value


def _has_ai_prefix(ref: str) -> bool:
    ref = ref.lower()
    return any(ref.startswith(prefix) or f"/{prefix}" in ref for prefix in AI_PREFIXES)


def _is_low_value_path(path: str) -> bool:
    stripped = path.removeprefix("./")
    filename = stripped.rsplit("/", 1)[-1]
    return (
        any(stripped.startswith(prefix) for prefix in LOW_VALUE_PATH_PREFIXES)
        or any(filename.startswith(prefix) for prefix in LOW_VALUE_FILE_PREFIXES)
    )
