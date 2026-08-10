from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class CheckerState:
    version: int = 1
    last_run_at: str | None = None
    repos: dict[str, Any] = field(default_factory=dict)
    github: dict[str, Any] = field(default_factory=dict)
    github_forks: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "CheckerState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            version=data.get("version", 1),
            last_run_at=data.get("last_run_at"),
            repos=data.get("repos", {}),
            github=data.get("github", {}),
            github_forks=data.get("github_forks", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "last_run_at": self.last_run_at,
            "repos": self.repos,
            "github": self.github,
            "github_forks": self.github_forks,
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
