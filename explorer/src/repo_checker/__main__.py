from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .runner import run


def _utc_since_days(days: int) -> str:
    since = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=days)
    return since.isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PTOAS repo checker once.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument(
        "--since",
        help="One-off lookback start time, as an ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        help="One-off lookback window in days, measured from now in UTC.",
    )
    parser.add_argument(
        "--preserve-state",
        action="store_true",
        help="Write reports but do not update the normal daily state cursors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and summarize, but do not write state or reports.",
    )
    args = parser.parse_args()

    if args.since and args.since_days is not None:
        parser.error("Use only one of --since or --since-days.")
    if args.since_days is not None and args.since_days < 1:
        parser.error("--since-days must be positive.")

    since = args.since or (_utc_since_days(args.since_days) if args.since_days else None)
    run(
        Path(args.config),
        dry_run=args.dry_run,
        since_override=since,
        preserve_state=args.preserve_state,
    )


if __name__ == "__main__":
    main()
