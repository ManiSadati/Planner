from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PTOAS repo checker once.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and summarize, but do not write state or reports.",
    )
    args = parser.parse_args()

    run(Path(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

