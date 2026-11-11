#!/usr/bin/env python3
"""List MLIR callees and the number of case folders containing each one."""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path, help="run directory containing cases/")
args = parser.parse_args()
cases_dir = args.run_dir / "cases"
if not cases_dir.is_dir():
    parser.error(f"cases directory not found: {cases_dir}")

# Match call @name(...) and func.call @name(...), including quoted symbols.
call = re.compile(r'(?<![\w.])(?:func\.)?call\s+@(?:"([^"\n]+)"|([\w.$-]+))\s*\(')
instances = defaultdict(set)
file_count = 0
for case_dir in sorted(cases_dir.iterdir()):
    if not case_dir.is_dir():
        continue
    for path in case_dir.rglob("*.mlir"):
        file_count += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.lstrip().startswith("//"):
                continue
            for match in call.finditer(line):
                instances[match[1] or match[2]].add(case_dir.name)

writer = csv.writer(sys.stdout)
writer.writerow(("instance", "shape_count"))
for instance, cases in sorted(instances.items()):
    writer.writerow((instance, len(cases)))
print(f"Scanned {file_count} MLIR files; found {len(instances)} unique callees.", file=sys.stderr)
