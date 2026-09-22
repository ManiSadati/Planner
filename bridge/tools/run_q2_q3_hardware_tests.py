#!/usr/bin/env python3
"""Run the Q2 and Q3 Triton pytest benchmarks on one A5 NPU."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar


DEFAULT_JOBS = 2
DEFAULT_TIMEOUT_SECONDS = 900
CSV_FIELDS = ("testcase_name", "shape", "status", "reason")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_LINE = re.compile(
    r"(?:\d+%\|.*\|\s*\d+/\d+|collecting \.{3}|"
    r"=+ .* (?:passed|failed|skipped).* =+)",
    re.IGNORECASE,
)
DIAGNOSTIC_LINE = re.compile(
    r"(?:traceback|\bfile \".*\", line \d+|(?:error|exception|assertionerror|"
    r"runtimeerror|typeerror|valueerror|keyerror|importerror|modulenotfounderror)\b|"
    r"failed command|exit code|compiler|compilation|mlir|triton|\bacl[_ :]|\bnpu[_ :]|"
    r"segmentation fault|core dumped|fatal|device task|kernel launch)",
    re.IGNORECASE,
)
REASON_LINE = re.compile(
    r"(?:error|exception|assertionerror|runtimeerror|typeerror|valueerror|keyerror|"
    r"importerror|modulenotfounderror|segmentation fault|fatal)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SuiteConfig:
    key: str
    repo: str
    relative_path: str
    ignores: tuple[str, ...] = ()


@dataclass(frozen=True)
class CollectedCase:
    suite: SuiteConfig
    repo_root: Path
    nodeid: str
    ordinal: int

    @property
    def testcase_name(self) -> str:
        return f"{self.suite.repo}::{self.nodeid}"


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int | None
    timed_out: bool
    duration_seconds: float


@dataclass
class TestResult:
    testcase_name: str
    shape: str
    status: str
    reason: str
    command: list[str]
    exit_code: int | None
    duration_seconds: float
    raw_log: str
    reduced_log: str


SUITES = (
    SuiteConfig(
        key="q2-fla-perf",
        repo="q2",
        relative_path="tests/fla_perf",
        ignores=("tests/fla_perf/test_causal_conv1d_bwd_gpu_perf.py",),
    ),
    SuiteConfig("q3-fbgemm-perf", "q3", "tests/fbgemm_perf"),
    SuiteConfig("q3-fla-perf", "q3", "tests/fla_perf"),
    SuiteConfig("q3-minimax-m3-perf", "q3", "tests/minimax_m3_perf"),
    SuiteConfig("q3-recsys-example-perf", "q3", "tests/recsys_example_perf"),
    SuiteConfig("q3-recsys-perf", "q3", "tests/recsys_perf"),
    SuiteConfig("q3-vllm-ascend-perf", "q3", "tests/vllm_ascend_perf"),
)

T = TypeVar("T")
R = TypeVar("R")


def extract_shape(nodeid: str) -> str:
    """Return pytest's final parameter ID, which carries benchmark shape/config."""
    match = re.search(r"\[([^\n]*)\]$", nodeid)
    return match.group(1) if match else "unknown"


def parse_collected_nodeids(output: str) -> list[str]:
    """Parse the stable node-ID lines emitted by ``pytest --collect-only -q``."""
    nodeids: list[str] = []
    for raw_line in output.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line).strip()
        if "::" not in line or line.startswith(("ERROR", "INTERNALERROR")):
            continue
        if line.startswith("<") or " " in line.split("::", 1)[0]:
            continue
        nodeids.append(line)
    return nodeids


def matches_patterns(testcase_name: str, patterns: Sequence[str]) -> bool:
    """Apply repeatable shell-style filters with OR semantics."""
    return not patterns or any(
        fnmatch.fnmatchcase(testcase_name, pattern) for pattern in patterns
    )


def _xml_kind(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _short_text(value: str | None, limit: int = 300) -> str:
    compact = " ".join((value or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def parse_junit_report(path: Path, exit_code: int | None) -> tuple[str, str]:
    """Map one-test JUnit output to the runner's four statuses."""
    if not path.is_file():
        return "failed", f"pytest exited with code {exit_code}; no JUnit report was written"
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        return "failed", f"could not parse JUnit report: {_short_text(str(error))}"

    testcase = next((item for item in root.iter() if _xml_kind(item.tag) == "testcase"), None)
    if testcase is None:
        return "failed", f"pytest exited with code {exit_code}; JUnit report has no testcase"
    for child in testcase:
        kind = _xml_kind(child.tag)
        if kind == "skipped":
            return "skipped", _short_text(child.get("message") or child.text or "skipped")
        if kind in {"failure", "error"}:
            reason = child.get("message") or child.text or kind
            return "failed", _short_text(reason)
    if exit_code == 0:
        return "passed", ""
    return "failed", f"pytest exited with code {exit_code} despite a passing JUnit testcase"


def _clean_log_lines(raw_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in ANSI_ESCAPE.sub("", raw_text).splitlines():
        # Terminal progress often rewrites one physical line with carriage returns.
        line = raw_line.split("\r")[-1].rstrip()
        if not line.strip() or PROGRESS_LINE.search(line.strip()):
            continue
        lines.append(line)
    return lines


def reduce_failure_log(
    *,
    raw_text: str,
    testcase_name: str,
    shape: str,
    command: Sequence[str],
    exit_code: int | None,
    raw_log: Path,
) -> tuple[str, str]:
    """Create a compact diagnostic excerpt without guessing at an unknown failure."""
    lines = _clean_log_lines(raw_text)
    important = {index for index, line in enumerate(lines) if DIAGNOSTIC_LINE.search(line)}
    selected: set[int] = set()
    for index in important:
        selected.update(range(max(0, index - 2), min(len(lines), index + 3)))

    if selected:
        excerpt_lines: list[str] = []
        previous = -2
        for index in sorted(selected):
            if index > previous + 1 and excerpt_lines:
                excerpt_lines.append("...")
            excerpt_lines.append(lines[index])
            previous = index
    else:
        # Retain a bounded tail so the reduced log is still useful while explicitly
        # declining to turn arbitrary output into a fabricated diagnosis.
        excerpt_lines = lines[-80:]

    reason = "Diagnostic extraction inconclusive; see reduced failure log"
    for line in reversed(lines):
        candidate = line.strip()
        if REASON_LINE.search(candidate) and not candidate.lower().startswith("traceback"):
            reason = _short_text(candidate)
            break

    header = [
        f"testcase: {testcase_name}",
        f"shape: {shape}",
        "status: failed",
        f"command: {shlex.join(list(command))}",
        f"exit_code: {exit_code}",
        f"raw_log: {raw_log}",
        "",
        "diagnostics:",
    ]
    return reason, "\n".join(header + (excerpt_lines or ["(no diagnostic output)"])) + "\n"


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    log_path: Path,
) -> ProcessOutcome:
    """Run a command in its own process group and terminate the group on timeout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    returncode: int | None = None
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {shlex.join(list(command))}\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait()
    return ProcessOutcome(returncode, timed_out, time.monotonic() - started)


def run_worker_pool(
    items: Sequence[T],
    *,
    jobs: int,
    worker: Callable[[T], R],
    on_error: Callable[[T, BaseException], R],
    on_result: Callable[[R], None] | None = None,
) -> list[R]:
    """Run a rolling bounded queue while converting per-item crashes into results."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="npu-test") as executor:
        future_to_index = {
            executor.submit(worker, item): index for index, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except BaseException as error:  # one broken testcase must not drain the queue
                results[index] = on_error(items[index], error)
            if on_result is not None:
                on_result(results[index])  # type: ignore[arg-type]
    return [item for item in results if item is not None]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_reports(results: Sequence[TestResult], output_dir: Path) -> None:
    """Write required CSV plus richer JSON and a consolidated failure excerpt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for result in results:
        writer.writerow({field: getattr(result, field) for field in CSV_FIELDS})
    _atomic_write(output_dir / "results.csv", csv_buffer.getvalue())
    _atomic_write(
        output_dir / "results.json",
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
    )

    reduced_logs: list[str] = []
    for result in results:
        if result.status not in {"failed", "timed out"} or not result.reduced_log:
            continue
        reduced_path = output_dir / result.reduced_log
        if reduced_path.is_file():
            reduced_logs.append(reduced_path.read_text(encoding="utf-8", errors="replace"))
    _atomic_write(
        output_dir / "failures.log",
        ("\n" + "=" * 80 + "\n\n").join(reduced_logs),
    )


def _python_paths(repo: str, root: Path) -> list[Path]:
    return [root / "src/kernels", root] if repo == "q2" else [root]


def collection_command(
    suite: SuiteConfig, root: Path, python_executable: str
) -> list[str]:
    pythonpath = " ".join(str(path.resolve()) for path in _python_paths(suite.repo, root))
    command = [
        python_executable,
        "-m",
        "pytest",
        "--rootdir",
        str(root.resolve()),
        "-o",
        f"pythonpath={pythonpath}",
        suite.relative_path,
        "--collect-only",
        "-q",
    ]
    for ignored in suite.ignores:
        command.extend(("--ignore", ignored))
    return command


def case_command(
    case: CollectedCase,
    *,
    junit_path: Path,
    cache_dir: Path,
    base_temp: Path,
    python_executable: str,
) -> list[str]:
    root = case.repo_root.resolve()
    file_part, *selectors = case.nodeid.split("::")
    absolute_nodeid = "::".join([str((root / file_part).resolve()), *selectors])
    pythonpath = " ".join(str(path.resolve()) for path in _python_paths(case.suite.repo, root))
    return [
        python_executable,
        "-m",
        "pytest",
        "--rootdir",
        str(root),
        "-o",
        f"pythonpath={pythonpath}",
        "-o",
        f"cache_dir={cache_dir}",
        "--basetemp",
        str(base_temp),
        "--junitxml",
        str(junit_path),
        "-sv",
        absolute_nodeid,
    ]


def _slug(value: str, ordinal: int) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[-80:].strip("_.") or "case"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{ordinal:05d}_{readable}_{digest}"


def _case_directory(output_dir: Path, case: CollectedCase) -> Path:
    return output_dir / "cases" / _slug(case.testcase_name, case.ordinal)


def _child_environment(case: CollectedCase, case_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(path.resolve()) for path in _python_paths(case.suite.repo, case.repo_root)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "NPU_DEVICE": "0",
            "TMPDIR": str(case_dir / "tmp"),
            "TMP": str(case_dir / "tmp"),
            "TEMP": str(case_dir / "tmp"),
            "TRITON_CACHE_DIR": str(case_dir / "triton-cache"),
            "TORCH_EXTENSIONS_DIR": str(case_dir / "torch-extensions"),
            "XDG_CACHE_HOME": str(case_dir / "xdg-cache"),
        }
    )
    return env


def _prepare_case_workspace(repo_root: Path, work_dir: Path) -> None:
    """Mirror repo-relative inputs while keeping top-level generated outputs isolated."""
    work_dir.mkdir(parents=True, exist_ok=True)
    excluded = {"result_dir", ".pytest_cache", "build", "out"}
    for source in repo_root.iterdir():
        if source.name in excluded:
            continue
        destination = work_dir / source.name
        if not destination.exists():
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def run_one_case(
    case: CollectedCase,
    *,
    output_dir: Path,
    timeout_seconds: float,
    python_executable: str,
) -> TestResult:
    case_dir = _case_directory(output_dir, case)
    work_dir = case_dir / "work"
    tmp_dir = case_dir / "tmp"
    for directory in (
        work_dir,
        tmp_dir,
        case_dir / "triton-cache",
        case_dir / "torch-extensions",
        case_dir / "xdg-cache",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _prepare_case_workspace(case.repo_root, work_dir)

    junit_path = case_dir / "junit.xml"
    raw_path = case_dir / "raw.log"
    reduced_path = case_dir / "failure.log"
    command = case_command(
        case,
        junit_path=junit_path,
        cache_dir=case_dir / "pytest-cache",
        base_temp=tmp_dir / "pytest",
        python_executable=python_executable,
    )
    try:
        outcome = run_process(
            command,
            cwd=work_dir,
            env=_child_environment(case, case_dir),
            timeout_seconds=timeout_seconds,
            log_path=raw_path,
        )
    except OSError as error:
        with raw_path.open("a", encoding="utf-8") as log:
            log.write(f"runner process error: {type(error).__name__}: {error}\n")
        outcome = ProcessOutcome(None, False, 0.0)
    shape = extract_shape(case.nodeid)
    if outcome.timed_out:
        status = "timed out"
        reason = f"Exceeded timeout of {timeout_seconds:g} seconds"
    else:
        status, reason = parse_junit_report(junit_path, outcome.returncode)

    raw_relative = raw_path.relative_to(output_dir).as_posix()
    reduced_relative = ""
    if status in {"failed", "timed out"}:
        extracted_reason, reduced = reduce_failure_log(
            raw_text=raw_path.read_text(encoding="utf-8", errors="replace"),
            testcase_name=case.testcase_name,
            shape=shape,
            command=command,
            exit_code=outcome.returncode,
            raw_log=Path(raw_relative),
        )
        reduced_path.write_text(reduced.replace("status: failed", f"status: {status}", 1), encoding="utf-8")
        reduced_relative = reduced_path.relative_to(output_dir).as_posix()
        if status == "failed":
            if extracted_reason.startswith("Diagnostic extraction inconclusive"):
                if reason and not reason.startswith(("pytest exited", "could not parse")):
                    reason = reason
                else:
                    reason = f"Diagnostic extraction inconclusive; see {reduced_relative}"
            else:
                reason = extracted_reason

    return TestResult(
        testcase_name=case.testcase_name,
        shape=shape,
        status=status,
        reason=reason,
        command=command,
        exit_code=outcome.returncode,
        duration_seconds=round(outcome.duration_seconds, 3),
        raw_log=raw_relative,
        reduced_log=reduced_relative,
    )


def _failure_result(
    case: CollectedCase,
    error: BaseException,
    output_dir: Path,
) -> TestResult:
    case_dir = _case_directory(output_dir, case)
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = case_dir / "raw.log"
    message = f"runner error: {type(error).__name__}: {error}"
    raw_path.write_text(message + "\n", encoding="utf-8")
    reduced_path = case_dir / "failure.log"
    relative_raw = raw_path.relative_to(output_dir)
    reason, reduced = reduce_failure_log(
        raw_text=message,
        testcase_name=case.testcase_name,
        shape=extract_shape(case.nodeid),
        command=[],
        exit_code=None,
        raw_log=relative_raw,
    )
    reduced_path.write_text(reduced, encoding="utf-8")
    return TestResult(
        testcase_name=case.testcase_name,
        shape=extract_shape(case.nodeid),
        status="failed",
        reason=reason,
        command=[],
        exit_code=None,
        duration_seconds=0.0,
        raw_log=relative_raw.as_posix(),
        reduced_log=reduced_path.relative_to(output_dir).as_posix(),
    )


def _collection_failure(
    suite: SuiteConfig,
    command: list[str],
    outcome: ProcessOutcome,
    log_path: Path,
    output_dir: Path,
) -> TestResult:
    name = f"{suite.repo}::collection::{suite.relative_path}"
    raw_relative = log_path.relative_to(output_dir)
    reason, reduced = reduce_failure_log(
        raw_text=log_path.read_text(encoding="utf-8", errors="replace"),
        testcase_name=name,
        shape="unknown",
        command=command,
        exit_code=outcome.returncode,
        raw_log=raw_relative,
    )
    reduced_path = log_path.with_name(log_path.stem + "-failure.log")
    status = "timed out" if outcome.timed_out else "failed"
    reduced = reduced.replace("status: failed", f"status: {status}", 1)
    reduced_path.write_text(reduced, encoding="utf-8")
    if reason.startswith("Diagnostic extraction inconclusive"):
        reason = f"Collection failed; see {reduced_path.relative_to(output_dir).as_posix()}"
    return TestResult(
        testcase_name=name,
        shape="unknown",
        status=status,
        reason=reason,
        command=command,
        exit_code=outcome.returncode,
        duration_seconds=round(outcome.duration_seconds, 3),
        raw_log=raw_relative.as_posix(),
        reduced_log=reduced_path.relative_to(output_dir).as_posix(),
    )


def discover_cases(
    suites: Iterable[SuiteConfig],
    roots: dict[str, Path],
    *,
    output_dir: Path,
    python_executable: str,
    timeout_seconds: float,
) -> tuple[list[CollectedCase], list[TestResult]]:
    cases: list[CollectedCase] = []
    failures: list[TestResult] = []
    collection_dir = output_dir / "collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    for suite in suites:
        root = roots[suite.repo]
        command = collection_command(suite, root, python_executable)
        log_path = collection_dir / f"{suite.key}.log"
        try:
            outcome = run_process(
                command,
                cwd=root,
                env=os.environ.copy(),
                timeout_seconds=timeout_seconds,
                log_path=log_path,
            )
        except OSError as error:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"runner collection error: {type(error).__name__}: {error}\n")
            outcome = ProcessOutcome(None, False, 0.0)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        nodeids = parse_collected_nodeids(output)
        if outcome.timed_out or outcome.returncode != 0 or not nodeids:
            failures.append(
                _collection_failure(suite, command, outcome, log_path, output_dir)
            )
            continue
        for nodeid in nodeids:
            cases.append(CollectedCase(suite, root, nodeid, len(cases) + 1))
    return cases, failures


def _env_positive_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _env_positive_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _default_root(repo: str) -> Path:
    upper = repo.upper()
    configured = os.environ.get(f"{upper}_TRITON_ROOT") or os.environ.get(f"{upper}_ROOT")
    if configured:
        return Path(configured).expanduser()
    names = (f"{upper}TritonKernel-main", f"{upper}TritonKernel")
    candidates = [Path.home() / "Workspace" / name for name in names]
    candidates.extend(Path.home() / name for name in names)
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def _parser() -> argparse.ArgumentParser:
    planner_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    parser = argparse.ArgumentParser(
        description="Run Q2/Q3 pytest benchmark cases through a bounded queue on one A5 NPU."
    )
    parser.add_argument("--repo", choices=("all", "q2", "q3"), default="all")
    parser.add_argument("--q2-root", type=Path, default=_default_root("q2"))
    parser.add_argument("--q3-root", type=Path, default=_default_root("q3"))
    parser.add_argument(
        "--jobs",
        type=int,
        default=_env_positive_int("HARDWARE_TEST_JOBS", DEFAULT_JOBS),
        help=f"maximum concurrent cases (default: {DEFAULT_JOBS}; env HARDWARE_TEST_JOBS)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_env_positive_float("HARDWARE_TEST_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
        help=f"seconds per testcase (default: {DEFAULT_TIMEOUT_SECONDS}; env HARDWARE_TEST_TIMEOUT)",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="shell-style testcase-name filter; repeat for OR matching",
    )
    parser.add_argument("--limit", type=int, help="run only the first N matching cases")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=planner_root / "bridge/out/hardware-tests" / f"run_{timestamp}",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="logical NPU device; only 0 is supported (select physical device with ASCEND_RT_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the configured hardware environment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="collect cases and print their commands without executing them",
    )
    return parser


def _validate_args(args: argparse.Namespace, roots: dict[str, Path]) -> None:
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.device != 0:
        raise ValueError(
            "tests use logical NPU 0; set ASCEND_RT_VISIBLE_DEVICES to select a physical NPU"
        )
    for repo, root in roots.items():
        if not root.is_dir():
            raise ValueError(f"{repo.upper()} root does not exist: {root}")


def _print_table(results: Sequence[TestResult]) -> None:
    headers = ("testcase_name", "shape", "status", "reason")
    rows = [[getattr(item, field) for field in headers] for item in results]
    limits = (68, 28, 10, 72)
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = min(limits[index], max(widths[index], len(str(value))))

    def render(row: Sequence[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            text = str(value)
            if len(text) > widths[index]:
                text = text[: max(0, widths[index] - 1)] + "…"
            cells.append(text.ljust(widths[index]))
        return " | ".join(cells)

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = _parser()
    except ValueError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    try:
        args = parser.parse_args(argv)
        selected_repos = ("q2", "q3") if args.repo == "all" else (args.repo,)
        roots = {
            repo: (args.q2_root if repo == "q2" else args.q3_root).expanduser().resolve()
            for repo in selected_repos
        }
        _validate_args(args, roots)
    except ValueError as error:
        parser.error(str(error))

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        parser.error(
            f"output directory is not empty: {output_dir}; choose a new --output-dir"
        )
    selected_suites = [suite for suite in SUITES if suite.repo in selected_repos]
    print(f"Collecting cases into {output_dir}")
    cases, collection_failures = discover_cases(
        selected_suites,
        roots,
        output_dir=output_dir,
        python_executable=args.python,
        timeout_seconds=args.timeout,
    )
    cases = [case for case in cases if matches_patterns(case.testcase_name, args.pattern)]
    if args.limit is not None:
        cases = cases[: args.limit]

    if args.dry_run:
        print(f"Dry run: {len(cases)} testcase(s); no testcase will execute.")
        for case in cases:
            case_dir = _case_directory(output_dir, case)
            print(
                shlex.join(
                    case_command(
                        case,
                        junit_path=case_dir / "junit.xml",
                        cache_dir=case_dir / "pytest-cache",
                        base_temp=case_dir / "tmp/pytest",
                        python_executable=args.python,
                    )
                )
            )
        if collection_failures:
            _print_table(collection_failures)
            write_reports(collection_failures, output_dir)
            return 1
        if not cases:
            print("No testcases matched.", file=sys.stderr)
            return 2
        return 0

    print(
        f"Running {len(cases)} testcase(s) with {args.jobs} worker(s), "
        f"{args.timeout:g}s timeout, logical NPU 0"
    )
    partial_results: list[TestResult] = []

    def record_completion(result: TestResult) -> None:
        partial_results.append(result)
        write_reports(collection_failures + partial_results, output_dir)
        print(f"{result.status.upper():9} {result.testcase_name}", flush=True)

    completed = run_worker_pool(
        cases,
        jobs=args.jobs,
        worker=lambda case: run_one_case(
            case,
            output_dir=output_dir,
            timeout_seconds=args.timeout,
            python_executable=args.python,
        ),
        on_error=lambda case, error: _failure_result(case, error, output_dir),
        on_result=record_completion,
    )
    results = collection_failures + completed
    write_reports(results, output_dir)
    _print_table(results)
    print(f"\nCSV:  {output_dir / 'results.csv'}")
    print(f"JSON: {output_dir / 'results.json'}")
    print(f"Failures: {output_dir / 'failures.log'}")

    if not results and not cases:
        print("No testcases matched.", file=sys.stderr)
        return 2
    return 1 if any(result.status in {"failed", "timed out"} for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
