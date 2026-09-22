import csv
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run_q2_q3_hardware_tests.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("hardware_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunnerArtifactTests(unittest.TestCase):
    def test_runner_module_exists(self):
        self.assertTrue(RUNNER.is_file(), f"missing runner: {RUNNER}")


class RunnerBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_extract_shape_uses_pytest_parameter_id(self):
        self.assertEqual(
            self.runner.extract_shape(
                "tests/fla_perf/test_chunk.py::test_chunk_bwd[B2-H8-T1024-D64]"
            ),
            "B2-H8-T1024-D64",
        )
        self.assertEqual(
            self.runner.extract_shape("tests/fla_perf/test_chunk.py::test_chunk_bwd"),
            "unknown",
        )

    def test_collection_parser_ignores_summaries_and_noise(self):
        output = """
tests/fla_perf/test_a.py::test_one[B1-T64]
tests/fla_perf/test_a.py::test_two

2 tests collected in 0.04s
"""
        self.assertEqual(
            self.runner.parse_collected_nodeids(output),
            [
                "tests/fla_perf/test_a.py::test_one[B1-T64]",
                "tests/fla_perf/test_a.py::test_two",
            ],
        )

    def test_patterns_match_full_case_name_with_or_semantics(self):
        name = "q3::tests/fbgemm_perf/test_jagged.py::test_sum[B4-D128]"
        self.assertTrue(self.runner.matches_patterns(name, ["*jagged*", "*fla*"]))
        self.assertFalse(self.runner.matches_patterns(name, ["*attention*", "*conv*"]))
        self.assertTrue(self.runner.matches_patterns(name, []))

    def test_junit_status_mapping_covers_pass_fail_and_skip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            passed = temp / "passed.xml"
            failed = temp / "failed.xml"
            skipped = temp / "skipped.xml"
            passed.write_text(
                '<testsuite><testcase classname="x" name="ok"/></testsuite>',
                encoding="utf-8",
            )
            failed.write_text(
                '<testsuite><testcase classname="x" name="bad">'
                '<failure message="assert values match">details</failure>'
                '</testcase></testsuite>',
                encoding="utf-8",
            )
            skipped.write_text(
                '<testsuite><testcase classname="x" name="skip">'
                '<skipped message="unsupported shape"/>'
                '</testcase></testsuite>',
                encoding="utf-8",
            )

            self.assertEqual(
                self.runner.parse_junit_report(passed, 0), ("passed", "")
            )
            self.assertEqual(
                self.runner.parse_junit_report(failed, 1),
                ("failed", "assert values match"),
            )
            self.assertEqual(
                self.runner.parse_junit_report(skipped, 0),
                ("skipped", "unsupported shape"),
            )

    def test_reduced_log_keeps_diagnostics_and_drops_progress_noise(self):
        raw = """
collecting ...
  1%|#         | 1/100 [00:01<01:39, 1.00it/s]
Traceback (most recent call last):
  File \"kernel.py\", line 42, in run
    compile_kernel()
RuntimeError: ACL_ERROR_RT_DEVICE_TASK_ABORT
================ 1 failed in 2.0s ================
"""
        reason, reduced = self.runner.reduce_failure_log(
            raw_text=raw,
            testcase_name="q2::tests/test_kernel.py::test_kernel[B1]",
            shape="B1",
            command=["python3", "-m", "pytest", "nodeid"],
            exit_code=1,
            raw_log=Path("cases/case/raw.log"),
        )
        self.assertEqual(reason, "RuntimeError: ACL_ERROR_RT_DEVICE_TASK_ABORT")
        self.assertIn("Traceback (most recent call last):", reduced)
        self.assertIn("File \"kernel.py\", line 42", reduced)
        self.assertIn("exit_code: 1", reduced)
        self.assertNotIn("1%|#", reduced)
        self.assertNotIn("1 failed in", reduced)

    def test_reduced_log_is_conservative_when_no_diagnostic_is_found(self):
        reason, reduced = self.runner.reduce_failure_log(
            raw_text="ordinary output only\n",
            testcase_name="q3::tests/test_kernel.py::test_kernel",
            shape="unknown",
            command=["python3", "-m", "pytest", "nodeid"],
            exit_code=3,
            raw_log=Path("cases/case/raw.log"),
        )
        self.assertEqual(
            reason,
            "Diagnostic extraction inconclusive; see reduced failure log",
        )
        self.assertIn("ordinary output only", reduced)

    def test_worker_pool_bounds_parallelism_and_continues_after_failure(self):
        active = 0
        maximum = 0
        lock = threading.Lock()

        def worker(item):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.03)
                if item == 2:
                    raise RuntimeError("case failed")
                return f"ok-{item}"
            finally:
                with lock:
                    active -= 1

        reported = []
        results = self.runner.run_worker_pool(
            list(range(6)),
            jobs=2,
            worker=worker,
            on_error=lambda item, error: f"error-{item}:{error}",
            on_result=reported.append,
        )
        self.assertEqual(maximum, 2)
        self.assertEqual(
            results,
            ["ok-0", "ok-1", "error-2:case failed", "ok-3", "ok-4", "ok-5"],
        )
        self.assertEqual(sorted(reported), sorted(results))

    def test_process_timeout_returns_without_waiting_for_hung_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            outcome = self.runner.run_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=temp,
                env={},
                timeout_seconds=0.1,
                log_path=temp / "raw.log",
            )
        self.assertTrue(outcome.timed_out)
        self.assertIsNotNone(outcome.returncode)
        self.assertLess(outcome.duration_seconds, 5)

    def test_reports_have_required_machine_readable_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            results = [
                self.runner.TestResult(
                    testcase_name="q2::tests/test_a.py::test_a[B1]",
                    shape="B1",
                    status="failed",
                    reason="RuntimeError: boom",
                    command=["python3", "-m", "pytest", "node"],
                    exit_code=1,
                    duration_seconds=1.25,
                    raw_log="cases/a/raw.log",
                    reduced_log="cases/a/failure.log",
                )
            ]
            self.runner.write_reports(results, output)

            with (output / "results.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))

        self.assertEqual(
            list(rows[0]), ["testcase_name", "shape", "status", "reason"]
        )
        self.assertEqual(rows[0]["shape"], "B1")
        self.assertEqual(payload[0]["status"], "failed")
        self.assertEqual(payload[0]["exit_code"], 1)

    def test_q2_commands_preserve_import_paths_and_gpu_perf_exclusion(self):
        root = Path("/bench/Q2TritonKernel-main")
        suite = self.runner.SUITES[0]
        collection = self.runner.collection_command(suite, root, "/venv/python3")
        self.assertIn(
            "pythonpath=/bench/Q2TritonKernel-main/src/kernels /bench/Q2TritonKernel-main",
            collection,
        )
        self.assertEqual(
            collection[-2:],
            ["--ignore", "tests/fla_perf/test_causal_conv1d_bwd_gpu_perf.py"],
        )

        case = self.runner.CollectedCase(
            suite=suite,
            repo_root=root,
            nodeid="tests/fla_perf/test_chunk.py::test_chunk[B1-T64]",
            ordinal=1,
        )
        command = self.runner.case_command(
            case,
            junit_path=Path("/out/junit.xml"),
            cache_dir=Path("/out/cache"),
            base_temp=Path("/out/tmp"),
            python_executable="/venv/python3",
        )
        self.assertIn(
            "pythonpath=/bench/Q2TritonKernel-main/src/kernels /bench/Q2TritonKernel-main",
            command,
        )
        self.assertEqual(
            command[-1],
            "/bench/Q2TritonKernel-main/tests/fla_perf/test_chunk.py::test_chunk[B1-T64]",
        )


if __name__ == "__main__":
    unittest.main()
