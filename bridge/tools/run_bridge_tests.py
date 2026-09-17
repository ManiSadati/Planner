#!/usr/bin/env python3

import argparse
import datetime
import os
import subprocess
from pathlib import Path


# ============================================================
# CONFIGURATIONS
# ============================================================

ASCEND_NPU_IR_ROOT = os.environ["ASCEND_NPU_IR_ROOT"]

BRIDGE_RUNNER = (
    ASCEND_NPU_IR_ROOT
    + "/bishengir/test/Integration/PTOASBridge/tools/"
    + "run_comparison_flow.sh"
)

# These kernels run if you do NOT pass --kernels.
DEFAULT_KERNELS = [
    "cube_dotproduct",
    "flash_atten",
    "flash_atten_small",
    "matmul_64",
    "matmul_257",
    "matmul_513",
    "matmul_bf16_nn",
    "matmul_f16_f32_tb",
    "matmul_f32_hf32_nn",
    "matmul_i8_i32_nn",
    "mixed_matmul_bias",
    "q_kt_matmul",
    "qk_matmul",
    "rmsnorm_256",
    "rmsnorm_1600",
    "row_softmax",
    "row_softmax_div_sum",
    "row_softmax_exp",
    "row_softmax_max_broadcast",
    "row_softmax_sanitize",
    "row_softmax_sum_broadcast",
    "vadd",
    "vadd_large",
]

# These options run if you do NOT pass --options.
DEFAULT_OPTIONS = [
    "early-ir",
    "npu-sim",
    "emit-vmi",
    "emit-vpto",
    "bridge-sim",
    "print-all",
    "bridge-print-all",
]

# These are all options that are allowed.
VALID_OPTIONS = [
    "early-ir",
    "print-all",
    "bridge-print-all",
    "npu-sim",
    "emit-vmi",
    "emit-vpto",
    "bridge-sim",
]

DEFAULT_BRIDGE_MODE = "ptodsl"


# ============================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================

def get_arguments():
    """
    Read flags such as:

        --kernels fa_01_qk fa_02_scale
        --options emit-vmi emit-vpto
        --bridge-mode direct
        --clean-build
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kernels",
        nargs="+",
        help="Use these kernels instead of DEFAULT_KERNELS",
    )

    parser.add_argument(
        "--options",
        nargs="+",
        help="Use these options instead of DEFAULT_OPTIONS",
    )

    parser.add_argument(
        "--bridge-mode",
        default=DEFAULT_BRIDGE_MODE,
        choices=[
            "ptodsl",
            "direct",
            "external-calls",
        ],
    )

    parser.add_argument(
        "--clean-build",
        action="store_true",
        help="Pass --clean-build to the bridge runner",
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failing test",
    )

    parser.add_argument(
        "--report-dir",
        help="Directory where reports should be saved",
    )

    return parser.parse_args()


# ============================================================
# CHOOSE KERNELS
# ============================================================

def choose_kernels(args):
    """
    If the user passed --kernels, use those.
    Otherwise use DEFAULT_KERNELS.
    """

    if args.kernels:
        return args.kernels

    return DEFAULT_KERNELS


# ============================================================
# CHOOSE OPTIONS
# ============================================================

def choose_options(args):
    """
    If the user passed --options, use those.
    Otherwise use DEFAULT_OPTIONS.
    """

    if args.options:
        options = args.options
    else:
        options = DEFAULT_OPTIONS

    # Make sure every requested option is valid.
    for option in options:
        if option not in VALID_OPTIONS:
            print("Unknown option:", option)
            print("Valid options:", VALID_OPTIONS)
            raise SystemExit(1)

    return options


# ============================================================
# BUILD THE COMMAND
# ============================================================

def build_command(kernel, option, args):
    """
    Build a command like:

        run_comparison_flow.sh \
            --bridge-mode ptodsl \
            emit-vmi \
            fa_01_qk
    """

    command = [
        BRIDGE_RUNNER,
    ]

    if args.clean_build:
        command.append("--clean-build")

    command += [
        "--bridge-mode",
        args.bridge_mode,
        option,
        kernel,
    ]

    return command


# ============================================================
# RUN ONE KERNEL + OPTION
# ============================================================

def classify_result(output, returncode):
    """
    Decide what kind of result this run produced.

    Possible values:
      PASS        = everything succeeded
      PARTIAL     = compiler failed, but useful dumps were captured
      SETUP_ERROR = testcase/configuration problem
      FAIL        = any other failure
    """

    text = output.lower()

    # Testcase/setup problems.
    if (
        "no triton python file with @triton.jit found" in text
        or "no such file or directory" in text
        or "not found" in text
    ):
        return "SETUP_ERROR"

    # Compiler failed, but diagnostic output was still produced.
    if (
        "compiler exited nonzero" in text
        and "pass dumps were captured" in text
    ):
        return "PARTIAL"

    # The bridge runner itself failed.
    # This catches unknown future errors too.
    if returncode != 0:
        return "FAIL"

    # Sometimes the bridge runner exits 0 even though the inner compiler failed.
    if "compiler exited nonzero" in text:
        return "FAIL"

    return "PASS"

def run_test(kernel, option, args, report_root):
    """
    Run one test, save its terminal output,
    and return a simple dictionary describing the result.
    """

    print()
    print("Running:", kernel, option)

    command = build_command(
        kernel,
        option,
        args,
    )

    print("Command:")
    print(" ".join(command))

    # Each test gets its own log directory.
    log_dir = (
        report_root
        / "logs"
        / kernel
        / option
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    terminal_log = (
        log_dir / "terminal.log"
    )

    # Actually run the bridge command.
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )

    output = process.stdout or ""

    # Save the complete output.
    terminal_log.write_text(
        output,
        encoding="utf-8",
    )

    status = classify_result(
        output,
        process.returncode,
    )

    passed = (
        status == "PASS"
    )

    if status == "PASS":
        print("PASS")

    elif status == "PARTIAL":
        print("PARTIAL")
        print(
            "Compiler exited nonzero, "
            "but diagnostic dumps were captured."
        )

    elif status == "SETUP_ERROR":
        print("SETUP ERROR")

    else:
        print("FAIL")


    # For anything that is not a full pass,
    # show the end of the terminal output.
    if status != "PASS":

        if process.returncode != 0:
            print(
                "Bridge runner exit code:",
                process.returncode,
            )

        print("Last 15 lines:")

        lines = output.splitlines()

        for line in lines[-15:]:
            print("  " + line)

    result = {
        "kernel": kernel,
        "option": option,

        # PASS / PARTIAL / SETUP_ERROR / FAIL
        "status": status,

        "passed": passed,

        "returncode": process.returncode,

        "log": str(terminal_log),

        "log_link": str(
            terminal_log.relative_to(
                report_root
            )
        ),

        "command": " ".join(command),
    }

    return result


# ============================================================
# FIND A RESULT
# ============================================================

def find_result(results, kernel, option):
    """
    Search through the results list and return the result
    for a specific kernel + option pair.
    """

    for result in results:

        if (
            result["kernel"] == kernel
            and result["option"] == option
        ):
            return result

    return None


# ============================================================
# PRINT TERMINAL SUMMARY
# ============================================================

def print_summary(kernels, options, results):
    """
    Print something like:

        fa_01_qk
          early-ir : PASS
          emit-vmi : FAIL
    """

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for kernel in kernels:

        print()
        print(kernel)

        for option in options:

            result = find_result(
                results,
                kernel,
                option,
            )

            if result is None:
                status = "NOT RUN"

            else:
                status = result["status"]

            print(
                "  ",
                option,
                ":",
                status,
            )


# ============================================================
# WRITE MARKDOWN REPORT
# ============================================================

def write_report(
    report_file,
    kernels,
    options,
    results,
    args,
):
    """
    Create results.md with:
      - summary table
      - command for each test
      - log path
      - failure output for failed tests
    """

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        # ----------------------------------------------------
        # Report header
        # ----------------------------------------------------

        f.write(
            "# PTOAS Bridge Test Results\n\n"
        )

        f.write(
            "Bridge runner: `"
            + BRIDGE_RUNNER
            + "`\n\n"
        )

        f.write(
            "Bridge mode: `"
            + args.bridge_mode
            + "`\n\n"
        )

        # ----------------------------------------------------
        # Summary table
        # ----------------------------------------------------

        f.write(
            "## Summary\n\n"
        )

        f.write(
            "**Legend:** "
            "✅ pass | "
            "❌ fail | "
            "⚠️ compiler failed but diagnostic output was captured | "
            "⛔ testcase/setup error | "
            "— not run\n\n"
        )

        f.write(
            "| Kernel |"
        )

        for option in options:
            f.write(
                " "
                + option
                + " |"
            )

        f.write("\n")

        f.write("|---|")

        for option in options:
            f.write("---|")

        f.write("\n")

        for kernel in kernels:

            f.write(
                "| `"
                + kernel
                + "` |"
            )

            for option in options:

                result = find_result(
                    results,
                    kernel,
                    option,
                )

                if result is None:
                    symbol = "—"

                elif result["status"] == "PASS":
                    symbol = "✅"

                elif result["status"] == "PARTIAL":
                    symbol = "⚠️"

                elif result["status"] == "SETUP_ERROR":
                    symbol = "⛔"

                else:
                    symbol = "❌"

                f.write(
                    " "
                    + symbol
                    + " |"
                )

            f.write("\n")

        # ----------------------------------------------------
        # Detailed result for each test
        # ----------------------------------------------------

        f.write(
            "\n# Detailed Results\n\n"
        )

        for result in results:

            f.write(
                "## "
                + result["kernel"]
                + " / "
                + result["option"]
                + "\n\n"
            )

            if result["status"] == "PASS":
                f.write(
                    "Result: **PASS**\n\n"
                )

            elif result["status"] == "PARTIAL":
                f.write(
                    "Result: **PARTIAL** "
                    "(compiler failed, diagnostic output captured)\n\n"
                )

            elif result["status"] == "SETUP_ERROR":
                f.write(
                    "Result: **SETUP ERROR**\n\n"
                )

            else:
                f.write(
                    "Result: **FAIL**\n\n"
                )

            f.write(
                "Exit code: `"
                + str(result["returncode"])
                + "`\n\n"
            )

            f.write(
                "Terminal log: [`"
                + result["log"]
                + "`]("
                + result["log_link"]
                + ")\n\n"
            )

            f.write(
                "Command:\n\n"
            )

            f.write(
                "```bash\n"
            )

            f.write(
                result["command"]
                + "\n"
            )

            f.write(
                "```\n\n"
            )

            # Include the last part of the log for failures.
            if not result["passed"]:

                log_text = Path(
                    result["log"]
                ).read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                last_lines = (
                    log_text
                    .splitlines()[-100:]
                )

                f.write(
                    "Failure output:\n\n"
                )

                f.write(
                    "```text\n"
                )

                for line in last_lines:
                    f.write(
                        line + "\n"
                    )

                f.write(
                    "```\n\n"
                )


# ============================================================
# CREATE REPORT DIRECTORY
# ============================================================

def create_report_directory(args):
    """
    Make a new timestamped directory for this run.

    Example:

        bridge-test-reports/
            run_2026-09-17_12-30-00/
    """

    timestamp = (
        datetime.datetime.now()
        .strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    if args.report_dir:

        base_dir = Path(
            args.report_dir
        )

    else:

        script_dir = (
            Path(__file__)
            .resolve()
            .parent
        )

        base_dir = (
            script_dir
            / "bridge-test-reports"
        )

    report_root = (
        base_dir
        / ("run_" + timestamp)
    )

    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return report_root


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # Read command-line arguments.
    args = get_arguments()

    # Decide what should be tested.
    kernels = choose_kernels(args)
    options = choose_options(args)

    # Create a directory to hold logs and results.
    report_root = (
        create_report_directory(args)
    )

    report_file = (
        report_root / "results.md"
    )

    print(
        "Report directory:",
        report_root,
    )

    # This list will hold every test result.
    results = []

    stop = False

    # Run every kernel against every option.
    for kernel in kernels:

        for option in options:

            result = run_test(
                kernel,
                option,
                args,
                report_root,
            )

            results.append(
                result
            )

            # Rewrite the report after every test.
            # This means you still have partial results if
            # you stop the script halfway through.
            write_report(
                report_file,
                kernels,
                options,
                results,
                args,
            )

            if (
                not result["passed"]
                and args.stop_on_failure
            ):
                stop = True
                break

        if stop:
            break

    # Show results in the terminal.
    print_summary(
        kernels,
        options,
        results,
    )

    # Write the final Markdown report.
    write_report(
        report_file,
        kernels,
        options,
        results,
        args,
    )

    print()
    print(
        "Report written to:",
        report_file,
    )


# This makes main() run when you execute the file.
if __name__ == "__main__":
    main()
