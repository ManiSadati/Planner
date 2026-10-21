#!/usr/bin/env python3
"""Replay one native NPU-IR compilation; Python standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True,
                        help="Local bishengir-compile executable or PATH name")
    parser.add_argument("--cann-root", type=Path,
                        help="Optional local CANN root containing set_env.sh")
    parser.add_argument("--target", help="Override recorded target; recorded in result")
    parser.add_argument("--output", type=Path,
                        help="New output directory; default: a fresh system temporary directory")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without running tools or creating output")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    compiler = shutil.which(args.compiler)
    if not compiler:
        parser.error("compiler not found or not executable: " + args.compiler)
    compiler = Path(compiler).resolve()
    case = json.loads((ROOT / "case.json").read_text())
    source = ROOT / case["input_file"]
    if sha256(source) != case["input_sha256"]:
        parser.error("bundled input hash does not match case.json")
    env = os.environ.copy()
    # Both variables can enable the bridge in the historical fork.
    for key in ("BISHENGIR_ENABLE_PTOAS_BRIDGE", "BISHENGIR_PTOAS_BRIDGE_MODE"):
        env.pop(key, None)
    env.update(TRITON_ASCEND_COMPILE_FLOW="npuir", FALLBACK="0")
    paths = [compiler.parent]
    cann = args.cann_root.expanduser().resolve() if args.cann_root else None
    if cann:
        if not (cann / "set_env.sh").is_file():
            parser.error("--cann-root must contain set_env.sh")
        paths += [cann / "tools/bishengir/bin", cann / "tools/ccec_compiler/bin",
                  cann / "tools/bisheng_compiler/bin", cann / "bin"]
    env["PATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get("PATH", "")

    def launch(command):
        if not cann:
            return command
        # Paths are positional arguments, never interpolated into shell code.
        # Clear bridge flags again in case set_env.sh changed them.
        return ["bash", "-c",
                'source "$1/set_env.sh" || exit $?; shift; '
                'unset BISHENGIR_ENABLE_PTOAS_BRIDGE BISHENGIR_PTOAS_BRIDGE_MODE; '
                'exec "$@"', "npuir-repro", str(cann)] + command

    flags = list(case["compiler_flags"])
    if args.target:
        flags = ["--target=" + args.target if x.startswith("--target=") else x
                 for x in flags]
    if args.dry_run:
        command = [str(compiler), str(source)] + flags + ["-o", "<OUTPUT>/kernel"]
        print(shlex.join(launch(command)))
        return 0
    if args.output:
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="q2-native-repro-"))
    for folder in ("tmp", "cache", "dump", "logs"):
        (output / folder).mkdir()
    for key, folder in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "cache"),
                        ("TRITON_DUMP_DIR", "dump"), ("ASCEND_PROCESS_LOG_PATH", "logs")):
        env[key] = str(output / folder)
    shutil.copyfile(source, output / source.name)
    shutil.copyfile(ROOT / "case.json", output / "case.json")
    command = [str(compiler), str(output / source.name)] + flags + ["-o", str(output / "kernel")]
    actual_command = launch(command)
    (output / "command.json").write_text(json.dumps(actual_command, indent=2) + "\n")
    (output / "command.txt").write_text(shlex.join(actual_command) + "\n")
    try:
        version = subprocess.run(launch([str(compiler), "--version"]), env=env,
                                 cwd=output, capture_output=True, text=True, timeout=15)
        version_text = version.stdout + version.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        version_text = "unavailable: " + str(error)
    libraries = compiler.parent.parent / "lib"
    provenance = {"compiler": str(compiler), "compiler_sha256": sha256(compiler),
                  "compiler_version": version_text,
                  "cann_root": str(cann) if cann else "inherited environment",
                  "input_sha256": sha256(source), "bridge_enabled": False,
                  "target_overridden": bool(args.target),
                  "adjacent_c310_library_hashes": {
                      p.name: sha256(p) for p in sorted(libraries.glob("meta_op.*.c310.bc"))},
                  "library_hash_scope": "Adjacent install files; not proof of which backend libraries were linked."}
    (output / "toolchain.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Evidence:", output, flush=True)
    started = time.monotonic()
    interrupted = None
    with (output / "compile.log").open("w") as log:
        try:
            process = subprocess.Popen(actual_command, cwd=output, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except OSError as error:
            log.write(str(error) + "\n")
            code = 127
        else:
            try:
                code = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                interrupted = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                code = 124 if interrupted == "timeout" else 130
    text = (output / "compile.log").read_text(errors="replace")
    undefined = sorted(set(re.findall(r"undefined symbol:\s*([\w.$]+)", text)))
    expected = sorted(set(undefined) & set(case["expected_unresolved_symbols"]))
    outcome = (interrupted or ("linked_successfully" if code == 0 else
               "reproduced_missing_template_link_failure" if expected else "other_failure"))
    result = {"outcome": outcome, "compiler_exit_code": code,
              "elapsed_seconds": time.monotonic() - started,
              "undefined_symbols": undefined, "expected_undefined_symbols_seen": expected,
              "native_call_symbols_seen_in_dumps": sorted(set(re.findall(
                  r"\b(?:func\.)?call\s+@([\w.$]+)", text))),
              "before_HIVMToStandard_marker_seen": bool(re.search(
                  r"IR Dump Before[^\n]*convert-hivm-to-std", text)),
              "after_HIVMToStandard_marker_seen": bool(re.search(
                  r"IR Dump After[^\n]*convert-hivm-to-std", text)),
              "kernel_execution_performed": False,
              "limitations": "Missing dumps or earlier failures do not establish template absence. Success may reflect fixed libraries or changed lowering."}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "exit-code.txt").write_text(str(code) + "\n")
    print("Outcome:", outcome, "(compiler exit", str(code) + ")")
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
