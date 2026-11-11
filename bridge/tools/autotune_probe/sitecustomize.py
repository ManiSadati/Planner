"""Log every declared Triton autotuning candidate in the pytest raw log."""
import os
import math
import time
from threading import local

if os.environ.get("HARDWARE_ALL_AUTOTUNE") == "1":
    try:
        from triton.runtime.autotuner import Autotuner
        import triton.compiler as compiler
        import triton.compiler.compiler as compiler_impl
        import triton.runtime.jit as jit

        if os.environ.get("HARDWARE_DUMP_TEMPLATE_IR") == "1":
            import subprocess
            import tempfile
            from pathlib import Path

            original_subprocess_run = subprocess.run

            def run_with_ir(command, *args, **kwargs):
                if not isinstance(command, (list, tuple)) or Path(str(command[0])).name != "bishengir-compile":
                    return original_subprocess_run(command, *args, **kwargs)
                root = Path(os.environ["TMPDIR"]).parent / "template-ir"
                root.mkdir(exist_ok=True)
                folder = Path(tempfile.mkdtemp(dir=root, prefix="compile-"))
                command = list(command) + ["--mlir-print-ir-after-all", "--mlir-disable-threading",
                                          f"--save-temps={folder}"]
                check = kwargs.pop("check", False)
                print(f"TEMPLATE_IR {folder}", flush=True)
                result = original_subprocess_run(command, *args, check=False, **kwargs)
                output = "\n".join(
                    part.decode("utf-8", errors="replace") if isinstance(part, bytes) else str(part)
                    for part in (result.stdout, result.stderr) if part is not None
                )
                (folder / "compiler.log").write_text(output, encoding="utf-8")
                if check:
                    result.check_returncode()
                return result

            subprocess.run = run_with_ir

        original_run = jit.JITFunction.run
        seen_configs = set()
        tuning = local()
        compile_only = os.environ.get("HARDWARE_COMPILE_ONLY") == "1"

        def run(self, *args, **kwargs):
            arguments = dict(zip(self.arg_names, args))
            arguments.update({k: v for k, v in kwargs.items() if k in self.arg_names})
            constants = {
                self.arg_names[i]: arguments[self.arg_names[i]]
                for i in self.constexprs if self.arg_names[i] in arguments
            }
            options = {k: v for k, v in kwargs.items()
                       if k not in self.arg_names and k not in ("grid", "warmup")}
            identity = (self.__module__, self.__name__,
                        repr(sorted(constants.items())), repr(sorted(options.items())))
            fresh = identity not in seen_configs and not getattr(tuning, "active", False)
            seen_configs.add(identity)
            label = f"{self.__module__}.{self.__name__} constexpr={constants} options={options}"
            if fresh:
                print(f"CONFIG_RUN {label} START", flush=True)
            try:
                if compile_only:
                    kwargs["warmup"] = True
                result = original_run(self, *args, **kwargs)
                if fresh:
                    state = "COMPILED" if compile_only else "SUBMITTED"
                    print(f"CONFIG_RUN {label} {state}", flush=True)
                if compile_only and not getattr(tuning, "active", False):
                    import pytest
                    pytest.skip("Compile-only: first Triton kernel compiled; target execution skipped")
                return result
            except Exception as error:
                print(f"CONFIG_RUN {label} ERROR {error}", flush=True)
                raise

        jit.JITFunction.run = run

        original_compile = compiler.compile

        def compile_timed(*args, **kwargs):
            started = time.perf_counter()
            status = "error"
            try:
                result = original_compile(*args, **kwargs)
                status = "ok"
                return result
            finally:
                elapsed = (time.perf_counter() - started) * 1000
                print(f"AUTOTUNE compile_call_ms={elapsed:.3f} status={status}", flush=True)

        compiler.compile = compiler_impl.compile = compile_timed
        if getattr(jit, "compile", None) is original_compile:
            jit.compile = compile_timed

        original_init = Autotuner.__init__
        original_bench = Autotuner._bench

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.cache_results = False
            self.cache.clear()

        def bench(self, *args, config, **kwargs):
            label = f"CONFIG_RUN autotune {self.base_fn.__name__}: {config}"
            print(f"{label} START", flush=True)
            previous = getattr(tuning, "active", False)
            tuning.active = True
            try:
                if compile_only:
                    self.fn.run(*args, **dict(kwargs, **config.all_kwargs(), warmup=True))
                    print(f"{label} COMPILED", flush=True)
                    return
                timing = original_bench(self, *args, config=config, **kwargs)
                print(f"{label} timings_ms={timing}", flush=True)
                values = timing if isinstance(timing, (list, tuple)) else [timing]
                if all(math.isfinite(float(value)) for value in values):
                    print(f"{label} SUCCESS", flush=True)
                return timing
            except BaseException as error:
                print(f"{label} ERROR {error}", flush=True)
                raise
            finally:
                tuning.active = previous

        Autotuner.__init__ = init
        Autotuner.prune_configs = lambda self, *a, **kw: list(self.configs)
        Autotuner._bench = bench
        if compile_only:
            def compile_candidates(self, *args, **kwargs):
                errors = []
                for config in self.configs:
                    try:
                        bench(self, *args, config=config, **kwargs)
                    except Exception as error:
                        errors.append(str(error))
                if errors:
                    raise RuntimeError(f"Compile-only: {len(errors)}/{len(self.configs)} candidates failed; see raw.log")
                import pytest
                pytest.skip("Compile-only: first Triton kernel's candidates compiled; target execution skipped")

            Autotuner.run = compile_candidates
        print("AUTOTUNE tracing enabled; pruning and disk tuning cache disabled", flush=True)
    except Exception as error:
        import sys
        print(f"Autotune hook failed: {error}", file=sys.stderr, flush=True)
        os._exit(2)
