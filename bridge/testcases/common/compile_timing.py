import json
import sys

try:
    import triton.knobs as knobs
except ModuleNotFoundError:
    knobs = None


def _compilation_listener(*, src, metadata, metadata_group, times, cache_hit):
    record = {
        "flow": metadata.get("compile_flow"),
        "cache_hit": cache_hit,
        "ir_initialization_us": times.ir_initialization,
        "stages_us": dict(times.lowering_stages),
        "store_results_us": times.store_results,
        "total_us": times.total,
    }
    print("TRITON_COMPILE_TIME=" + json.dumps(record, sort_keys=True), flush=True)


def enable_compile_timing():
    if knobs is None:
        print(
            "TRITON_COMPILE_TIME_UNAVAILABLE=triton.knobs module not found",
            file=sys.stderr,
            flush=True,
        )
        return False

    if not hasattr(knobs, "compilation") or not hasattr(knobs.compilation, "listener"):
        print(
            "TRITON_COMPILE_TIME_UNAVAILABLE=triton.knobs.compilation.listener not found",
            file=sys.stderr,
            flush=True,
        )
        return False

    knobs.compilation.listener = _compilation_listener
    return True
