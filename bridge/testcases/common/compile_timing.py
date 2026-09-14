import json

import triton.knobs as knobs


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
    knobs.compilation.listener = _compilation_listener
