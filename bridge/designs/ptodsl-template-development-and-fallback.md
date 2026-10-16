# PTODSL Template Development and Fallback

Updated: 2026-10-16. Implementation branch: `mani/generic-template-fallback`
in AscendNPU-IR and its PTOAS submodule. This records the branch under review,
not a claim that the changes are already merged or published.

## Current Flow

The normal entry point is a Triton Python file. With the patched Triton backend:

```text
python kernel.py
  -> Triton / TTAdapter MLIR
  -> NPU-IR structured HIVM operations
  -> PTODSL selection and existing argument adapters
  -> native HIVM/HIVMAVE-to-standard lowering for remaining operations
  -> exact-symbol PTODSL call resolution
  -> PTOAS VMI, then PTOAS object emission
  -> launch and compare results
```

On this server, run that Python file under `msprof op simulator`. Do not use
`run_comparison_flow.sh` as the template runtime-test entry point. It remains
legacy fixture tooling and can still be useful for inspecting compiler stages.

Selection reuses NPU-IR's native instance name. For example:

```text
hivm.hir.mmadL1
  native identity: mma_tile_half_to_float
  PTODSL identity: __pto_mma_tile_half_to_float
```

There are two integration points today, not one universal adapter:

1. Existing structured MmadL1 and Vector adapters run before
   `convert-hivm-to-std` when an exact-name resource is available. They adapt
   arguments and call the imported PTO implementation.
2. After native HIVM/HIVMAVE-to-standard lowering, the generic resolver checks
   remaining native library declarations marked `llvm.emit_c_interface`.
   It looks up `__pto_<native-symbol>`, imports the selected definition and
   dependencies, and retargets the call. Its entry signature must match the
   native declaration exactly; this resolver does not invent memref adapters.

A missing registry entry or installed resource preserves the native call.
A present resource that cannot parse, lacks its entry definition, or has an
incompatible ABI is an error. It must not silently fall back.

## Writing an Instance

1. Establish the native instance name and behavior from NPU-IR's source and
   native-flow IR. Preserve datatype, transpose, padding, layout, event,
   initialization, and accumulation semantics; names alone do not prove parity.
2. Author the implementation in PTODSL Python. For MmadL1 the current source is
   `bishengir/lib/Template/lib/RegBase/Cube/mmadl1_ptodsl.py`, not the historical
   combined `nd2nz_mmadl1_64_ptodsl.py` file.
3. Generate the standalone helper MLIR offline. Its exported identity should
   be `__pto_` followed by the exact native instance name. The filename can
   retain a descriptive specialization name: function identity drives lookup.
4. Supply the instance's argument adapter if necessary. Existing MmadL1 helpers
   receive pointers plus descriptor fields through a structured C++ adapter.
   A new helper discovered only by the generic resolver needs a native-ABI
   entry wrapper, or an explicitly implemented structured adapter.
5. Add the MLIR resource to the appropriate install list in
   `bishengir/lib/Conversion/HIVMTemplatesToPTO/CMakeLists.txt`. CMake copies the
   resources and generates the symbol registry from their `__pto_` definitions.
   Regenerate/rebuild and install; do not copy one new file into an old install
   and expect the compiled registry to discover it automatically.
6. Add compile coverage for selection and ABI handling, then runtime cases
   covering native-supported tensor shapes and block configurations. Verify
   both the selected symbol and numerical results. A passing result produced
   only by native fallback does not validate the PTODSL instance.

Example offline generation, in an environment able to load the matching PTOAS
Python bindings:

```bash
export ASCEND_NPU_IR_ROOT="$HOME/AscendNPU-IR"
export PYTHONPATH="$ASCEND_NPU_IR_ROOT/build-ptoas/python:$ASCEND_NPU_IR_ROOT/third-party/ptoas/ptodsl${PYTHONPATH:+:$PYTHONPATH}"
CUBE="$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube"
python "$CUBE/mmadl1_ptodsl.py" --emit-instantiation mmadl1_f16_f32_nn \
  > "$CUBE/PTODSL/instantiations/mmadl1_f16_f32_nn.mlir"
```

Rebuild the compiler/registry, then install the compiler and resources:

```bash
cmake --build "$ASCEND_NPU_IR_ROOT/build" \
  --target bishengir-compile bishengir-opt -j96
cmake --build "$ASCEND_NPU_IR_ROOT/build" \
  --target tools/bishengir/install-bishengir-publish-products -j96
```

This is incremental; unchanged LLVM sources are not rebuilt. If the PTOAS
submodule source changed too, build/install that tool separately:

```bash
cmake --build "$ASCEND_NPU_IR_ROOT/build-ptoas" --target install -j96
```

Normal `bishengir-compile` does not execute this generator, start a daemon, or
run a PTODSL Python subprocess. It parses installed MLIR and clones definitions
into the current module. Native fallback is different: PTOAS uses the existing
`llvm-link --only-needed` mechanism on the native CCE bitcode bundle.

## Memrefs and Modes

In `ptodsl` and `external-calls` modes, native calls retain their memrefs,
including runtime sizes, strides, offsets, casts, and subviews. HIVM memory-space
attributes become PTO attributes; the descriptor information is not discarded
and reconstructed at each call. PTO consumers extract addresses separately,
including the view offset where applicable.

The kernel launch ABI remains raw-pointer based. Opaque rank-one launch
carriers do not supply tensor extents: reinterpret views provide their real
metadata. This does not add arbitrary missing dimensions to a raw pointer.

| Setting | Behavior |
| --- | --- |
| `BISHENGIR_PTOAS_BRIDGE_MODE=ptodsl` | Preferred bridge mode: exact-name PTODSL implementations plus native calls where resources are missing. Set this explicitly for Python runs. |
| `BISHENGIR_PTOAS_BRIDGE_MODE=direct` | Legacy guarded C++ rewrites, such as a simple `hivm.hir.load` directly becoming `pto.mte_gm_ub`. Not the uniform template-development path. |
| `BISHENGIR_PTOAS_BRIDGE_MODE=external-calls` | Compatibility/reference mode: skip early PTODSL/template substitution and preserve native CCE template calls through VMI. |
| `FALLBACK=1` | Allow PTOAS to selectively link unresolved native calls from the matching AIC/AIV bitcode bundle. |
| `FALLBACK=0` or unset | Default: reject remaining native calls during object emission. VMI generation alone can still succeed. |

Vadd currently uses native `load_gm_to_ubuf_1d_float` and
`store_ubuf_to_gm_1d_float` templates in `ptodsl` mode, while its addition
becomes `pto.vmi.vadd`. The direct DMA rewrite was not deleted; it is no longer
an unconditional shortcut around exact-name selection in PTODSL mode.

Future DMA implementations should export
`__pto_load_gm_to_ubuf_1d_float` and `__pto_store_ubuf_to_gm_1d_float`.
Their simple paths may reduce to one PTO instruction, but their full contracts
must still preserve padding, tails, cache policy, and other native behavior.
Legacy-name ND2NZ/Fixpipe resources likewise do not count as exact matches.

## Python Simulator Setup

Use the existing patched Triton environment. The runs below were validated
with `$HOME/.venv/npuir-sim-3.2.2`; the older `npuir-sim-system` environment on
this account did not have the required compile-flow hook when inspected.
Do not assume the package version alone proves that the patch is installed.
See [the patch instructions](../patches/triton-ascend-3.2.2-ptoas/README.md).

Set `CANN_ROOT` to your installed CANN 9.1 directory before this block:

```bash
cd "$HOME/Planner"
export CANN_ROOT="${CANN_ROOT:?Set CANN_ROOT to your CANN installation}"
export ASCEND_NPU_IR_ROOT="$HOME/AscendNPU-IR"
export NPU_IR_ROOT="$ASCEND_NPU_IR_ROOT"
export NPU_IR_SIM_VENV="$HOME/.venv/npuir-sim-3.2.2"
source NPUIR/tools/source_npuir_simulator_env.sh

export PATH="$NPU_IR_SIM_VENV/bin:$CANN_ROOT/tools/bisheng_compiler/bin:$CANN_ROOT/tools/profiler/bin:$PATH"
export LD_LIBRARY_PATH="$ASCEND_NPU_IR_ROOT/build-ptoas-llvm/lib:$ASCEND_NPU_IR_ROOT/build-ptoas/lib:${LD_LIBRARY_PATH:-}"
export TRITON_NPU_COMPILER_PATH="$ASCEND_NPU_IR_ROOT/build/install"
export TRITON_PTOAS_PATH="$ASCEND_NPU_IR_ROOT/build-ptoas/install/bin/ptoas"
export PTOAS_AICORE_LL_MODULE="$ASCEND_NPU_IR_ROOT/build/install/lib/meta_op.aic.c310.bc"

export TRITON_ASCEND_COMPILE_FLOW=ptoas
export BISHENGIR_ENABLE_PTOAS_BRIDGE=1
export BISHENGIR_PTOAS_BRIDGE_MODE=ptodsl
export FALLBACK=1
export TRITON_ASCEND_ARCH=Ascend910_9589
export TRITON_PRINT_COMPILE_COMMAND=1
export TRITON_KERNEL_DUMP=1
export PYTHONUNBUFFERED=1
ulimit -Sn 65536
```

The compiler target spelling above is the validated local Triton spelling;
the profiler target below is `Ascend950PR_9589`. This is not a switch to A3.
Use NPU-IR's current compiler and the pinned PTOAS binary, not the bundled
Triton compiler copies or an unrelated PTOAS checkout. `hivmc-a5` should come
from CANN. AIV fallback derives the sibling `meta_op.aiv.c310.bc` from the AIC
path; `PTOAS_AIVECTOR_LL_MODULE` can explicitly override it.

Create a new private output directory for each run, especially when changing
flow or installed template resources:

```bash
mkdir -p "$HOME/.cache/ptoas-template-tests"
chmod 700 "$HOME/.cache/ptoas-template-tests"
export TEST_OUT="$(mktemp -d "$HOME/.cache/ptoas-template-tests/smoke.XXXXXX")"
mkdir -p "$TEST_OUT"/{cache,dump,logs,profile,tmp}
chmod 700 "$TEST_OUT" "$TEST_OUT"/{cache,dump,logs,profile,tmp}
export TRITON_CACHE_DIR="$TEST_OUT/cache"
export TRITON_DUMP_DIR="$TEST_OUT/dump"
export ASCEND_PROCESS_LOG_PATH="$TEST_OUT/logs"
export TMPDIR="$TEST_OUT/tmp"
```

Run the small vadd directly from its Python file:

```bash
msprof op simulator --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9589 --core-id=0 \
  --output="$TEST_OUT/profile" \
  "$NPU_IR_SIM_VENV/bin/python" \
  "$ASCEND_NPU_IR_ROOT/bishengir/test/Integration/PTOASBridge/testcases/vadd/vector_add.py"
```

For the single 64x64 MmadL1 smoke case, create another fresh output directory
with the block above, then run:

```bash
export MMADL1_INSTANCE_FILTER=f16-nn
export MMADL1_SHAPE_FILTER=native-b64
msprof op simulator --kernel-name=mmadl1_native_instance_kernel \
  --soc-version=Ascend950PR_9589 --core-id=0 \
  --output="$TEST_OUT/profile" \
  "$NPU_IR_SIM_VENV/bin/python" \
  "$ASCEND_NPU_IR_ROOT/bishengir/test/Integration/PTOASBridge/template_runtime/a5/Cube/MmadL1/native-instances/test_mmadl1_native_instances.py"
```

For native NPU-IR comparison, select the following and use a fresh output/cache
directory before rerunning the same Python test:

```bash
export TRITON_ASCEND_COMPILE_FLOW=npuir
export BISHENGIR_ENABLE_PTOAS_BRIDGE=0
unset BISHENGIR_PTOAS_BRIDGE_MODE
export FALLBACK=0
```

The profiler can return zero even when its child failed. Check the Python
`PASS` / numerical comparison and the compiler diagnostics, not just the shell
exit status or a profiler timing table.

## Validation and Review

Validated on 2026-10-16:

- 91 conversion lit tests pass, including dynamic descriptors, subviews,
  shared native/PTO buffer use, exact-name resolution, and ABI rejection.
- Python f16 NN MmadL1 smoke passes for tensor/block `64x64x64`, grid `1`, and
  confirms `__pto_mma_tile_half_to_float`. ND2NZ and Fixpipe use native fallback.
- Python vadd smoke passes with maximum error `0.0` and `allclose: True`.
- `FALLBACK=0` rejects the unresolved native calls; the explicit external-calls
  mode still compiles a fat object. No large runtime sweep was rerun for this MR.

Logs: `$HOME/tmp/preserved-memref-python-final.zd2T0Z/` contains
`matmul64/run.log` and `vadd/run.log`. The compile-test log is
`$HOME/tmp/preserved-memref-lit.log`. These are local evidence paths, not
checked-in test artifacts.

The compile-test command used for this scope is:

```bash
"$ASCEND_NPU_IR_ROOT/build/bin/llvm-lit" -v \
  "$ASCEND_NPU_IR_ROOT/build/tools/bishengir/bishengir/test/Conversion/HIVMTemplatesToPTO" \
  "$ASCEND_NPU_IR_ROOT/build/tools/bishengir/bishengir/test/Conversion/HIVMAVEToPTOASVMI" \
  "$ASCEND_NPU_IR_ROOT/build/tools/bishengir/bishengir/test/Conversion/HIVMToStandard"
```

The branch is ready for MR review within this tested scope. Publication still
requires committing/pushing the PTOAS changes first, recording that commit as
NPU-IR's submodule pin, and then committing/pushing the NPU-IR branch. No claim
of full model, every template instance, or new A5 hardware validation is made.

## Future Shared-Helper Import

This is a code-review finding, not a reproduced failure of the current
templates and not a blocker for this MR. The current pre-generated Cube/Vector
resources are self-contained and do not call shared helper functions.

If two future entries both call `common_helper`, importing the second entry
would currently report a collision because the first import already cloned
that helper. Proposed follow-up:

1. Track `(resource file, original symbol)` to imported symbols for the whole
   module import, not just separately for each entry.
2. Reuse a dependency previously imported from the same resource when the
   required core context is compatible.
3. Keep errors for unrelated same-name symbols and incompatible definitions;
   do not assume equal names mean equal functions.
4. Test two entries sharing one helper and a genuine conflicting symbol.

This does not require template-body changes or a new instance-selection scheme.
For inventory and broader planning, see
[PTODSL dispatch and inventory](../planning/ptodsl-template-dispatch-and-inventory.md).
