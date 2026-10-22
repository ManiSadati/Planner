# MmadL1 Unit-Flag Implementation

Branch: AscendNPU-IR `mani/mmadl1-scratch-state`, based on `75525e82`, published
as `58f887f7688ba26e17ea02548558113abb702c19`. PTOAS remains pinned at
`d36100d0`; neither native CCE templates nor submodule revisions changed.

## Scope and Native Contract

First supported producer instances:

- `mma_tile_half_to_float` -> `__pto_mma_tile_half_to_float`.
- `mma_tile_half_to_float_tb` -> `__pto_mma_tile_half_to_float_tb`.

Paired consumer:
`fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf`, with the same `__pto_` prefix.

In native `LocalMmad.cpp`, `mma_tile_core` passes `unit_flag != 0` to
`L1Mmad` with `isL1LastK=true`. Both incoming 2 and 3 therefore enable the
producer: intermediate internal K slices check (2), and the final slice checks
and sets (3). Fixpipe preserves the exact incoming value: off (0), check (2),
or check and clear (3). This is a hardware handshake, separate from the shared
software scratch selector introduced earlier on this branch.

## Implementation

Paths here are relative to AscendNPU-IR:

- `bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/MatmulRegionToPTO.cpp`
  and `MmadL1PTODSLAdapter.cpp` reuse native
  `getUnitFlagModeLibValue(rewriter)` for constants and runtime conditions.
  They forward the resulting `i8`; they do not reimplement condition priority.
- `bishengir/lib/Template/lib/RegBase/Cube/mmadl1_ptodsl.py` owns producer
  instruction dispatch inside the internal K loop, for both init and accumulate.
  Its two migrated resources append one scalar (39 arguments total) and
  advertise `pto.ptodsl.unit_flag`. Other MmadL1 resources retain their old ABI
  and unit-flag rejection.
- `nd2nz_fixpipe_ptodsl.py` owns the consumer's 0/2/3 dispatch. Only the named
  single F32 UB Fixpipe appends a scalar (9 total). Existing quantization, ReLU
  and channel-split restrictions remain.
- The three resources are regenerated offline and installed normally. Kernel
  compilation still imports MLIR, never executes the Python generator.
- New tests live under `bishengir/test/Conversion/HIVMTemplatesToPTO/` and
  `bishengir/test/Integration/PTOASBridge/template_runtime/a5/Cube/MmadL1/native-instances/test_mmadl1_unit_flags.py`.
  The shared native inspection utility now preserves the unit-flag option and
  the no-inline compatibility option during its IR replay.

## Verification Status

Build/install with `-j96`, both offline generator self-tests, and all 105 tests
across `HIVMTemplatesToPTO`, `HIVMAVEToPTOASVMI` and `HIVMToStandard` pass.
Regeneration reproduces all three checked-in MLIR resources byte-for-byte;
installed compiler/resource hashes remain stable during the completed runs.

All four configurations pass natively first and then through the bridge:

| Configuration | M/N/K per round | Flags | Native | Bridge |
| --- | --- | --- | --- | --- |
| `nn-single` | 16/16/16 | enabled | PASS | PASS |
| `tb-single` | 16/16/16 | enabled | PASS | PASS |
| `nn-split-k` | 64/64/512 | enabled | PASS | PASS |
| `nn-disabled` | 16/16/16 | disabled | PASS | PASS |

Each has three rounds per launch, two launches, fresh NaN-filled outputs, exact
target selection and `atol=0`, `rtol=0`. The bridge requires both PTODSL targets
and their flag-bearing bodies; native fallback is not accepted for either.
Native selection replay preserves the unit-flag setting and captures the last
structured HIVM boundary as well as the actual native calls.

The split-K bridge trace has six MAD instructions, two per round: XT control
words `0xa100000040100040` and `0x2180000040100040` decode as flags 2 and 3
(`(XT >> 55) & 3`). The same words were observed in the native split-K run.
L0A/L0B byte addresses alternate 0/32768 for the internal slices. This checks
the emitted instruction controls in addition to numerical results, not only
the presence of an unused flagged branch in a helper.

Evidence: `$HOME/tmp/mmadl1-unit-flags.Rovzfe/`. The active Python environment
is `$HOME/.venv/npuir-sim-3.2.2`. Its local Triton `compiler.py` needed two
option-forwarding fixes: bridge `unit_flag`, and the existing native no-inline
workaround. Original file and the small diff are preserved as
`compiler.py.before` and `triton-option-forwarding.patch` in the evidence root.
This environment change is not part of the NPU-IR Git diff. CANN itself is
unchanged; compiler/simulator targets are `Ascend910_9589`/`Ascend950PR_9589`.
Successful logs are `native-r3/run.log`, `native-rest/run.log`, and
`bridge-{smoke,tb,split-k,disabled}/run.log`. `lit-final-r2.log`,
`installed.sha256` and `bridge-split-k-mmads.txt` preserve compiler/hash/trace
evidence. Earlier `native-smoke` and `native-r2` failures were missing
compatibility options in the main compile or IR replay, not numerical failures.
No runtime jobs remain running. A final pre-push compiler recheck also passes
105/105 (`lit-pre-push.log`).

Published to Wilson's fork on 2026-10-21:
[mani/mmadl1-scratch-state](https://gitcode.com/wilsoncxfeng/AscendNPU-IR/tree/mani/mmadl1-scratch-state),
commit `58f887f7`, "Support shared MmadL1 banks and unit flags".
The NPU-IR worktree is clean and tracks that remote branch. No master push,
merge request creation, native CCE/PTOAS changes or submodule updates occurred.
The local virtualenv fixes remain outside the NPU-IR commit. This handoff
accompanies Planner's Triton-Ascend installer role and shared skill; other
pending Planner edits are not part of that focused publication.

## Remaining Boundaries

Other MmadL1/Fixpipe resources, bias/I4, non-default final sync operands and
dynamic physical MmadL1 metadata are not generalized. Runtime-dependent flag
conditions and exact consumer mode 2 have compiler coverage, not yet dedicated
numerical coverage. TB internal split-K with flags enabled, full existing
template matrices, model-level performance
and physical A5 validation remain separate gates. Native fallback is permitted
for unrelated dependencies, never as evidence that these three targets passed.

## Next Instance And Test Work

First address the separate Triton option-sharing issue below before extending
the template campaign; then use this migration order.

1. Migrate BF16 NN/TB first, since the native SDPA audit observed enabled unit
   flags there. Continue through existing TA/TA+TB, INT8 and F32 resources after
   checking each native contract. Use the same native shared scratch symbol;
   do not create one independent bank counter per datatype or instance.
2. Extend each helper and its generation probe together, regenerate its MLIR,
   and advertise the unit-flag capability only once its body and adapter agree.
   Preserve native symbol selection and the existing per-template memref ABI.
3. Cover the required Fixpipe consumers as well as producers: native MmadL1
   converts a nonzero mode to final/internal-slice controls, whereas Fixpipe
   must preserve modes 0/2/3 exactly. Unsupported present resources must still
   fail, not discard controls or silently fall back.
4. Run the complete native-supported shape/block/grid matrix per migrated
   instance. Extend targeted cases for repeated and conditional calls, caller
   accumulation, internal split-K, full-bank transitions and mixed native/PTO
   calls sharing scratch. Check exact PTODSL selection before numerical parity.
5. Add numerical checks for runtime flag conditions, consumer check-only (2)
   followed by check-and-clear (3), and TB internal split-K with flags enabled.
   Validate physical A5 core isolation and repeated launches separately.

## Separate Triton-Ascend Follow-Up

Update 2026-10-22: implemented the shared native/bridge compilation path in the
published Triton-Ascend branch
[`feature/shared-a5-native-ptoas`](https://github.com/Melikano/triton-ascend/tree/feature/shared-a5-native-ptoas)
at `461aa8f0`.
The isolated 3.6 candidate passed command/differential tests, native/bridge
vadd, one paired F16 NN unit-flag case with repeated launches, and one F32 UB
Fixpipe tail case in both flows. It is the preferred environment for new local
tests. See the [integration handoff](triton-ascend-3.6-shared-a5-2026-10-22.md)
for the shared source baseline, setup, installed patch provenance and limits. This is
not full 3.6 template-matrix or physical-A5 coverage. The audit and proposed
actions below describe the original state before that driver follow-up.

Confirmed by the locally available Melika 3.6 bridge branch `e701011bd`:
`add_stages()` selects `linalg_to_ptoas_vmi()` for the bridge and does not call
the native `linalg_to_bin_enable_npu_compile_910_95()` function. Native unit-flag
forwarding exists, but its separate bridge option builder omits it. This is not
a missing upstream unit-flag algorithm.

Before broader template validation:

- Audit every native A5 compilation option against the bridge builder,
  including synchronization/unit flags, multi-buffering, vectorization,
  precision and other explicit kernel controls. Record intentional differences
  rather than assuming every option applies identically to both backends.
- Extract shared NPU-IR compiler-option construction. The bridge should add
  `--emit-ptoas-vmi` and its output destination; do not copy the entire native
  binary-compilation function into another partial implementation.
- Keep legitimate separate steps: PTOAS invocation, binary handling, launcher
  metadata, workspace/lock resources, target validation and diagnostics.
- Test the actual emitted argument lists with True, False and None, then run a
  bounded native/bridge Python smoke. Keep tools pinned and fresh caches separate.
- Carry the local option-forwarding fixes into the maintained Triton patch/fork;
  they are currently only in this server's virtualenv, outside the NPU-IR commit.

The older installed backend accepts `Ascend910_9589`; NPU-IR explicitly defines
that exact target using `Ascend950_BaseSpec`. The simulator uses
`Ascend950PR_9589`. There was no failed A3 compile followed by a silent A5
fallback. Melika's local 3.6 `_needs_lib_call_no_inline()` recognizes only
`Ascend950...`; broaden alias handling only if those older spellings are retained.
Do not blindly rename targets without checking the installed backend's whitelist.

At this record's initial publication, the option-sharing redesign was deferred
from the MmadL1 branch. Its subsequent driver implementation and bounded parity
evidence are recorded in the dated integration handoff above.
