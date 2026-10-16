# PTODSL Template Dispatch and NPU-IR Inventory

Last reviewed: 2026-10-16

This document records how PTODSL templates are matched today, how native
NPU-IR dispatches CCE/AscendC templates, the current A5 template inventory,
and the design to use as PTODSL coverage grows.

For the current developer workflow, exact selection/adapter contract, Python
simulator exports, and MR scope, use
[PTODSL template development and fallback](../designs/ptodsl-template-development-and-fallback.md).

## Generic Selection And Fallback Update (2026-10-16)

Implementation is on `mani/generic-template-fallback` in both
AscendNPU-IR and its PTOAS submodule. The scope is exact-symbol selection and
native fallback; new PTODSL memref adapters and template bodies are separate
instance-development work.

- Generate the resource registry from the installed MLIR function names.
- Reuse native NPU-IR call materialization and look up
  `__pto_<native-symbol>` for any remaining native library call.
- Import and retarget exact matches; a registered implementation with a wrong
  ABI must fail explicitly rather than fall back.
- Keep unmatched calls with their native arguments and descriptor ABI.
- Apply `FALLBACK=0/1` to both AIC and AIV object emission, using the existing
  selective native-bitcode linker and the corresponding core bundle.
- Keep existing working structured adapters without adding new argument
  transformations in the generic resolver.

Legacy ND2NZ, Fixpipe, and cumsum helpers whose names differ from the exact
native symbol now use native fallback. Instance authors must supply matching
entry points before those implementations are selected by the new policy.
No native CCE bitcode changes or large runtime tests are part of this task.

Compile validation: 91 conversion tests pass. The small `matmul_64` and `vadd`
fixtures both reach VMI and VPTO and produce fat objects with `FALLBACK=1`.
Both reject their remaining native calls with `FALLBACK=0`. A mixed AIC/AIV
compile also produces a fat object and links only the names used by each
device module. These are compile/link checks, not numerical or performance
results.

Small Python simulator follow-up (not the comparison runner):

- `vector_add.py` passes through the patched Triton backend with maximum error
  `0.0` and `allclose: True`. Its two loads and store use native
  `load_gm_to_ubuf_1d_float` / `store_ubuf_to_gm_1d_float` fallback; addition
  lowers to `pto.vmi.vadd`.
- The initial single `f16-nn` / `native-b64` MmadL1 attempt exposed the hybrid
  fallback's static-descriptor reconstruction restriction. The VMI pass now
  preserves interior memrefs and their runtime sizes, strides, offsets, casts,
  and subviews. Native calls no longer reconstruct descriptors from PTO
  pointers. Only memory-space attributes change from HIVM to PTO.
- PTO consumers extract addresses separately; non-identity views include their
  runtime offset. Kernel entry points retain the raw-pointer launch ABI. An
  opaque rank-one launch carrier supplies only its base buffer to reinterpret
  views, which retain their explicit NPU-IR-computed metadata.
- After this fix, the `f16-nn` / `native-b64` Python test passes numerical
  comparison for shape/block `64x64x64`, grid `1`, with selected symbol
  `__pto_mma_tile_half_to_float`. ND2NZ and Fixpipe use native CCE fallback.
- The post-fix `vector_add.py` Python simulator run also passes with maximum
  error `0.0`. Both checks use the existing patched `npuir-sim-3.2.2` venv and
  invoke the Python files under `msprof op simulator`, not the comparison runner.
- The external-calls mode also compiles to a fat object. `FALLBACK=0` still
  rejects unresolved native calls explicitly. No native bitcode or PTODSL
  template body changes were needed for memref preservation.

Logs: `$HOME/tmp/generic-template-fallback-runtime.vgQ5JH/` under
`python-host-vadd/run.log` and `python-mmadl1-f16-nn/run.log`.
Successful post-fix Python logs are under
`$HOME/tmp/preserved-memref-python-final.zd2T0Z/`, in
`matmul64/run.log` and `vadd/run.log`.

Source references in AscendNPU-IR:
`bishengir/lib/Conversion/HIVMTemplatesToPTO/PTODSLTemplateRegistry.cpp`,
`bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp`, and
`third-party/ptoas/tools/ptoas/ObjectEmission.cpp`.

### Review Follow-up: Shared Helpers

The current resources are self-contained. A future pair of entries sharing a
helper would hit the generic importer's collision check on the second import.
This was identified by code inspection, not by an existing runtime failure,
and is not a blocker for this MR. Track imports by resource and symbol, reuse
compatible dependencies already imported from that resource, and retain
errors for unrelated collisions. Add shared-dependency and real-collision
tests when implementing that follow-up. The full example and proposal are in
the developer guide linked above.

## Current PTODSL Template Flow

The PTODSL path uses pre-generated MLIR helpers. Existing MmadL1 and Vector
adapters run before native call lowering when an exact resource is available.
The generic resolver then checks the native library calls for all remaining
families. A missing exact entry preserves the native call for explicit fallback.

### 1. Define explicit instantiations in Python

The PTODSL source defines `SPECIALIZATIONS` and `EXPLICIT_INSTANTIATIONS` in:

```text
$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/mmadl1_ptodsl.py
```

Current MMAD variants include:

- `mmadl1_f16_f32_nn`
- `mmadl1_f16_f32_tb`
- `mmadl1_bf16_f32_nn`
- `mmadl1_i8_i32_nn`
- `mmadl1_f32_f32_hf32_nn`

Python is run offline to generate one standalone MLIR helper for each explicit
instantiation. It is not run during normal NPU-IR compilation.

Each generated helper receives:

- a deterministic symbol such as `__pto_mma_tile_half_to_float_tb`
- a `pto.ptodsl.instantiation` attribute
- a private function body containing PTO dialect operations

### 2. Install the generated MLIR resources

The generated files are checked in under:

```text
$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/PTODSL/instantiations/
```

`HIVMTemplatesToPTO/CMakeLists.txt` manually lists and installs these files
under:

```text
lib/bishengir/ptodsl/cube/
```

### 3. Import the helper definitions

`PTODSLTemplateImporter.cpp` contains another manually maintained table with:

- MLIR filename
- helper symbol
- instantiation identity

For MmadL1, `select-hivm-ptodsl-templates` first records the exact native and
PTODSL symbols. The converter builds a deduplicated required-symbol set and
imports only the selected MmadL1 resources. The importer parses each required
file once, validates its symbol and instantiation attribute, clones its
function into the kernel module, and marks it as an AIC function. The generic
resolver performs exact resource lookup for remaining native calls, without
importing all ND2NZ or Fixpipe helpers by family.

This is MLIR-level importing or linking: the helper definition and its call are
placed in the same MLIR module. It is not external LLVM bitcode linking.

### 4. Match an HIVM operation

For `hivm.hir.mmadL1`, the selection pass calls native NPU-IR's
`getOpLibraryCallName()` and performs exact registry lookup. The later
`matchMmadL1Contract()` validates:

- input datatype
- accumulator datatype
- physical memref shapes
- memory spaces
- synchronization operands
- supported runtime M/K/N range

Native symbol selection determines the exact helper entry. For example:

```text
f16 input + f32 accumulator + transpose B
    -> __pto_mma_tile_half_to_float_tb
```

The rewrite then replaces the original operation with a call:

```mlir
func.call @__pto_mma_tile_half_to_float_tb(...)
```

Runtime values such as actual M/K/N, pointers, event IDs, and initialization
state remain function operands. They do not create separate explicit
instantiations.

`applyPatternsGreedily` drives the existing structured adapters. It does not
search or rank templates. Exact native names and the generated resource index
own instance selection.

## Native NPU-IR CCE/AscendC Dispatch

Native NPU-IR follows a related model, but uses external function symbols and
precompiled bitcode.

The representative `MmadL1` flow is:

```text
hivm.hir.mmadL1
    -> derive a specialized function name
    -> emit a func.call declaration and call
    -> lower the call to LLVM
    -> resolve it from an architecture/core-specific meta_op bitcode bundle
```

### Function name generation

`MmadL1Op::getOpLibraryCallName()` constructs a name from:

- source datatype
- destination datatype
- transpose A
- transpose B
- HF32 mode
- I4 mode
- optional bias datatype

For example, an f16-to-f32 operation with B transposed selects a symbol similar
to:

```text
mma_tile_half_to_float_tb
```

The specialized symbol name is effectively the native template identity.

### Conversion to a library call

`MmadL1OpToLibraryCallPattern` collects the operation operands, result buffer,
synchronization values, and unit-flag value. It then creates a private function
declaration and replaces `MmadL1` with a `func.call`.

Memref operands are canonicalized to dynamic-shape, dynamic-stride descriptor
types before the call. This matches the ABI used by the CCE template wrappers.

### Ahead-of-time C++ instantiation

The native implementation in `LocalMmad.cpp` uses C++ templates and registration
macros such as:

```cpp
REGISTER_MMA_TILE(...)
REGISTER_MMA_TILE_TA(...)
REGISTER_MMA_TILE_TB(...)
REGISTER_MMA_TILE_HF32(...)
REGISTER_MMA_TILE_BIAS(...)
```

These macros instantiate datatype, transpose, precision, and bias combinations
ahead of time. This is directly analogous to explicit PTODSL instantiation.

### Bitcode bundling and linking

The C++ template files are compiled into per-file LLVM bitcode. CMake combines
them with `llvm-link` into architecture/core bundles such as:

```text
meta_op.aic.c310.bc
meta_op.aiv.c310.bc
meta_op.mix.aic.c310.bc
meta_op.mix.aiv.c310.bc
```

The compiler attaches the correct bundle based on target architecture and core
type. It does not link a separate small bitcode file for every operation. The
larger bundle contains all implementations, and ordinary symbol resolution
finds the selected function.

## A5 Template Inventory

The complete proposed per-instance catalog is generated under:

```text
bridge/inventory/npuir-template-instance-classification.md
bridge/inventory/npuir-template-instance-assignments.tsv
```

The Markdown file groups every canonical exported symbol beneath a semantic
category directory. Each instance then owns an exact-name resource, for
example:

```text
Cube/Fixpipe/normal/fixpipe_normal_float_to_half_2d_to_2d.mlir
```

The TSV retains that exact proposed path, source bitcode, installed bundle,
execution domain, and physical duplicate count for tooling. Test planning is
tracked separately in `bridge/inventory/ptodsl-template-test-triage.md`.
Regenerate the classification from an installed NPU-IR build with:

```bash
NPUIR/tools/find_npuir_template_names.sh
bridge/tools/classify_npuir_template_names.sh
```

The classification is a proposed resource plan, not a claim that those PTODSL
implementations already exist.

The installed A5 (`c310`) bitcode bundles contain the following wrapper-symbol
inventory. These are instantiated function definitions, not counts of C++
source files or high-level operation classes.

| Category | Bundle/core | Operation templates | Notes |
|---|---|---:|---|
| Cube | AIC | 286 | Matmul, Cube-local movement, fixpipe, and related operations |
| Vector/SIMD | AIV | 663 | Non-SIMT vector operations and data movement |
| Explicit SIMT families | AIV | 1,466 | SIMT arithmetic, load/store, atomic, indexing, and related wrappers |
| **Unique operation total** | AIC + AIV | **2,415** | 286 Cube plus 2,129 total Vector wrappers |
| AIC debug/support | AIC | 106 | Assertions, printing, and debug lifecycle helpers |
| AIV debug/support | AIV | 210 | Assertions, printing, and debug lifecycle helpers |
| **Normal bundle total** | AIC + AIV | **2,731** | Operations plus support wrappers |
| Mixed AIC bundle | MIX AIC | 106 | Duplicated AIC debug/support wrappers in this build |
| Mixed AIV bundle | MIX AIV | 210 | Duplicated AIV debug/support wrappers in this build |
| Separate mixed operations | MIX | 0 | No additional mixed operation-template family in this build |
| **Physical definition total** | All four bundles | **3,047** | Includes duplicated support wrappers in the MIX bundles |

The useful operation hierarchy is therefore:

```text
2,415 operation-template instances
|- 286 Cube/AIC
`- 2,129 Vector/AIV
   |- 1,466 explicit SIMT-family instances
   `-   663 other Vector/SIMD instances
```

SIMT is an execution style within the AIV template collection, not another
physical core category. The remaining AIV templates should not all be treated
as one homogeneous instruction class: the 663 count includes ordinary SIMD
work as well as DMA and other vector-side support operations.

### Mixed kernels versus mixed templates

NPU-IR supports logical mixed kernels and operations such as `MixMatmulOp` and
`MixGroupMatmulOp`. Pipeline infrastructure such as `SplitMixKernel` separates
a mixed function into cooperating AIC and AIV portions and preserves the
required synchronization and metadata.

That does not imply a third set of operation templates. In this configured A5
build, regular Cube sources are compiled for `aic`, regular Vector sources are
compiled for `aiv`, and the `mix.aic`/`mix.aiv` bundles contain only duplicated
debug/support wrappers. The operation bodies used by a mixed kernel therefore
come from the ordinary Cube and Vector template collections after the kernel is
partitioned.

For the PTODSL bridge, mixed-kernel support should normally reuse the same Cube
or Vector PTODSL specialization on the corresponding side. A separate
mix-specific specialization is justified only when its generated body or ABI
genuinely differs, not merely because the containing kernel began as MIX.

### What names and numeric suffixes mean

The native symbol name usually identifies a finite compile-time specialization:

- operation family
- source and destination datatype
- transpose or layout variant
- precision mode such as HF32
- optional bias or semantic mode
- sometimes rank or transfer form, such as `1d`, `2d`, or `4d_to_2d`

Numeric text in a template name is therefore not automatically a concrete
problem size. Names such as `1d` and `2d` commonly describe rank or an algorithm
form. Most runtime extents, strides, offsets, event IDs, and M/K/N values remain
function operands and should not produce a new specialization.

Physical tile shapes can still be part of a template contract when they change
the generated instruction structure or required memory layout. The correct
rule is not "shapes never matter"; it is that only structurally significant,
compile-time shape properties belong in the specialization identity.

### Implication for one-to-one PTODSL coverage

The native function identity is a useful basis for PTODSL coverage. Where a
native wrapper has distinct behavior that should remain visible to PTOAS, its
canonical native specialization should map to one deterministic pre-rendered
PTODSL specialization.

This should be a semantic one-to-one mapping, not a blind copy of every physical
symbol in every bitcode bundle. Debug duplicates, architecture duplicates,
internal implementation helpers, and MIX support copies are not new logical
PTODSL templates. Conversely, one generic PTODSL helper may cover several
native symbols if the native distinction is entirely representable by runtime
operands and does not alter the emitted PTO structure.

The generated registry should record both the native identity and PTODSL
identity so coverage can be audited without writing an independent manual
matching hierarchy.

## Structured Load Example

`hivm.hir.load` demonstrates the distinction between existing structured
adapters and exact template availability. In PTODSL mode, the early adapter
is eligible only when the native symbol has an exact installed counterpart:

| Load form | PTODSL bridge result |
|---|---|
| Contiguous rank-one GM to UB without an exact PTODSL resource | Preserve the native load call; explicit native fallback is required for object compilation. The separate `direct` mode retains its existing `pto.mte_gm_ub` conversion. |
| Compatible rank-two f16 implicit-transpose GM to UB | Call to imported `__pto_load_gm_to_ubuf_2d_half` MLIR helper |
| Exact PTODSL entry exists but its adapter cannot handle the contract | Explicit conversion failure; no silent native fallback |
| No exact PTODSL entry exists | Preserve the native call and its descriptor ABI for `FALLBACK=1` |

The rank-two helper receives sizes and strides as runtime operands. Its `2d`
identity indicates a structurally different NDDMA algorithm; it does not mean
that a separate pre-rendered helper is needed for every concrete two-dimensional
shape.

## Current PTODSL Scalability Problem

The first native-name selection slice is implemented for MmadL1, but
specialization metadata is still repeated in several places:

1. Python `SPECIALIZATIONS` and `EXPLICIT_INSTANTIATIONS`
2. Generated MLIR filenames and attributes
3. CMake resource lists
4. Symbol constants in `PTODSLTemplateImporter.h`
5. The importer resource table
6. C++ matching `if/else` chains

Adding a new specialization requires coordinated edits across these locations.
That is manageable for a proof of concept, but it is fragile when supporting
many operations, datatypes, layouts, architectures, and flags.

MmadL1 no longer imports every neighboring specialization. Remaining native
calls now use the exact-symbol resolver; ND2NZ and Fixpipe no longer trigger
family-wide imports in the default PTODSL path.

## Recommended Long-Term Design

Keep the useful native NPU-IR convention:

```text
canonical specialization identity
    -> deterministic symbol
    -> ahead-of-time generated artifact
```

For PTODSL, represent this identity with a typed `TemplateKey` and generate the
registry from the same source that generates the MLIR artifacts.

### Template key

A key could contain:

```cpp
TemplateKey {
  operation;
  architecture;
  inputType;
  accumulatorType;
  outputType;
  biasType;
  transposeA;
  transposeB;
  precisionMode;
  layoutVariant;
}
```

Only include properties that change the generated MLIR structure.

The following should normally remain runtime operands:

- actual M/K/N
- buffer addresses and offsets
- event IDs
- initialization or accumulation condition
- runtime strides supported by the helper

If a value is used by Python to choose which operations are emitted, it belongs
in the key. If it can be represented by an MLIR operand or control-flow
operation inside one helper, it normally does not.

### Separate identity from compatibility

Use two concepts:

- `TemplateKey` selects the structural specialization.
- `TemplateContract` validates its required ABI and buffer representation.

The contract can check:

- ranks
- element types
- memory spaces
- physical tile layouts
- supported extents
- synchronization operand shape
- required static or dynamic strides

This avoids putting every buffer property into the specialization key.

### Generated registry

The explicit-instantiation definition should generate both the MLIR helpers and
a C++ registry resembling:

```cpp
constexpr TemplateDescriptor templates[] = {
  {
    key: {MmadL1, A5, F16, F32, NoTranspose},
    nativeSymbol: "mma_tile_half_to_float",
    symbol: "__pto_mmadl1_f16_f32_nn",
    file: "Cube/MmadL1/classic-float/mma_tile_half_to_float.mlir",
    adapter: MmadL1Adapter,
  },
  {
    key: {MmadL1, A5, F16, F32, TransposeB},
    nativeSymbol: "mma_tile_half_to_float_tb",
    symbol: "__pto_mmadl1_f16_f32_tb",
    file: "Cube/MmadL1/classic-float/mma_tile_half_to_float_tb.mlir",
    adapter: MmadL1Adapter,
  },
};
```

The registry should support lookup by both structured `TemplateKey` and native
template symbol. A generated sorted table or `StringMap` is sufficient; the
important property is deterministic exact lookup, not the particular container.

### One source of truth

One declarative specialization list should generate:

1. One pre-rendered MLIR resource per canonical native template instance
2. Deterministic helper symbols
3. A generated C++ registry, such as `PTODSLTemplateRegistry.inc`
4. The CMake resource/install list
5. Dispatch tests and resource validation tests

Generated outputs can remain checked in. CI should regenerate them and fail if
the checked-in products are stale. Normal NPU-IR kernel compilation still does
not run Python.

### Preferred native-call resolver

The preferred implementation should not reproduce NPU-IR's template-selection
logic in a second family-specific matcher. Native NPU-IR already computes the
canonical specialization symbol through `getOpLibraryCallName()` and already
owns the operation-specific construction of the external-call operands.

The bridge should reuse that result directly:

```text
hivm template operation
    -> native NPU-IR call materialization
    -> func.call @<native-symbol>(native ABI operands)
    -> look up @__pto_<native-symbol> in the installed PTODSL registry
       |- found: import that exact resource and retarget the call
       `- absent: retain the native call for explicit CCE fallback
```

For example:

```text
@mma_tile_half_to_float
    -> registry lookup for @__pto_mma_tile_half_to_float
```

`TemplateKey` remains useful as generated metadata for validation, inventory,
and diagnostics, but the canonical native symbol should be the primary
selection key. This removes the MmadL1-only `StringSwitch` and avoids writing a
second selection hierarchy for ND2NZ, Fixpipe, Vector, and SIMT families.

Each imported `@__pto_<native-symbol>` must expose an adapter entry point with
the same external ABI as the native call. The adapter keeps the native memref
descriptor at the boundary, extracts the effective PTO pointer plus any sizes,
strides, offsets, dimensions, synchronization values, and mode operands needed
by the PTO implementation, and then calls the internal PTODSL helper. The
internal helper does not need to accept memrefs.

This ABI rule is what makes call retargeting generic. Without an ABI-compatible
adapter, changing only the symbol name is insufficient because the existing
PTODSL helper may accept PTO pointers and explicit descriptor fields instead
of the native memref arguments.

Implementation should share or extract native call materialization rather than
copying the complete `HIVMToStandard` implementation. Both the native and
bridge pipelines must continue to use the same symbol-generation and operand
construction code so future NPU-IR template changes cannot silently make the
two paths disagree.

## Module-Wide Discovery and Rewrite

The expected compilation workload is usually a model block or module containing
many template calls, rather than one isolated template operation. Template
selection should therefore be performed once across the complete MLIR module.
The pass should not parse a resource independently every time a rewrite pattern
matches an operation.

The PTODSL bridge should use this sequence. MmadL1 now has the first
implementation of native-name discovery and exact-resource import, but its
manual lookup should be replaced by the generic native-call resolver above.
Generated registries, dependency closure, and aggregated side-effect-free
preflight diagnostics remain future work:

```text
1. Materialize each template operation using NPU-IR's native symbol and ABI.
2. Scan the resulting native template calls once across the module.
3. Form the exact candidate symbol `__pto_<native-symbol>` for each call.
4. Look up the candidate in the generated installed-resource registry.
5. Build a deduplicated RequiredTemplateSet for candidates that exist.
6. Parse each required resource once and import its adapter plus dependencies.
7. Retarget matched calls to their imported `__pto_` adapter symbols.
8. Leave unmatched native calls unchanged for the explicit fallback policy.
9. Diagnose ABI mismatches and unresolved calls according to fallback mode.
10. Continue through PTOAS lowering.
```

For example, hundreds of calls in one module may reduce to only a few dozen
unique native symbols. The pass then parses only those few dozen exact resource
files. Category directories are organizational and do not cause neighboring
instances to be parsed or imported.

Discovery and contract validation must be side-effect-free. The pass should not
import functions or mutate the module until all required specializations have
matched and passed validation. Unsupported templates can then be reported
together with their native names and contracts, without leaving partial imports
behind.

`applyPatternsGreedily` can remain as the rewrite driver, but it should consume
the already selected descriptors and imported symbol table. It must not perform
resource discovery or importing as an incidental rewrite side effect.

The sweep is scoped to one `bishengir-compile` MLIR module. If Triton invokes a
separate compiler process for every kernel, each process performs its own
sweep. Cross-process caching can be considered later, but it is not required
for the initial design.

## Proposed Semantic Resource Tree

Resources should be organized first by the three meaningful execution domains:
`Cube`, `Vector`, and `SIMT`. Each major native template family owns several
semantic category directories. Inside the final category, every canonical
native template instance has one exact-name `.mlir` resource.

Per-instance naming rules:

- Strip only the `_mlir_ciface_` export prefix from the native symbol.
- Name the resource `<canonical-native-symbol>.mlir` without inventing a second
  normalized identity.
- Keep exactly one canonical resource for symbols physically duplicated across
  normal and MIX bundles.
- Let an instance resource reference shared files under `common/`, but do not
  define unrelated template instances in the same resource.
- Treat the 2,941 unique canonical symbols as the resource inventory, not the
  3,047 physical definitions that include bundle duplicates.

```text
PTODSL/
|- README.md
|- registry/
|  |- templates.yaml
|  |- categories.yaml
|  `- generated/
|     |- TemplateRegistry.inc
|     `- TemplateDependencies.inc
|- generators/
|  |- generate_all.py
|  |- generate_cube.py
|  |- generate_vector.py
|  |- generate_simt.py
|  `- validate_registry.py
`- instantiations/a5/
   |- manifest.json
   |- common/
   |  |- addressing.mlir
   |  |- synchronization.mlir
   |  |- masking.mlir
   |  `- type-conversion.mlir
   |- Cube/
   |  |- MmadL1/
   |  |  |- classic-float/
   |  |  |  |- mma_tile_half_to_float.mlir
   |  |  |  |- mma_tile_half_to_float_tb.mlir
   |  |  |  `- ...
   |  |  |- fp32-ieee-hf32/
   |  |  |- integer/
   |  |  |- fp8-mx/
   |  |  `- bias/
   |  |- GlobalMmad/
   |  |  |- classic-float/
   |  |  |- integer/
   |  |  `- fp8-mx/
   |  |- Nd2Nz/
   |  |  |- standard-float/
   |  |  |- integer/
   |  |  |- fp8-mx/
   |  |  `- bias/
   |  |- Fixpipe/
   |  |  |- normal/
   |  |  |- nz2nd/
   |  |  |- nz2dn/
   |  |  `- dual-output/
   |  |- Copy/
   |  |  |- copy1d/
   |  |  |- l1-to-ub/
   |  |  `- ub-to-l1/
   |  `- Setup/
   |     |- mx-scale/
   |     `- set2d-initialize/
   |- Vector/
   |  |- DMA/
   |  |  |- gm-to-ub/
   |  |  |- ub-to-gm/
   |  |  |- ub-to-ub/
   |  |  |- ub-to-l1/
   |  |  `- unaligned-layout/
   |  |- Math/
   |  |  |- basic-arithmetic/
   |  |  |- trigonometric/
   |  |  |- nonlinear/
   |  |  |- logarithm-power/
   |  |  |- reciprocal-rounding/
   |  |  `- integer-special/
   |  |- Collective/
   |  |  |- prefix-sum-product/
   |  |  |- prefix-minmax/
   |  |  |- reduction-with-index/
   |  |  |- sorting/
   |  |  `- rearrangement/
   |  `- Integer64/
   |     |- arithmetic/
   |     |- compare-select/
   |     |- conversion/
   |     |- reduction/
   |     `- dma-addressing/
   |- SIMT/
   |  |- Direct/
   |  |  |- load-store/
   |  |  |- strided-rank1/
   |  |  |- strided-rank2/
   |  |  `- strided-rank3/
   |  |- IndirectLoad/
   |  |  |- rank1-rank2/
   |  |  |- rank3/
   |  |  `- rank4-rank5/
   |  |- IndirectStore/
   |  |  |- masked-low-rank/
   |  |  |- masked-high-rank/
   |  |  |- unmasked-low-rank/
   |  |  `- unmasked-high-rank/
   |  |- Indexing/
   |  |  |- select-low-rank/
   |  |  |- select-high-rank/
   |  |  |- index-put/
   |  |  |- gather/
   |  |  `- scatter/
   |  |- Atomic/
   |  |  |- arithmetic/
   |  |  |- minmax/
   |  |  |- compare-swap/
   |  |  |- block/
   |  |  `- software/
   |  `- Collective/
   |     |- histogram/
   |     `- scan/
   `- Support/
      |- cube-debug/
      |- vector-debug/
      |- assertions/
      |- print-lifecycle/
      `- runtime-sync/
```

The tree is a proposed semantic ownership model, not a claim that every listed
instance is already implemented. The category level remains useful for
ownership, review, and navigation, but it is not an import unit. One exact
instance file is the import unit.

## Fallback Policy

Use exact PTODSL symbol lookup first, then explicit native fallback when that
entry is unavailable. A generic PTODSL body may be reused behind exact entry
points, but selection must not perform an additional shape-based search.

Fallback applies uniformly to every native template family:

- If an exact `__pto_<native-symbol>` resource is registered, the bridge must
  use it. A contract, import, or lowering failure is an error and must not
  silently fall back.
- If no PTODSL resource is registered, retain the native call unchanged.
- With `FALLBACK=0` (the default), an unresolved native call is a compile-time
  error.
- With `FALLBACK=1`, link only the required native CCE symbols and their
  transitive dependencies.

This distinguishes missing PTODSL coverage from a broken PTODSL
implementation. It also gives ND2NZ, Fixpipe, Vector, and SIMT the same
fallback behavior instead of hard-failing in family-specific conversion code.

The default PTODSL bridge therefore does not silently fall back to opaque CCE
bitcode. Compatibility fallback remains explicit and auditable.

## Minimal Implementation Steps

1. Extract or share native template-call materialization so the default and
   bridge paths use the same symbol and operand construction.
2. Add a generated registry keyed by exact `__pto_<native-symbol>` names.
3. Add a module-wide resolver that discovers native calls, imports only exact
   required resources, and retargets matching calls.
4. Give each existing PTODSL instance a native-ABI adapter entry point.
5. Remove the MmadL1-only manual `StringSwitch` after equivalent tests pass.
6. Route unmatched calls through the uniform `FALLBACK=0/1` policy.
7. Migrate ND2NZ, Fixpipe, and existing Vector helpers to the same mechanism.
8. Add tests for exact selection, mandatory failure of a registered-but-broken
   helper, fallback disabled, fallback enabled, mixed AIC/AIV modules, and
   import deduplication.
9. Generate CMake resource lists and registry entries from the explicit
   instantiation specification.

## Implementation Scope

Generic selection and fallback are a focused compiler change: reuse native
symbol/operand construction, resolve exact PTODSL resources, and preserve
unmatched native calls for the existing selective linker. The earlier 2-4 day
estimate included developing adapters and broad runtime validation, which are
not required for this fallback task.

Implementing additional PTO bodies and their argument adapters remains
separate work owned by each template instance's developer. Improving fallback
does not require implementing all roughly 2,415 native operation instances.

## Conclusion

The current PTODSL implementation is a sound proof of concept: pre-generated
MLIR avoids Python execution during compilation, and imported PTO bodies remain
visible to PTOAS optimizations.

For long-term support, the manual specialization knowledge should be replaced
with a generated typed registry and a module-wide requirement sweep. Semantic
directories keep the roughly 3,000 exact-name resources navigable, while one
resource per canonical instance makes selection, import, testing, and coverage
unambiguous. This preserves NPU-IR's deterministic symbol and ahead-of-time
instantiation model without retaining duplicated tables, large `if/else`
dispatch code, or import-all behavior.
