# PTODSL Template Dispatch and NPU-IR Inventory

Last reviewed: 2026-10-01

This document records how PTODSL templates are matched today, how native
NPU-IR dispatches CCE/AscendC templates, the current A5 template inventory,
and the design to use as PTODSL coverage grows.

## Current PTODSL Template Flow

The PTODSL path currently uses pre-generated MLIR helpers and manual C++
selection.

### 1. Define explicit instantiations in Python

The PTODSL source defines `SPECIALIZATIONS` and `EXPLICIT_INSTANTIATIONS` in:

```text
$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py
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

- a deterministic symbol such as `__pto_mmadl1_f16_f32_tb`
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

When the pass sees any Cube template operation, it currently imports all Cube
PTODSL resources, even if the kernel only needs one helper. The importer parses
each MLIR file, validates its symbol and instantiation attribute, clones its
function into the kernel module, and marks it as an AIC function.

This is MLIR-level importing or linking: the helper definition and its call are
placed in the same MLIR module. It is not external LLVM bitcode linking.

### 4. Match an HIVM operation

For `hivm.hir.mmadL1`, `matchMmadL1Contract()` manually examines:

- input datatype
- accumulator datatype
- transpose flags
- HF32 mode
- physical memref shapes
- memory spaces
- synchronization operands
- supported runtime M/K/N range

An `if/else` chain selects the helper symbol. For example:

```text
f16 input + f32 accumulator + transpose B
    -> __pto_mmadl1_f16_f32_tb
```

The rewrite then replaces the original operation with a call:

```mlir
func.call @__pto_mmadl1_f16_f32_tb(...)
```

Runtime values such as actual M/K/N, pointers, event IDs, and initialization
state remain function operands. They do not create separate explicit
instantiations.

`applyPatternsGreedily` is only the rewrite driver. It does not automatically
search or rank the generated templates. The individual C++ rewrite patterns
perform template selection explicitly.

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

`hivm.hir.load` demonstrates why not every native template call needs the same
replacement strategy. The bridge handles it before `convert-hivm-to-std`, while
the structured memory contract is still available:

| Load form | PTODSL bridge result |
|---|---|
| Contiguous rank-one GM to UB | Direct `pto.mte_gm_ub` operation |
| Compatible rank-two f16 implicit-transpose GM to UB | Call to imported `__pto_load_gm_to_ubuf_2d_half` MLIR helper |
| Unsupported layout, padding, initialization, or descriptor contract | Explicit conversion failure; no silent fallback in default PTODSL mode |

The rank-two helper receives sizes and strides as runtime operands. Its `2d`
identity indicates a structurally different NDDMA algorithm; it does not mean
that a separate pre-rendered helper is needed for every concrete two-dimensional
shape.

## Current PTODSL Scalability Problem

The same specialization information is currently repeated in several places:

1. Python `SPECIALIZATIONS` and `EXPLICIT_INSTANTIATIONS`
2. Generated MLIR filenames and attributes
3. CMake resource lists
4. Symbol constants in `PTODSLTemplateImporter.h`
5. The importer resource table
6. C++ matching `if/else` chains

Adding a new specialization requires coordinated edits across these locations.
That is manageable for a proof of concept, but it is fragile when supporting
many operations, datatypes, layouts, architectures, and flags.

Importing all helpers before knowing which ones are needed will also become
wasteful as the resource set grows.

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
    file: "Cube/MmadL1/classic-float.mlir",
    adapter: MmadL1Adapter,
  },
  {
    key: {MmadL1, A5, F16, F32, TransposeB},
    nativeSymbol: "mma_tile_half_to_float_tb",
    symbol: "__pto_mmadl1_f16_f32_tb",
    file: "Cube/MmadL1/classic-float.mlir",
    adapter: MmadL1Adapter,
  },
};
```

The registry should support lookup by both structured `TemplateKey` and native
template symbol. A generated sorted table or `StringMap` is sufficient; the
important property is deterministic exact lookup, not the particular container.

### One source of truth

One declarative specialization list should generate:

1. Pre-rendered MLIR helper functions grouped into semantic resource packs
2. Deterministic helper symbols
3. A generated C++ registry, such as `PTODSLTemplateRegistry.inc`
4. The CMake resource/install list
5. Dispatch tests and resource validation tests

Generated outputs can remain checked in. CI should regenerate them and fail if
the checked-in products are stale. Normal NPU-IR kernel compilation still does
not run Python.

## Module-Wide Discovery and Rewrite

The expected compilation workload is usually a model block or module containing
many template calls, rather than one isolated template operation. Template
selection should therefore be performed once across the complete MLIR module.
The pass should not parse a resource independently every time a rewrite pattern
matches an operation.

The PTODSL bridge should use this sequence:

```text
1. Scan every candidate HIVM template operation in the module.
2. Derive its native template name and normalized TemplateKey.
3. Look up and validate every exact TemplateDescriptor.
4. Build a deduplicated RequiredTemplateSet.
5. Group the required descriptors by semantic MLIR resource pack.
6. Parse each required pack once.
7. Import only the selected helper functions and dependency closure.
8. Rewrite all matched HIVM operations into calls to those helpers.
9. Verify that no required template operation remains unresolved.
10. Continue through PTOAS lowering.
```

For example, hundreds of calls in one module may reduce to only a few dozen
unique native symbols and a small set of resource packs. A pack may contain many
generated functions, but only the exact selected functions should be cloned
into the kernel module. Pack size therefore affects resource parsing cost, not
the amount of helper IR passed to PTOAS.

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
`Cube`, `Vector`, and `SIMT`. Each major native template family then owns a
small number of semantic MLIR packs, normally three to five, rather than one
file per explicit instantiation.

```text
PTODSL/
|- README.md
|- registry/
|  |- templates.yaml
|  |- packs.yaml
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
   |  |  |- classic-float.mlir
   |  |  |- fp32-ieee-hf32.mlir
   |  |  |- integer.mlir
   |  |  |- fp8-mx.mlir
   |  |  `- bias.mlir
   |  |- GlobalMmad/
   |  |  |- classic-float.mlir
   |  |  |- integer.mlir
   |  |  `- fp8-mx.mlir
   |  |- Nd2Nz/
   |  |  |- standard-float.mlir
   |  |  |- integer.mlir
   |  |  |- fp8-mx.mlir
   |  |  `- bias.mlir
   |  |- Fixpipe/
   |  |  |- normal.mlir
   |  |  |- nz2nd.mlir
   |  |  |- nz2dn.mlir
   |  |  `- dual-output.mlir
   |  |- Copy/
   |  |  |- copy1d.mlir
   |  |  |- l1-to-ub.mlir
   |  |  `- ub-to-l1.mlir
   |  `- Setup/
   |     |- mx-scale.mlir
   |     `- set2d-initialize.mlir
   |- Vector/
   |  |- DMA/
   |  |  |- gm-to-ub.mlir
   |  |  |- ub-to-gm.mlir
   |  |  |- ub-to-ub.mlir
   |  |  |- ub-to-l1.mlir
   |  |  `- unaligned-layout.mlir
   |  |- Math/
   |  |  |- trigonometric.mlir
   |  |  |- nonlinear.mlir
   |  |  |- logarithm-power.mlir
   |  |  |- reciprocal-rounding.mlir
   |  |  `- integer-special.mlir
   |  |- Collective/
   |  |  |- prefix-sum-product.mlir
   |  |  |- prefix-minmax.mlir
   |  |  |- reduction-with-index.mlir
   |  |  |- sorting.mlir
   |  |  `- rearrangement.mlir
   |  `- Integer64/
   |     |- arithmetic.mlir
   |     |- compare-select.mlir
   |     |- conversion.mlir
   |     |- reduction.mlir
   |     `- dma-addressing.mlir
   |- SIMT/
   |  |- Direct/
   |  |  |- load-store.mlir
   |  |  |- strided-rank1.mlir
   |  |  |- strided-rank2.mlir
   |  |  `- strided-rank3.mlir
   |  |- IndirectLoad/
   |  |  |- rank1-rank2.mlir
   |  |  |- rank3.mlir
   |  |  `- rank4-rank5.mlir
   |  |- IndirectStore/
   |  |  |- masked-low-rank.mlir
   |  |  |- masked-high-rank.mlir
   |  |  |- unmasked-low-rank.mlir
   |  |  `- unmasked-high-rank.mlir
   |  |- Indexing/
   |  |  |- select-low-rank.mlir
   |  |  |- select-high-rank.mlir
   |  |  |- index-put.mlir
   |  |  |- gather.mlir
   |  |  `- scatter.mlir
   |  |- Atomic/
   |  |  |- arithmetic.mlir
   |  |  |- minmax.mlir
   |  |  |- compare-swap.mlir
   |  |  |- block.mlir
   |  |  `- software.mlir
   |  `- Collective/
   |     |- histogram.mlir
   |     `- scan.mlir
   `- Support/
      |- cube-debug.mlir
      |- vector-debug.mlir
      |- assertions.mlir
      `- print-lifecycle.mlir
```

The tree is a proposed semantic ownership model, not a claim that every listed
pack is already implemented. Exact boundaries should be checked against actual
generated MLIR size and template co-occurrence from representative model or
block compilations. Exceptionally large families such as indirect SIMT stores
may need another split, while consistently tiny packs may be combined.

## Fallback Policy

A clear fallback order should be used:

1. Exact PTODSL specialization
2. Generic PTODSL specialization, when one helper can safely accept dynamic
   operands
3. External CCE/AscendC template only when compatibility mode is explicitly
   requested

The default PTODSL bridge should not silently fall back to opaque CCE bitcode,
because that would hide unsupported coverage and bypass PTOAS optimization.

## Minimal Implementation Steps

1. Introduce `TemplateKey` and `TemplateDescriptor` types.
2. Add a module-wide discovery phase that records native names, keys, contracts,
   source operations, and all unsupported requirements without changing IR.
3. Convert the existing MMAD variants into a generated registry without
   changing their generated MLIR bodies.
4. Group the existing MMAD helpers into the first semantic resource packs.
5. Change the importer to parse each selected pack once and clone only selected
   helper symbols plus their transitive dependencies.
6. Rewrite all operations only after module-wide discovery and importing have
   completed successfully.
7. Generate symbol declarations and CMake resource lists from the explicit
   instantiation specification.
8. Add checks for duplicate keys, duplicate symbols, missing resources, wrong
   instantiation attributes, helper ABI mismatches, and unresolved operations.
9. Apply the mechanism to ND2NZ, Fixpipe, Vector, and SIMT families after the
   MMAD migration proves the design.

## Conclusion

The current PTODSL implementation is a sound proof of concept: pre-generated
MLIR avoids Python execution during compilation, and imported PTO bodies remain
visible to PTOAS optimizations.

For long-term support, the manual specialization knowledge should be replaced
with a generated typed registry and a module-wide requirement sweep. A small
number of semantic resource packs per major template family avoids thousands
of tiny files, while exact function-level importing keeps unrelated helper IR
out of PTOAS. This preserves NPU-IR's deterministic symbol and ahead-of-time
instantiation model without retaining duplicated tables, large `if/else`
dispatch code, or import-all behavior.
