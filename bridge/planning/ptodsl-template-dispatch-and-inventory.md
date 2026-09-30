# NPU-IR and PTODSL Template Matching

This document answers the questions raised in `template.md`: how PTODSL
templates are matched today, how native NPU-IR dispatches CCE/AscendC
templates, and what design should be used as the number of PTODSL templates
grows.

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
    symbol: "__pto_mmadl1_f16_f32_nn",
    file: "mmadl1_f16_f32_nn.mlir",
    adapter: MmadL1Adapter,
  },
  {
    key: {MmadL1, A5, F16, F32, TransposeB},
    symbol: "__pto_mmadl1_f16_f32_tb",
    file: "mmadl1_f16_f32_tb.mlir",
    adapter: MmadL1Adapter,
  },
};
```

A hash map is unnecessary. The specialization space is small and discrete, so
a generated sorted array or typed lookup table is simpler to inspect and test.

### One source of truth

One declarative specialization list should generate:

1. Pre-rendered MLIR helper files
2. Deterministic helper symbols
3. A generated C++ registry, such as `PTODSLTemplateRegistry.inc`
4. The CMake resource/install list
5. Dispatch tests and resource validation tests

Generated outputs can remain checked in. CI should regenerate them and fail if
the checked-in products are stale. Normal NPU-IR kernel compilation still does
not run Python.

## Recommended Compilation Flow

The PTODSL bridge should eventually use this sequence:

```text
1. Scan candidate HIVM template operations.
2. Normalize each operation into a TemplateKey.
3. Look up an exact TemplateDescriptor.
4. Validate the selected TemplateContract.
5. Collect only the selected MLIR resources.
6. Import each required helper once.
7. Rewrite the HIVM operations into calls.
8. Continue through PTOAS lowering.
```

Selection should be side-effect-free. It should not import functions or mutate
the module until a specialization has matched and passed validation. This
prevents partial imports when a later compatibility check fails.

`applyPatternsGreedily` can remain as the rewrite driver, but it should consume
the selected descriptors rather than contain a growing collection of unrelated
manual specialization branches.

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
2. Convert the existing five MMAD variants into a generated registry without
   changing their generated MLIR bodies.
3. Split MMAD handling into key extraction, registry lookup, contract
   validation, and call adaptation.
4. Change the importer to accept selected descriptors and import only the
   required files.
5. Generate symbol declarations and CMake resource lists from the explicit
   instantiation specification.
6. Add checks for duplicate keys, duplicate symbols, missing resources, wrong
   instantiation attributes, and helper ABI mismatches.
7. Apply the same mechanism to ND2NZ, Fixpipe, and vector templates after the
   MMAD migration proves the design.

## Conclusion

The current PTODSL implementation is a sound proof of concept: pre-generated
MLIR avoids Python execution during compilation, and imported PTO bodies remain
visible to PTOAS optimizations.

For long-term support, the manual specialization knowledge should be replaced
with a generated typed registry. This preserves NPU-IR's proven deterministic
symbol and ahead-of-time-instantiation model while avoiding duplicated tables,
large `if/else` dispatch code, and unnecessary importing of every helper.
