# PTOAS object and NPUBIN comparison

Source data: [comparison.txt](./comparison.txt)

## 1. PTOAS emits a fat host object

The PTOAS output is an x86-64 relocatable object:

```text
Type:                              REL (Relocatable file)
Machine:                           Advanced Micro Devices X86-64
```

It contains host `.text` and an embedded AICore object:

```text
[ 2] .text                 PROGBITS ... AX
[ 5] __aicore_rel_binary   PROGBITS ... A
[ 6] __aicore_rel_rec      PROGBITS ... WA
[ 8] .init_array           INIT_ARRAY ... WA
```

The symbol table confirms that the outer object contains a PTOAS host stub and registration code:

```text
FILE  LOCAL  DEFAULT ABS ptoas-host-stub-8bf80d..cpp
FUNC  LOCAL  DEFAULT   2 rtRegisterGlobals
FUNC  LOCAL  DEFAULT   2 cceModuleCtor
```

Therefore, the PTOAS artifact is a fat object containing:

```text
x86-64 host code
  + host registration information
  + embedded __aicore_rel_binary device object
```

## 2. The outer fat object is not linked

The ELF header identifies it as `REL`, and it has no program headers:

```text
Type: REL (Relocatable file)
There are no program headers in this file.
```

It also has unresolved host-runtime symbols:

```text
UND __cce_rtKernelLaunchWithFlagV2
UND __stack_chk_fail
UND rtFunctionRegister
UND rtLinkedDevBinaryRegisterDelay
```

The corresponding relocations are x86-64 host relocations:

```text
R_X86_64_PLT32 __cce_rtKernelLaunchWithFlagV2
R_X86_64_PLT32 __stack_chk_fail
R_X86_64_PLT32 rtFunctionRegister
R_X86_64_PLT32 rtLinkedDevBinaryRegisterDelay
```

These unresolved symbols belong to the outer host stub. They are not part of the extracted AICore object used by Triton.

## 3. The extracted AICore object still needs final linking

The `__aicore_rel_binary` section is extracted as `vadd.extracted-aicore-rel.o`. Its inspected properties are:

```text
Type: REL (Relocatable file)
Machine: 0x1029
Program headers: none
Undefined global symbols: none
Relocations: none
```

Although this example has no unresolved AICore symbols or relocations, it is still `ET_REL` and has no loadable program segments.

By comparison, the linked NPUBIN in `comparison.txt` has:

```text
Type:                              EXEC (Executable file)
Machine:                           <unknown>: 0x1029
Number of program headers:         2
```

It also has an executable load segment:

```text
Type  Offset    VirtAddr  PhysAddr  FileSiz  MemSiz  Flg  Align
LOAD  0x0000b0  0x0       0x0       0x000230 0x000230 R E 0x1000

Section to Segment mapping:
  00  .text
```

The AICore linker therefore changes the extracted device object from a section-only `ET_REL` file into a runtime-loadable `ET_EXEC` file with a `LOAD` segment.

Current implementation: [compiler.py](../../../../../triton-ascend/third_party/ascend/backend/compiler.py#L609-L664)

## 4. NPUIR output from `bishengir-compile`

The NPUIR flow asks `bishengir-compile` to generate `kernel.o`. Despite the `.o` filename, the output is already a linked AICore executable:

```text
Type:                              EXEC (Executable file)
Machine:                           <unknown>: 0x1029
Number of program headers:         2
```

It has an executable `LOAD` segment containing `.text`:

```text
Type  Offset    VirtAddr  PhysAddr  FileSiz  MemSiz  Flg  Align
LOAD  0x0000b0  0x0       0x0       0x000230 0x000230 R E 0x1000

Section to Segment mapping:
  00  .text
```

Its symbol table has no real undefined symbols. The remaining relocations are only in `.rela.debug_frame`, not in the executable `.text` section:

```text
Relocation section '.rela.debug_frame' ... contains 4 entries
```

Therefore, `bishengir-compile` performs the final AICore linking internally. `ttir_to_npubin()` only reads the generated `kernel.o` bytes, and Triton stores those same final binary bytes under the `.npubin` stage name.

Current implementation: [compiler.py](../../../../../triton-ascend/third_party/ascend/backend/compiler.py#L1387-L1445)

## Possible solutions

1. Make PTOAS emit an already linked AICore executable object (`ET_EXEC`) with the required program headers and `LOAD` segment. Triton can then use its bytes directly as the NPUBIN.
2. Perform the AICore linking inside `triton_ascend`, which is the solution currently implemented.
