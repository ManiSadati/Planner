# Static ELF Comparison

## Generated Reports

Raw `readelf` outputs are stored in this directory:

| Artifact | Header | Sections | Symbols | Notes |
| --- | --- | --- | --- | --- |
| NPU-IR baseline raw object | `npuir-baseline.readelf-header.txt` | `npuir-baseline.readelf-sections.txt` | `npuir-baseline.readelf-symbols.txt` | `npuir-baseline.readelf-notes.txt` |
| PTOAS bridge raw object | `ptoas-bridge.readelf-header.txt` | `ptoas-bridge.readelf-sections.txt` | `ptoas-bridge.readelf-symbols.txt` | `ptoas-bridge.readelf-notes.txt` |
| PTOAS host fat object | `ptoas-fatobj.readelf-header.txt` | `ptoas-fatobj.readelf-sections.txt` | `ptoas-fatobj.readelf-symbols.txt` | `ptoas-fatobj.readelf-notes.txt` |

## Header Contract

| Artifact | ELF Type | Machine | Flags | Section Count |
| --- | --- | --- | --- | --- |
| NPU-IR baseline raw object | `EXEC` | `0x1029` | `0x990000` | 10 |
| PTOAS bridge raw object | `EXEC` | `0x1029` | `0x990000` | 8 |
| PTOAS host fat object | `REL` | x86-64 | `0x0` | 16 |

The NPU-IR baseline and PTOAS bridge artifacts are both Ascend device executable ELFs. The PTOAS fat object is an x86-64 relocatable object containing host-side registration and wrapper code; it is a negative-control artifact for raw `rtDevBinaryRegister` loading.

## Symbol Contract

| Artifact | Public Launch Symbol | Metadata Prefix |
| --- | --- | --- |
| NPU-IR baseline raw object | `vector_add_large_kernel` | `.ParamInfo_vector_add_large_kernel_*` |
| PTOAS bridge raw object | `vector_add_large_kernel_mix_aiv` | `.ParamInfo_vector_add_large_kernel_mix_aiv_*` |
| PTOAS host fat object | `vector_add_large_kernel` host wrapper | embeds `__aicore_rel_binary` and registration records |

The main contract mismatch is symbol naming: the PTOAS raw object currently exposes `vector_add_large_kernel_mix_aiv`, while the Triton-origin native raw object exposes `vector_add_large_kernel`.

## Notes

GNU `readelf -n` reports the same malformed-note warning pattern for both raw Ascend device objects. This is likely because the Ascend metadata note format is not a standard GNU note layout. The warning itself is therefore not considered a difference between the two raw-device candidates.
