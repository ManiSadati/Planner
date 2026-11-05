# Bridge follow-ups

Track compatibility workarounds, missing behavior, and cleanup tasks here.

| ID | Component | Category | Topic | Status |
|---|---|---|---|---|
| 01 | MmadL1 | `todo` | [Metadata signature cleanup](01-mmadl1-todo-metadata-signature-cleanup.md) | Deferred refactor; review current metadata usage before implementation. |
| 02 | MmadL1 / Fixpipe | `missing` | [Unit-flag group coordination](02-mmadl1-fixpipe-missing-unit-flag-group-coordination.md) | Not implemented; no runtime failure isolated specifically to this gap. |
| 03 | Fixpipe | `ptoas-blocker` | [Selected-pointer store alignment](03-fixpipe-ptoas-blocker-selected-pointer-alignment.md) | NPU-IR workaround implemented; performance impact unmeasured. |

## Naming new notes

Use `NN-component-category-topic.md`, with lowercase words separated by hyphens.
For example: `04-nd2nz-todo-dynamic-view-adaptation.md`.

Categories:

- `todo`: planned cleanup, investigation, or improvement.
- `missing`: behavior or implementation that is not yet provided.
- `ptoas-blocker`: a PTOAS limitation affecting compilation or execution; an NPU-IR workaround may already exist.

Assign the next unused two-digit ID. IDs identify notes, not priority rankings.
Use both component names when an issue spans components. Keep progress and
resolution status in this index; the category describes the kind of issue.

Each note should describe the issue and impact, concrete evidence, the fix or
workaround, and remaining work or validation. Distinguish demonstrated failures
from potential parity gaps.
