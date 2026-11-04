# Pointer views across compiler-owned outlined vector helpers

The human explicitly requested fixing the defined outlined VF body and callers,
rather than leaving a source-public compiler-generated helper unsupported.
The prior private-only policy is superseded for these compiler-owned VFs.

Selected solution: recognize the existing outliner markers on both the defined
function and every direct symbol use before dialect conversion drops call
markers. Carry a temporary bridge ownership marker through conversion, then
allow the existing guarded pointer-argument specialization for those helpers
regardless of source symbol visibility. Remove the marker before output.
No name-based ownership inference, new pass, finalizer, PTOAS edit or inlining.

Unrelated public functions, mixed marked/unmarked callers, declarations, entry
points and explicit C interfaces keep their existing ABI. All-call alignment
proofs, static identity layouts, descriptor element compatibility, argument
attribute exclusions and native descriptor consumers remain unchanged.

- [x] Add a failing marked-public VF regression with an external descriptor
      consumer, and a negative mixed marked/unmarked public-call case.
- [x] Implement ownership recognition in the existing AVE-to-VMI pass.
- [x] Rebuild and run focused plus full converter tests.
- [x] Replay the source-derived nn-disabled and small-b16 AVE inputs through
      the updated converter and pinned PTOAS device emission; compilation only.
- [x] Record results, limits and revised mapping policy.

The user explicitly said not to run kernels. Preserve existing staged changes;
new changes remain unstaged. Never commit.


Verification: converter suite 98/101, unchanged three baseline failures;
source-derived small-b16 and nn-disabled AVE inputs both produce device
binaries with current converter/pinned PTOAS. Retained frontend/templates were
used for source capture; no full latest frontend rebuild or kernel execution.
Independent review found no important issues in the revised ownership policy.
