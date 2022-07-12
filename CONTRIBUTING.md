# Contributing to SSE2NEON

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to [SSE2NEON](https://github.com/DLTcollab/sse2neon),
hosted on GitHub. These are mostly guidelines, not rules. Use your best
judgment, and feel free to propose changes to this document in a pull request.

## Add New Intrinsic

The new intrinsic conversion should be added in the `sse2neon.h` file,
and it should be placed in the correct classification with the alphabetical order.
The classification can be referenced from [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#).

Classification: `SSE`, `SSE2`, `SSE3`, `SSSE3`, `SSE4.1`, `SSE4.2`

## Coding Convention

Software requirement: [clang-format](https://clang.llvm.org/docs/ClangFormat.html) version 12 or later.

Use the command `$ make indent` to enforce a consistent coding style.
