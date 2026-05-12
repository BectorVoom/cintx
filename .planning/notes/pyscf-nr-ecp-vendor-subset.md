# PySCF nr_ecp vendor subset rationale

**Phase:** 19 (int1e_ecp_* Type-1 / Type-2 evaluator)
**Plan:** 19-01 (Wave 0 install / scaffold)
**Created:** 2026-05-12

## What this records

The vendor subtree at `vendor/pyscf-nr-ecp/` mirrors three source files
from upstream PySCF — `pyscf/lib/gto/nr_ecp.{c,h}` and `nr_ecp_deriv.c`
— at upstream commit `60cd9022b5158b0eef46ded606a03b111a0ad08c`
(`master` HEAD as of 2026-05-12). Apache-2.0 provenance is preserved
verbatim in `vendor/pyscf-nr-ecp/LICENSE` (copied from `pyscf/LICENSE`)
and `vendor/pyscf-nr-ecp/NOTICE` records the upstream URL, commit SHA,
and the original `pyscf/lib/gto/` paths.

Per phase decision **D-01 REVISED (2026-05-12)** the PySCF nr_ecp
sources are the **primary byte-identity oracle** for cintx's
`int1e_ecp_*` family — libcint 6.1.3 upstream ships **zero** ECP source
files, and the C functions historically attributed to "libcint ECP"
all live in PySCF (same author, Qiming Sun). cintx-oracle's `build.rs`
gains a parallel `cc::Build` chain (gated by
`CINTX_ORACLE_BUILD_VENDOR=1`) that compiles these sources alongside
the existing vendored libcint 6.1.3 tree.

## Subset boundary

Only three upstream files were copied verbatim:

| Upstream                              | Vendor                          |
| ------------------------------------- | ------------------------------- |
| `pyscf/lib/gto/nr_ecp.c`              | `src/nr_ecp.c`                  |
| `pyscf/lib/gto/nr_ecp_deriv.c`        | `src/nr_ecp_deriv.c`            |
| `pyscf/lib/gto/nr_ecp.h`              | `include/nr_ecp.h`              |
|                                       | `include/gto/nr_ecp.h` (dup)    |
| `pyscf/LICENSE`                       | `LICENSE`                       |

`include/gto/nr_ecp.h` is a verbatim duplicate of `include/nr_ecp.h`.
nr_ecp.c contains `#include "gto/nr_ecp.h"`, and rather than patch the
upstream source we satisfy the path via a duplicate so the upstream `.c`
file stays unchanged.

## Shim headers (cintx-authored, NOT upstream PySCF)

PySCF's `nr_ecp.{c,_deriv.c}` `#include` two pyscf-internal headers:

| Pyscf path               | Shim path                          | Reason                |
| ------------------------ | ---------------------------------- | --------------------- |
| `np_helper/np_helper.h`  | `include/np_helper/np_helper.h`    | Avoid vendoring numpy helpers; no symbols called. |
| `vhf/fblas.h`            | `include/vhf/fblas.h`              | Avoid vendoring pyscf vhf tree; only `dgemm_` referenced. |

A grep of nr_ecp.c + nr_ecp_deriv.c confirms:

- **No** `NPdsymm_triu` / `NPdunpack_tril` / `NPdpack_tril` / `NPdtranspose`
  / `NPomp_*` symbols are called. The shim header forward-declares them
  for compilation parity but they are not link dependencies.
- **No** `daxpy_` / `dcopy_` / `dscal_` / `dasum_` / `ddot_` / `dgemv_` /
  `dger_` / `dsymm_` are called. They are declared in the shim
  `fblas.h` but unused.
- **`dgemm_` IS called** (9 call sites in `nr_ecp.c`; 0 in `nr_ecp_deriv.c`),
  with `transa,transb ∈ {'N','T'}` and standard column-major layout.

## BLAS dependency (`dgemm_`)

System BLAS detection: `/usr/lib/x86_64-linux-gnu/libblas.so.3` exists
on the dev host but **no `libblas.so` symlink** is provided, so plain
`-lblas` will not link without `BLAS_LIBS` or a development BLAS
package. Rather than make cintx-oracle's vendor build depend on the
caller installing `libblas-dev`, the vendor tree ships a minimal
cintx-authored reference `dgemm_` implementation at
`src/dgemm_shim.c`. The shim:

- Handles all four trans combinations (`'N'/'N'`, `'T'/'N'`, `'N'/'T'`,
  `'T'/'T'`); the cases actually used by nr_ecp.c are `'N'/'N'` and
  `'T'/'N'`.
- Is correctness-first (triple loop, no blocking / threading). Adequate
  for oracle parity gates where wall-clock latency is not gated.
- Lives in the vendor subtree but is NOT upstream PySCF code; the
  NOTICE file records this distinction explicitly.

If a future cintx-oracle build wants to use system BLAS for the PySCF
nr_ecp path, the integration steps are:

1. Drop `src/dgemm_shim.c` from the `cc::Build` chain.
2. Emit `cargo:rustc-link-lib=blas` (or `openblas`) from
   `crates/cintx-oracle/build.rs`.
3. Optionally remove the `dgemm_` declaration from the shim
   `fblas.h` (system BLAS provides it).

No other change is required — the shim header signature matches the
standard Fortran BLAS dgemm contract verbatim.

## Why not pull the full pyscf/lib tree?

- The upstream `np_helper` and `vhf` directories transitively pull in
  numpy and an OpenMP-augmented BLAS variant. cintx-oracle's existing
  `cc::Build` chain is hermetic and single-threaded; matching the full
  upstream toolchain would push the oracle into the much heavier pyscf
  build path.
- The upstream LICENSE file is Apache-2.0 for the whole project, but the
  cintx workspace prefers minimal subsets per
  `<canonical_refs>` ("keeps upstream libcint sync clean").

## Provenance audit checklist

- [x] LICENSE present, Apache-2.0 verbatim from upstream
- [x] NOTICE present, lists upstream URL + commit SHA + Apache-2.0 grant
- [x] Apache-2.0 §4 attribution requirement satisfied (NOTICE preserves
      copyright holder)
- [x] Shim files clearly labeled "NOT from upstream PySCF" in their
      header docstrings
- [x] No upstream source modified (verified with `diff` against the
      fetched HEAD)
