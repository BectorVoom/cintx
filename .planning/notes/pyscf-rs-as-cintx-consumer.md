---
title: pyscf_rs is the downstream consumer of cintx's safe API
date: 2026-05-12
type: note
context: explore session — resolve GitHub repository latest issue (#11)
---

# pyscf_rs is the downstream consumer of cintx's safe API

The cintx safe API surface (`SessionRequest` in `crates/cintx-rs/src/api.rs`)
is consumed by `pyscf_rs` (BectorVoom/pyscf_rs, private repo). This
relationship is not visible from cintx's own source tree because pyscf_rs
declares cintx as a sibling path dep, not a published-crate dep.

## Path-dep wiring (observed in pyscf_rs `crates/pyscf-gto/Cargo.toml`)

```
cintx-core    = { path = "../../../cintx/crates/cintx-core" }
cintx-compat  = { path = "../../../cintx/crates/cintx-compat" }
cintx-rs      = { path = "../../../cintx/crates/cintx-rs" }
cintx-ops     = { path = "../../../cintx/crates/cintx-ops" }
cintx-runtime = { path = "../../../cintx/crates/cintx-runtime" }
```

`pyscf-gto`'s package `description` literally says
**"Molecular structure & integrals via cintx. Implemented in Phase 2."**

## Files in pyscf_rs that bind to cintx's safe API

- `crates/pyscf-gto/src/intor.rs` — wraps `SessionRequest` into a
  `mol.intor(...)`-style dispatch matching pyscf's Python `moleintor.getints`
  calling convention. This is the single file most affected by issue #11.
- `crates/pyscf-gto/src/ecp_engine_stub.rs` — placeholder evaluator pending
  issue #11 Task 1 (`int1e_ecp_*` Type-1/Type-2 projectors). Stub naming
  signals the integration point is reserved but not implemented.
- `crates/pyscf-gto/src/format_ecp.rs` — ECP basis parsing already exists in
  pyscf_rs; the missing piece is cintx-side evaluation, not pyscf_rs-side
  parsing.

## Verification path

Two independent oracle gates exist for any safe-API change in cintx:

1. **Primary (in this repo):** `cintx-oracle/tests/one_electron_parity.rs`
   and analogous arity-2/3/4 parity tests. Byte-identity against vendored
   libcint 6.1.3.
2. **Secondary (in pyscf_rs):** `pyscf-gto`'s `tests/oracle/` harness driven
   by `release-oracle-tests` feature. Validates the cintx-backed pyscf-gto
   surface against PySCF reference outputs. Independent of cintx's own
   oracle and catches integration regressions invisible from this repo.

A safe-API change should be considered "done" only when both gates pass.

## Why this note matters

When triaging future cintx safe-API issues, two checks save time:

- Grep the live pyscf_rs `pyscf-gto/src/intor.rs` for the affected operator
  name — if the wrapper expects a shape cintx doesn't yet produce, the issue
  is gating pyscf_rs work, not just cintx-internal polish.
- Search for `ecp_engine_stub` or other `*_stub.rs` files in pyscf_rs — each
  is a marker that a downstream feature is parked waiting on cintx.

## References

- Issue #11: https://github.com/BectorVoom/cintx/issues/11
- libcint-rs (ajz34): https://github.com/ajz34/libcint-rs — alternative
  Rust libcint binding; useful for cross-reference on the F-order /
  `aosym` calling convention.
- libECP (chrr, JCC 2017): https://github.com/chrr/libECP — independent C
  reference implementation of Type-1/Type-2 ECP projectors; viable
  secondary oracle for issue #11 Task 1.
