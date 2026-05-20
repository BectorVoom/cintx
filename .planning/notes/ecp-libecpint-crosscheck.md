# libecpint secondary cross-check oracle — provenance & tolerance rationale

**Status:** Optional, **non-blocking** secondary oracle (Phase 19 D-02 REVISED).
**Scope:** ECP-04 secondary-oracle clause / ROADMAP SC#4. Never enters CI's required gates.

## What this is

A cross-implementation drift detector that compares cintx's safe-API ECP matrices
(`int1e_ecp_{cart,sph}`) against [libecpint](https://github.com/robashaw/libecpint),
the actually-published JCP 2017 ECP library. It is wired entirely behind an explicit
opt-in (`CINTX_LIBECPINT_ORACLE=1`), mirroring the Phase 16 ROCm opt-in precedent
(`CINTX_ROCM_ORACLE=1` → `has_rocm_oracle`-style env-gate). The new build cfg is
`has_libecpint_oracle`.

## Why it is informational, NOT byte-identity

The **primary, blocking** byte-identity gate for Phase 19 is the PySCF `nr_ecp` parity
at `atol=1e-12, rtol=0.0` (closed by plans 19-06 / 19-07; see
`crates/cintx-oracle/tests/safe_api_ecp_parity.rs`). That gate runs independently and
strictly, and a loose secondary tolerance cannot mask a primary-gate failure (threat
T-19-25: accepted by design).

libecpint and PySCF `nr_ecp` use **different internal recurrence + quadrature
conventions** (libecpint: recursive code generation + Gauss-Chebyshev quadrature with
analytical derivatives; PySCF `nr_ecp`: K-Taylor precomputed Bessel tables +
`ECPrad_part`/`ECPrad_block` radial machinery). They also differ in normalization /
phase conventions. Byte-identity at `atol=1e-12` is therefore NOT expected. The
cross-check tolerance is set to an **informational loose envelope**:

```
CROSSCHECK_ATOL = 1e-9
CROSSCHECK_RTOL = 0.0
```

documented in `crates/cintx-oracle/tests/ecp_libecpint_crosscheck_parity.rs` as
"libecpint and PySCF nr_ecp use different recurrence + quadrature conventions
internally". The `1e-9` value is a starting envelope pending empirical measurement on
a host with libecpint installed; any observed `|diff|` above it is reported as
**informational drift**, never a CI failure.

## Upstream provenance

- **Project:** libecpint — https://github.com/robashaw/libecpint
- **License:** MIT (cintx-workspace-compatible).
- **Paper:** R. A. Shaw & J. G. Hill, *J. Chem. Phys.* **147**, 074108 (2017),
  "Prescreening and efficiency in the evaluation of integrals over ab initio
  effective core potentials".
- **Language:** C++17 (CMake build). Rust FFI requires an `extern "C"` shim layer
  (libecpint does not expose a stable C API of its own), so the cintx integration
  declares a thin `extern "C"` surface; see `crates/cintx-oracle/src/libecpint_ffi.rs`.

## How an operator activates the cross-check (opt-in)

The default cintx-oracle build does **NOT** compile or link libecpint. To activate the
live cross-check on a host that has libecpint available, an operator must:

1. **Obtain libecpint** (one of):
   - Install a system package / build from source via CMake and `make install`, OR
   - Vendor the libecpint source tree under `vendor/libecpint/` (NOT done by default —
     cintx does not vendor a large external C++ dependency on its own initiative; the
     opt-in scaffolding is what ships).
   libecpint depends on a small set of headers (Eigen, pugixml) bundled in its own
   repo; build per its README.

2. **Provide a tiny `extern "C"` C++ shim** that wraps libecpint's `GaussianIntegral` /
   `ECPIntegral::compute_shell_pair`-style entry points into the C signature declared
   in `libecpint_ffi.rs` (`cintx_libecpint_ecp_cart` / `cintx_libecpint_ecp_sph`). The
   build branch in `crates/cintx-oracle/build.rs` (guarded by `CINTX_LIBECPINT_ORACLE`)
   compiles this shim with `-std=c++17` and links libecpint. The exact shim source path
   and any `LIBECPINT_DIR` discovery are operator-supplied; the build branch documents
   the expected layout inline.

3. **Run the cross-check:**

   ```
   CINTX_ORACLE_BUILD_VENDOR=1 CINTX_LIBECPINT_ORACLE=1 \
       cargo test --locked -p cintx-oracle --features cpu \
       --test ecp_libecpint_crosscheck_parity -- --ignored
   ```

   The two tests (`test_int1e_ecp_cart_libecpint_crosscheck`,
   `test_int1e_ecp_sph_libecpint_crosscheck`) are both `#[ignore]` AND
   `#[cfg(has_libecpint_oracle)]`, so they only exist when the cfg is emitted (env var
   set at build time) and only run when explicitly requested with `-- --ignored`.

## Environment status at scaffolding time (2026-05-20)

libecpint was **NOT** available on the dev host this session (no system package, no
`pkg-config` entry, not vendored). Per Phase 19's "optional, non-blocking" framing, the
opt-in mechanism (build cfg + FFI shim + env-gated `#[ignore]` test file + this note)
landed with the default build verified unchanged; the **live cross-check run is deferred
to a host with libecpint installed**. The harness is in place per D-02; activating it
requires the operator steps above.

## Normalization / convention adapter (when run live)

When the live cross-check is run, the collector must map libecpint's per-shell-pair
output ordering onto cintx's row-major `[ao_i, ao_j]` AO-matrix layout. libecpint emits
Cartesian/spherical components in its own internal ordering; the adapter in the test
file's `collect_ecp_matrix_libecpint` helper is the single place that documents and
applies any required component permutation / normalization factor. Because byte-identity
is not expected, small convention mismatches that survive within `atol=1e-9` are reported
as informational, not corrected.

## Threat-register cross-reference (19-08-PLAN STRIDE)

- **T-19-25** (Tampering — cross-check masks a real bug): **accepted**. Informational by
  design; the strict PySCF gate runs independently.
- **T-19-26** (Integrity — FFI UB across the C++ boundary): **mitigated**. The shim is
  gated behind `has_libecpint_oracle` (off by default), takes Rust slices, bounds the out
  buffer, and is only exercised under explicit opt-in on a host with libecpint installed.
- **T-19-27** (Build regression on the default path): **mitigated**. The libecpint build
  branch is fully skipped when `CINTX_LIBECPINT_ORACLE` is unset; the default
  `cargo build -p cintx-oracle` is byte-for-byte unchanged (verified in Plan 19-08 Task 1).
