---
phase: quick-260531-aw1
plan: 01
subsystem: cubecl
tags: [cubecl, rys, eigensolver, double-double, numerics, fma]

requires:
  - phase: 25-group-2-hessian
    provides: host eigh + rys_wheeler engine (FND-02), 29/29 vendor family parity
provides:
  - "#[cube] CPU-backend symmetric-tridiagonal eigensolver (cint_diagonalize) — bit-identical to the host MRRR/QL+Rayleigh+Sturm reference"
  - "FMA fidelity probe proving the CubeCL 0.10.0 CpuRuntime fuses fma() bit-for-bit"
  - "On-device double-double Wheeler engine for Rys nroots 8..12 (lrys_jacobi / lrys_laguerre / lrys_schmidt) using the fused device fma intrinsic"
  - "On-device f64 Wheeler kernels for nroots 6,7 (retained; production dispatch stays host for family-parity fidelity)"
  - "In-crate vendor reference-table regression test (rys_roots_host_nroots6to12_matches_libcint)"
affects: [phase-26, phase-27, rys, hessian, derivatives]

tech-stack:
  added: []
  patterns:
    - "Multi-launch CubeCL host orchestrator: moment/Wheeler #[cube] kernel -> device eigh -> transform #[cube] kernel (no fused mega-kernel)"
    - "Double-double in #[cube] via a DdDev CubeType + fma intrinsic; dd accumulators threaded as scalar hi/lo pairs (CubeType structs are not reassignable)"
    - "Coefficient tables passed as device input Arrays (no const-array runtime index); break/continue rewritten to bounded-loop + converged/skip flags"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/math/eigh.rs
    - crates/cintx-cubecl/src/math/rys_wheeler.rs
    - crates/cintx-cubecl/src/math/rys.rs

key-decisions:
  - "FMA probe verdict FUSED -> double-double two_prod uses the device fma intrinsic directly; no Dekker-split software product needed"
  - "nroots 8..12 (double-double) run fully on-device and preserve 29/29 family parity byte-identically"
  - "nroots 6,7 production dispatch stays host (parity-honest escape hatch): the f64 device kernels are bit-identical in isolation but a CubeCL launch in the family hot path perturbs subsequent host g-tensor accumulation ~1e-11, tripping the flat-1e-12 family gate"

patterns-established:
  - "CubeCL CpuRuntime multi-kernel numerics pipeline with a host-side eigh seam"
  - "DdDev double-double layer for extended-precision #[cube] arithmetic"

requirements-completed: [AW1-EIGH, AW1-WHEELER-F64, AW1-WHEELER-DD]

duration: 156min
completed: 2026-05-31
---

# Quick Task 260531-aw1: Port host eigh + rys_wheeler to CubeCL `#[cube]` Summary

**Symmetric-tridiagonal eigensolver and the long-double (double-double) Rys nroots 8..12 Wheeler engine ported to fused-FMA `#[cube]` CPU-backend kernels; 29/29 vendor family parity preserved byte-identically, with nroots 6,7 held on the host path as a documented parity-honest escape hatch.**

## Performance

- **Duration:** ~156 min
- **Started:** 2026-05-31T07:58Z (approx, first commit prep)
- **Completed:** 2026-05-31T01:10Z (UTC clock; commit stream 08:13–10:10 local JST)
- **Tasks:** 3 (+ Task 3a FMA probe) + 1 parity-honest dispatch fix
- **Files modified:** 3 (all under `crates/cintx-cubecl/src/math/`)

## Accomplishments

- **eigh on-device (Task 1):** `cint_diagonalize` (n≥3) launches `cint_diagonalize_kernel` on `CpuRuntime` — QL (tqli) + Wilkinson shift + eigenvector accumulation, Rayleigh-quotient refinement, and Sturm-bisection refinement, all `#[cube]`. Bit-identical to the pure-Rust reference across 2000 random tridiagonals (MAXDIFF = 0). n≤2 fast paths stay host (trivial dlaev2).
- **Wave-0 regression net:** froze vendor `CINTrys_roots` gold for nroots 6..12 into `rys_roots_host_nroots6to12_matches_libcint`, committed GREEN against the unmodified pre-port code BEFORE editing eigh.rs.
- **FMA probe (Task 3a):** proved the CubeCL 0.10.0 CPU backend lowers `fma(a,b,c)` to a true fused multiply-add, bit-for-bit identical to host `f64::mul_add`.
- **Double-double Wheeler on-device (Task 3):** nroots 8..12 (`lrys_jacobi`, `lrys_laguerre`, `lrys_schmidt`) run as `#[cube]` CPU kernels using a `DdDev` CubeType + the fused device `fma`. Byte-identical to the host dd path and to the vendor at the documented split.
- **f64 Wheeler kernels (Task 2):** Jacobi (erf + flocke Miller moments + Wheeler recursion → device eigh → transform) and Schmidt (gamma_inc + R_dsmit + Hessenberg-QR + R_dnode) ported to `#[cube]`; retained as the on-device implementation.

## Task Commits

1. **Wave-0 vendor gold reference table** — `dede96c` (test)
2. **Task 1: eigh → #[cube] CPU kernel** — `fb70d04` (feat)
3. **Task 3a: FMA fidelity probe (verdict FUSED)** — `9e2ae3c` (feat)
4. **Task 2: f64 Wheeler (nroots 6,7) → #[cube]** — `f25cf11` (feat)
5. **Task 3: double-double Wheeler (nroots 8..12) → #[cube]** — `16d7f43` (feat)
6. **Parity-honest dispatch fix (nroots 6,7 → host)** — `7f4fd0b` (fix)

## Files Created/Modified

- `crates/cintx-cubecl/src/math/eigh.rs` — `#[cube]` `cint_diagonalize_kernel` + host launcher; pure-Rust `cint_diagonalize_host` retained for unit tests + cross-check.
- `crates/cintx-cubecl/src/math/rys_wheeler.rs` — f64 + double-double `#[cube]` Wheeler/Jacobi/Schmidt/Laguerre kernels, `DdDev` dd layer, `erf_dev`/`gamma_inc`/QR root solver, FMA probe, host orchestrators, dispatch.
- `crates/cintx-cubecl/src/math/rys.rs` — `rys_roots_host_nroots6to12_matches_libcint` vendor reference-table test.

## REQUIRED RECORDS

### 1. FMA-probe verdict

**FUSED.** `fma_probe` (in `rys_wheeler.rs::tests`) computes the TwoProd error term `fma(a,b,-a*b)` on the CubeCL `CpuRuntime` for pairs whose product is not exactly representable and asserts it equals host `f64::mul_add(a,b,-a*b)` BIT-FOR-BIT. It passes — the CPU backend performs a true single-rounding fused multiply-add. Consequence: the double-double `two_prod` uses the device `fma` intrinsic directly; **no Dekker-split software product was needed.**

### 2. nroots 8..12 placement

**ON-DEVICE.** All of nroots 8..12 run as `#[cube]` CPU-backend kernels (double-double Jacobi/Laguerre tridiagonal + device eigh + transform; double-double Schmidt for the nroots-8 large-x tail). They are byte-identical to the host dd path (in-process MAXDIFF = 0) and match the vendor at the documented split (atol=1e-12 nroots 6-7, max(atol=1e-12, rtol=1e-9·|ref|) nroots 8-12). They preserve 29/29 family parity (verified by bisection: device 8..12 + host 6,7 = full family suite green).

### 3. Parity confirmation

- **Vendor-gated family parity = 29/29 byte-identical at atol=1e-12** — final config (nroots 6,7 host + 8..12 device): center_2c2e 2, center_3c1e 2, deriv34 14, hess1e_ipip 8, hess2e 2, hess_multicenter_ipip 2, int2c2e_ip 4 — all PASS.
- **Vendor-gated `rys_nroots_sweep` GREEN** at its documented split after the port.
- **NO tolerance was loosened below the documented baseline, and NO reference value was edited.** The in-crate reference table uses gold captured from the libcint vendor harness; nroots 6,7 assert strict atol=1e-12, nroots 8..12 assert max(atol=1e-12, rtol=1e-9·|ref|), mirroring `rys_nroots_sweep_parity.rs:38-42` exactly.

## Decisions Made

- **FMA fused → dd on-device.** Probe-confirmed fusion let the double-double layer use the device `fma` intrinsic with no software-FMA fallback.
- **Double-double via `DdDev` CubeType + scalar hi/lo threading.** CubeType structs are not reassignable in `#[cube]` (no `Assign`/`expand_assign`), so dd accumulators are carried as scalar `hi`/`lo` pairs and a `DdDev` is constructed transiently for each dd op.
- **eigh kept as a host-side launch seam.** The Wheeler kernels read back the tridiagonal `(a,b)` and call the existing `cint_diagonalize` (device eigh) rather than fusing eigh into a mega-kernel — matching the plan's guidance.

## Deviations from Plan

### Deviation 1 — [Escape hatch] nroots 6,7 production dispatch stays host

- **Found during:** Task 3 final gate (full vendor family suite).
- **Issue:** Routing nroots 6,7 through the f64 `#[cube]` device Wheeler reproducibly breaks `hess2e_parity` by ~1e-11 at the largest components (flat atol=1e-12, RTOL=0). The plan anticipated the *double-double* band (8..12) as the hard case; empirically the **opposite** held — 8..12 dd is clean on-device, and the f64 6,7 path is the one that trips the family gate.
- **Root-cause evidence:** The device 6,7 kernels are **bit-identical to the host path in isolation** — `rys_jacobi_device`/`rys_schmidt_device` vs host: MAXDIFF = 0 over a dense x grid; `rys_nroots_sweep` and the in-crate reference table both pass byte-identically. So the divergence is **not** in the roots/weights. Forced-rebuild bisection isolated it: device 8..12 + host 6,7 ⇒ 29/29 green; full device (incl. 6,7) ⇒ hess2e fails. The signature (~1e-11 only at the largest components, only when 6,7 — which dominate hess2e — launch a device kernel in the family hot path) points to a CubeCL `CpuRuntime` launch perturbing the subsequent **host** g-tensor accumulation (FP-environment side effect), not a numerics error in the kernel itself.
- **Resolution (parity-honest, plan escape hatch):** nroots 6,7 production dispatch routes through the host path; the f64 device kernels are retained in the module as the on-device implementation (`rys_jacobi_device` is still wired for nroots 8's x≤11 branch). nroots 8..12 stay on-device. **No tolerance loosened, no reference value edited.**
- **Files modified:** `crates/cintx-cubecl/src/math/rys_wheeler.rs`
- **Committed in:** `7f4fd0b`

---

**Total deviations:** 1 (documented escape hatch; parity-preserving).
**Impact:** 29/29 family parity preserved byte-identically. Maximal on-device coverage achieved (eigh + all of nroots 8..12) without compromising the sacred parity gate.

## Issues Encountered

- **Build-cache staleness during bisection.** Early dispatch-only edits did not reliably trigger a cintx-cubecl rebuild, producing misleading "pass" results. Resolved by `touch`-forcing rebuilds; the final conclusions are all from forced-rebuild, vendor-gated runs.
- **CubeType non-assignability.** `DdDev` cannot be reassigned to a `mut` local in `#[cube]` (`expand_assign` requires `CubePrimitive`). Worked around by threading dd state as scalar hi/lo pairs.
- **`#[cube]` type-inference quirks.** Bare `0u32` `let mut` counters and reused counter variables confused the macro (`NativeExpand<_>` ambiguity); fixed by explicit `let mut x: u32 = 0;` and distinct per-loop counters (bessel.rs idiom).

## Known Stubs

None — no placeholder/empty-data stubs introduced. The f64 device 6,7 kernels are complete, tested (bit-identical), and retained; they are intentionally not wired into the family-critical dispatch (documented above).

## Next Phase Readiness

- The Rys quadrature tail (eigh + nroots 8..12) now runs under the CubeCL CPU backend the parity gate uses, aligning with the CLAUDE.md "host = planning/validation/marshaling" constraint for that band.
- Open follow-up (out of scope): root-cause and eliminate the CubeCL `CpuRuntime` launch FP-environment side effect so the f64 6,7 kernels can also join the family-critical path; or batch the rys launches to amortize/quarantine the effect.

---
*Quick task: 260531-aw1*
*Completed: 2026-05-31*

## Self-Check: PASSED

- All modified files present (eigh.rs, rys_wheeler.rs, rys.rs) and SUMMARY.md created.
- All 6 commits verified in git log (dede96c, fb70d04, 9e2ae3c, f25cf11, 16d7f43, 7f4fd0b).
