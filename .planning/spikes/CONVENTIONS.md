# Spike Conventions

Patterns and stack choices established across spike sessions. New spikes follow these
unless the question requires otherwise.

## Stack

- **Spikes are workspace-linked Rust integration tests**, not standalone crates. cintx's
  device kernels are only reachable by depending on the workspace crates, so each spike is
  a file in `crates/cintx-oracle/tests/spike_<topic>_NNN.rs` that drives the real
  `eval_raw` path on the CubeCL `CpuRuntime`. The recorded copy + README live in
  `.planning/spikes/NNN-name/`.
- **`#[ignore]` every spike test** (`#[ignore = "spike NNN — run explicitly"]`) so the
  normal CI suite stays clean; run on demand with `-- --ignored --nocapture`.
- **Gate with `#![cfg(feature = "cpu")]`** at file top (the cubecl cpu backend).

## Structure

- Run (structural / vendor-free):
  `cargo test -p cintx-oracle --features cpu --test spike_<topic>_NNN -- --ignored --nocapture`
- Run (with libcint ground truth):
  prefix `CINTX_ORACLE_BUILD_VENDOR=1` — this turns on the `has_vendor_libcint` cfg.
  Vendored libcint 6.1.3 **builds and links cleanly in this environment** (cc + bindgen).
- The forensic output IS the deliverable: print a readable per-(tier × case) layout map
  (recovered strides, lens, populated counts, Δ vs ground truth), bracketed by
  `==== SPIKE NNN : ... ====` banners and a final `PASS`.

## Patterns

- **Dual ground truth.** Assert a vendor-FREE invariant that always runs, then gate the
  vendor byte-identity behind `#[cfg(has_vendor_libcint)]`. The spike degrades gracefully
  to "structural only" when vendor isn't built, and upgrades to "pinned" when it is.
- **Hand-derived numeric anchors beat tautologies.** `<g_R|r_c|g_R> = R_c·S` (gauge
  origin 0; read `S` from cintx's own `int1e_ovlp`) is an independent algebraic invariant
  with a hand-chosen number — it pins component identity + layout + origin without vendor.
- **Negative controls.** To claim an orientation/layout is pinned, also show the *wrong*
  interpretation fails (`mm(vendor, j-fastest) > 0`; `cintx != to_j_fastest(cintx)`).
- **Probe ladder.** `int1e_r / rr / rrr / rrrr` is the canonical uniform family spanning
  rank 3/9/27/81 with cart+sph `RawApiId` consts and `vendor_ffi::vendor_int1e_r{,r,rr,rrr}`.
- **Order-pinning by permutation-disagreement.** To prove an axis ordering is the one libcint
  uses (not a symmetric-block coincidence), reindex the buffer under every non-identity
  permutation and assert each diverges from vendor while the claimed order matches. The
  N-axis generalization of spike 003's `to_j_fastest` is `reindex(buf, rank, extents, perm)`
  with `extents` in fastest→slowest order (spike 004). Specializations: i-axis
  contraction-minor reinterpretation (spike 005), complex de-interleave + i/j transpose
  (spike 006).

## Tools & Libraries

- `cintx_oracle::fixtures::build_h2o_sto3g_common_orig()` — ready s/p fixture (nctr==1).
- Custom `atm/bas/env` builders (model on `moment_genctr_parity.rs`) for `d` shells / nctr>1.
- `cintx_cubecl::transform::c2s::cart_to_sph_1e::<f64>(cart, &mut sph, li, lj)` — the exact
  per-block cart→sph transform (input layout `j*nci+i`).
- `cintx_compat::raw::{eval_raw, RawApiId, ANG_OF, BAS_SLOTS, ATM_SLOTS, ...}`.

## Gotchas (carry into the real build)

- **STO-3G blocks are orientation-blind.** Every non-square s/p block has a unit axis
  (`ni==1` or `nj==1`), so i-fastest vs j-fastest is unobservable. Use a `d` shell
  (`p × d` = 3×6 cart / 3×5 sph) whenever a test must pin in-block orientation (D-07).
- **On the moment fixtures every component is populated** (81/81 at rank 81), so these
  fixtures cannot exercise a "legitimately-zero component correctly skipped" path.
- **`int1e_r` reads the gauge origin** (`drj = rj - env[PTR_COMMON_ORIG]`); set it to 0 for
  a clean `R_c·S` hand-check.
- **Multi-index tuples need EVERY axis >1.** A 3-/4-index layout test with any unit axis
  hides an adjacent-axis swap (the multi-index analog of the orientation-blind s×p block).
  Pick a quartet/triple like `(0,p)(0,d)(1,p)(1,d)` → extents `[3,6,3,6]` (spike 004).
- **nctr>1 tests must also span rank tiers.** The existing `moment_genctr` only covers
  rank-9; the contraction-major composition is rank-independent but should be probed across
  3/9/27/81 (spike 005). Contraction is the MAJOR within-axis index (`i_global = ci*di+ic`).
- **Spinor layout DIVERGES** — it is interleaved-complex, not real component-leading. Size
  buffers `rank*ni_sp*nj_sp*2` with `ni_sp = CINTcgto_spinor = 4l+2` (kappa=0); re/im is the
  fastest axis. Component-leading + ket-major i-fastest still hold *around* the interleave
  (spike 006). Set `KAPPA_OF` on spinor shells.
