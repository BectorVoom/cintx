---
quick_id: 260530-k0s
status: complete
date: 2026-05-30
---

# Quick Task 260530-k0s: Fix grids derivative host Rys-root array undersize

## Self-Check: PASSED

## What shipped
`crates/cintx-cubecl/src/kernels/unstable/grids.rs`: the four host contraction fns
`grids_contract_{nuclear_like,ip,ipip,ipvip}` now size their Rys root/weight arrays from
`nrys_roots` via `rys_roots_host(nrys_roots as usize, x_boys)`, replacing the old
`if nrys_roots == 1 { rys_root1_host } else { rys_root2_host }` (which always produced a 2-element
array). `grids_contract_spvsp` delegates to `_ipvip`, so it is fixed transitively. Import cleaned.

## Root cause
The device `#[cube]` grids kernels wire only `rys_root{1,2}` (`GRIDS_MAX_DEVICE_NROOTS = 2`) and the
launcher falls back to these host fns for `nroots > 2` (lines ~1665/1693). But the host fns fetched
only 2 roots while looping `for n in 0..nrys_roots`, so `u_arr[n]` panicked
`index out of bounds: len 2 index 2` for high (li,lj) — surfaced by the random ROCm grids fixture.
`rys_roots_host` is byte-identical to `rys_root{1,2}_host` for nroots<=2 and correct for 3..=5
(fail-closed >5 = the separate deferred Wheeler todo).

## Verification
- `cargo test -p cintx-cubecl --lib` — 280 passed.
- `grids_random_rocm_parity` on `gfx1152` (ROCm 7.1.1) — **1/1 passed** (was panicking).
- `unstable_source_parity` (`--features cpu,unstable-source-api` + vendor) — **23/23** (nroots<=2 byte-identity preserved).

## Commits
- `fix(260530-k0s): size grids host Rys-root arrays to nrys_roots (nroots>2)` — grids.rs (25+/29-), `1395aae`.

## Deviations
None. Single-file mechanical fix as planned. Closed todo grids-ipvip-rys-root-undersize-rocm.
Note: nroots>=6 still panics in rys_roots_host (separate deferred Wheeler-fallback todo, unchanged).
