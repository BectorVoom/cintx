---
id: grids-ipvip-rys-root-undersize-rocm
created: 2026-05-30
source: rocm device oracle run 2026-05-30
severity: warning
resolves_phase: null
status: resolved
resolved: 2026-05-30
resolved_by: quick task 260530-k0s (commit 1395aae)
---

# RESOLVED: grids derivative host fallback undersized the Rys-root arrays

`grids_contract_{nuclear_like,ip,ipip,ipvip}` computed the correct
`nrys_roots = (Σl)/2 + 1` (up to 5) but fetched only TWO roots via
`if nrys_roots == 1 { rys_root1_host } else { rys_root2_host }`. The device kernels wire
only rys_root{1,2} (GRIDS_MAX_DEVICE_NROOTS=2) and fall back to these host fns for
nroots>2, so `for n in 0..nrys_roots { u_arr[n] }` panicked
`index out of bounds: len 2 index 2` for high (li,lj) — surfaced by
`test_int1e_grids_random_rocm_parity`. `grids_contract_spvsp` delegates to `_ipvip` (covered).

## Fix (quick task 260530-k0s, commit 1395aae)
Replaced the four 2-root if/else blocks with
`let (u_arr, w_arr) = rys_roots_host(nrys_roots as usize, x_boys);` (rys.rs:3235 — dispatches
rys_root{1..5}_host, returns length-`nroots` Vecs; byte-identical to rys_root{1,2}_host for
nroots<=2, correct for 3..=5, fail-closed panic for >5 = the separate deferred Wheeler todo
[[rys-nroots-ge6-wheeler-fallback]]). Import cleaned (dropped now-unused rys_root{1,2}_host).
Device #[cube] kernels + nroots>GRIDS_MAX_DEVICE_NROOTS routing unchanged.

Verified: `grids_random_rocm_parity` passes on gfx1152 (was panicking); `unstable_source_parity`
23/23 (nroots<=2 byte-identity preserved); `cintx-cubecl --lib` 280.
