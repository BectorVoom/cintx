---
id: grids-ipvip-rys-root-undersize-rocm
created: {
  timestamp: 2026-05-30T05:12:49.275Z
}
source: rocm device oracle run 2026-05-30
severity: warning
resolves_phase: null
---

# grids IPVIP derivative path undersizes the Rys-root arrays (panics on high-AM)

`test_int1e_grids_random_rocm_parity` panics: `index out of bounds: the len is 2 but
the index is 2` at `crates/cintx-cubecl/src/kernels/unstable/grids.rs:1292`
(`u_arr[n]` in `grids_contract_ip`). The function computes
`nrys_roots = (li + 1 + lj + 1) / 2 + 1` and loops `for n in 0..nrys_roots`, but the
`(u_arr, w_arr)` allocation in the `if nrys_roots == 1 { .. } else { .. }` branch sizes
the root arrays for fewer roots than `nrys_roots` for some (li,lj). The random ROCm
fixture hits an angular-momentum combo needing nrys_roots>=3 while u_arr has len 2.

PRE-EXISTING and UNRELATED to the 2026-05-30 session work: the code was introduced in
commit b9a4c6b (quick-260529-twi); the last grids.rs commit (6ba50c5) predates the
phase-23 base 3d4a714; zero session commits (phase 23 / WR-03 / CR-01) touched grids.rs.
The fixed-fixture `unstable_source_parity` (23/23) does not exercise this AM combo, so it
stayed latent until the random ROCm fixture surfaced it.

Fix: size `u_arr`/`w_arr` from the SAME `nrys_roots` used in the loop (or clamp the loop
to the allocated length) in `grids_contract_ip`; mirror the root-count handling already
used in the scalar/other grids deriv paths. Add a fixed high-AM (li,lj) ipvip fixture so
it is covered without relying on the random rocm gate. Relates to [[project_unstable_derivative_ports]].
