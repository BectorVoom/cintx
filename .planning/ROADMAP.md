
**Plans**: 4 plans
- [ ] 28-01-PLAN.md (wave 1) — HOST si transform: `apply_bra_si_block` (NEW `a_bra_cart2spinor_si` sign convention, NOT `apply_si_block`) + `cart_to_spinor_si_2d` (4-block gc input, ordinary ket reuse, KET→BRA transpose owned internally, spinor_len sizing) in c2spinor.rs
- [ ] 28-02-PLAN.md (wave 1, parallel) — DEVICE σ·p generic `#[cube]` assembler `kernels/sigma_p.rs` (rank-parameterized per D-03; emits pre-blocked gc_x/gc_y/gc_z/gc_1, scalar slot 0 for int1e_sp)
- [ ] 28-03-PLAN.md (wave 1, parallel) — manifest infra: int1e_sp_spinor row (oracle_covered=false, D-01) + vendor_int1e_sp_spinor FFI shim + SC#4 skipped-fixture guard assertion; no capi/legacy surface
- [ ] 28-04-PLAN.md (wave 2) — int1e_sp Spinor dispatch wiring (σ·p → si_2d, nctr>1) + build_kappa_spinor_fixture (kappa≠0 p+1/d−1, non-square, nctr=2) + heavy-atom fixture + si_transform_parity.rs end-to-end byte-identity vs vendor at atol=1e-12 (no flag flip, no-silent-skip)
**Research flag**: DISCHARGED — the D-06 design spike is the 28-RESEARCH.md `## Validation Architecture` section (Spike Targets A–E), verified against vendored `cart2sph.c`/`intor3.c`. Key finding: the 2D `c2s_si_1e` bra step uses `a_bra_cart2spinor_si` signs, which DIFFER from the existing `apply_si_block`; the new `apply_bra_si_block` must transcribe them verbatim.
