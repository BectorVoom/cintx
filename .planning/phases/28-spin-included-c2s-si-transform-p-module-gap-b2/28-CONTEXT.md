# Phase 28: Spin-Included `c2s_si` Transform + σ·p Module (Gap B2) - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the **spin-included (si) 2D cart→spinor transform** `cart_to_spinor_si_2d` (the 1e bra+ket analog of `cart_to_spinor_sf_2d`, matching libcint `c2s_si_1e` at `cart2sph.c:4947`), consuming the **4-block `gc_x/gc_y/gc_z/gc_1` G-tensor** (three Pauli-σ component blocks + the scalar), plus a companion **σ·p G-tensor assembler** (the 12-component Pauli `gout` emitter that produces the four `gc_*` blocks the si transform reads). Validate end-to-end byte-identity at atol=1e-12 against a **kappa-bearing relativistic oracle fixture** (FND-05).

This is **Gap B2** — the σ-coupling foundation. It is the hard prerequisite for every σ-operator family (Group 4 relativistic σ in Phase 29, GIAO×σ in Phase 30, gauge/Breit–Gaunt 2e in Phase 31). Phase 28 itself does **not** implement those family groups — it lands the transform + assembler + fixture and proves them through a single thin vehicle family.

**Already exists (do NOT rebuild):** the *single-block* si transform `cart_to_spinor_si` (`c2spinor.rs:392`) + `cart_to_spinor_iket_si` (`:449`) from Phase 12, exposed as covered helpers `CINTc2s_ket_spinor_si1`/`CINTc2s_iket_spinor_si1`. Phase 28 adds the **2D (full bra+ket integral) si transform** on top of that proven single-block coupling, exactly as `sf_2d` relates to the per-component sf step.

No capi enum variants, no legacy `cint*` wrappers (project new-family surface policy).
</domain>

<decisions>
## Implementation Decisions

### Flip scope (what moves to oracle_covered this phase)
- **D-01:** Phase 28 flips `oracle_covered=true` for **only the single validation-vehicle σ family** (`int1e_sp_spinor`, see D-02) — it is the registered byte-identity anchor that proves FND-05. **Every other σ family stays `UnsupportedApi`** and is deferred to Phase 29 (`int1e_spsp/spnucsp/sprinvsp/srsr/sr/srnucsr/sigma` and the 2e `spsp1/srsr1/ssp*/sps*/vsp*/spv*`). Rationale: give FND-05 a real registered oracle gate (the project's "byte-identity oracle IS the proof" value) without over-claiming coverage for families whose kernels don't land until Phase 29. This is the honest-scope reading of SC#4 ("σ families stay UnsupportedApi until this phase passes").

### Validation vehicle
- **D-02:** The end-to-end byte-identity proof (SC#3) is driven by **`int1e_sp`** — σ·p on the **bra only**. libcint `c2s_si_1e` mixes the Pauli σ on the bra (`a_bra_cart2spinor_si` over `gc_x/gc_y/gc_z/gc_1`) and uses the **ordinary** `a_ket_cart2spinor` on the ket, so `int1e_sp` directly emits the four `gc_*` blocks the si transform consumes — the thinnest family that exercises BOTH new pieces (the si_2d transform AND the σ·p gout assembler) with the least extra operator machinery. It is the building block `spsp/spnucsp/sprinvsp` all compose from in Phase 29. `int1e_spsp` (σ·p on both sides) and `int1e_sigma` (pure σ, no p) were rejected as the *first* vehicle: spsp drags ket-side σ·p into the first proof, sigma under-tests the named σ·p deliverable.

### σ·p assembler architecture
- **D-03:** Build the σ·p gout assembler as a **standalone, reusable, generic `#[cube]` emitter** (a dedicated σ/`gout_si` module) parameterized so Phase 29's whole σ-group (`sp`, `spsp`, `spnucsp`, `sprinvsp`, `sigma`, …) reuses it directly. Front-load the architecture — this phase exists precisely to be that foundation. (Rejected: "minimal now, generalize in 29" — would force rework when the family group lands.)
- **D-04 (host/device split — user-confirmed reading):** The **si transform** (`cart_to_spinor_si_2d`) is a **HOST** function in `c2spinor.rs`, mirroring `cart_to_spinor_sf_2d` (transforms run on the contracted cart staging buffer post-kernel). The **σ·p gout assembler** that emits `gc_x/gc_y/gc_z/gc_1` is a **DEVICE `#[cube]`** step, mirroring the existing nabla/gout gradient machinery. The spike (D-06) confirms the exact device→host buffer hand-off.

### Kappa-bearing fixture design
- **D-05:** The fixture **reuses Phase 27's D-08 adversarial geometry** (non-square bra/ket e.g. p×d, at least one shell with nctr>1) **but with genuine kappa≠0** (e.g. p with kappa=+1 → LT-only `j=l−1/2`, d with kappa=−1 → GT-only `j=l+1/2`) so the si transform is proven on the **non-`(4l+2)` spinor sizing path** (`di = 2l` or `2l+2`) that Phase 27's kappa=0 fixture structurally could not exercise. Added as a sibling `build_kappa_spinor_fixture` in `fixtures.rs` (alongside `build_adversarial_spinor_fixture` at `:209`). One fixture keeps every prior landmine (non-square, nctr>1) AND adds the kappa axis.

### Research spike (HARD GATE)
- **D-06:** Run a **full design spike before plan tasks are finalized** (B1 D-11 precedent + the roadmap's own research flag). The spike must nail, against **hand-checked vendor values**: (a) the `a_bra_cart2spinor_si` **4-block stride/ordering** (`gc_x/gc_y/gc_z/gc_1` are 4 contiguous blocks each of size `nf = nfi*nfj` cart — confirm `cart2sph.c:3920` + `:4947`); (b) the **bra-Pauli-mix / ket-ordinary** split (don't symmetrize the ket); (c) the device→host buffer hand-off between the `#[cube]` σ·p gout assembler and the host si_2d transform; (d) the kappa≠0 GT/LT-only sizing through `spinor_len`. Do **not** shortcut it.

### Claude's Discretion
- Exact molecule/element + kappa assignments for the fixture (subject to D-05 hard constraints: non-square, nctr>1 somewhere, kappa≠0).
- Internal module naming/factoring for the reusable σ·p assembler and the si_2d transform.
- Plan boundaries (e.g. spike → transform+assembler → fixture+parity).
- Exact `int1e_sp` gout component ordering — resolve from libcint `intor3.c` during the spike.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — **FND-05** (line 82): the Gap B2 requirement definition. Also REL-01..04 (Group 4, the downstream consumers, lines 109-112) and SPIN-02/SPIN-04 (the `ket_spinor_si` variant + kappa interpretation this completes, lines 25/27).
- `.planning/ROADMAP.md` §"Phase 28" (lines 646-659) — Goal, the 4 success criteria, dependency on Phase 12, and the design-spike research flag (line 659).

### Prior phase context (decided conventions to inherit, do NOT re-decide)
- `.planning/phases/12-real-spinor-transform-c2spinor-replacement/12-CONTEXT.md` — Phase-12 D-01..D-08: CG coefficient source (`c2spinor.c` `g_c2s_*`), `c2spinor_coeffs.rs` location, the four `sf/iket_sf/si/iket_si` code paths (the **single-block si already lives here**), interleaved `[re,im,…]` staging, kappa→block dispatch (kappa<0=GT, kappa>0=LT, kappa==0=both).
- `.planning/phases/27-spinor-derivative-transform-gap-b1/27-CONTEXT.md` — sibling **Gap B1**. Inherit: KET→BRA transpose ownership inside the transform (D-06), adversarial-fixture rationale (D-08), no-silent-skip coverage assertion (D-10), design-spike-as-hard-gate (D-11), new-family surface policy. Do NOT re-decide these.

### Spinor transform implementation (the file this phase extends)
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — existing `cart_to_spinor_si` (L392) + `cart_to_spinor_iket_si` (L449) single-block transforms; the single-block si accumulation (L124-176, formula from `CINTc2s_ket_spinor_si1`); `cart_to_spinor_sf_2d` (L531, the **structural template** for the new `cart_to_spinor_si_2d`); `spinor_len(l, kappa)` (L25, GT/LT/both sizing).
- `crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` — CG coupling coefficient tables.
- `crates/cintx-cubecl/src/transform/mod.rs` — `apply_representation_transform()`; Spinor is dispatched explicitly in kernel launchers, NOT through the generic transform arm.

### Kernel launchers & σ·p machinery (call sites + reuse)
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — 1e launch path (imports `cart_to_spinor_sf_2d`/`sf_derivative_2d`; the new σ·p gout + `si_2d` dispatch wires in here).
- `crates/cintx-cubecl/src/kernels/center_4c1e.rs` — has `test_device_matches_host_spsp` (L1878), an existing spsp device/host harness to mine for the σ·p pattern.
- The nabla/gout gradient machinery (`one_electron.rs` / `f12.rs` `nabla1*`) produces the ∇ cart blocks the σ·p assembler combines into `gc_x/gc_y/gc_z`.

### Manifest & coverage
- `crates/cintx-ops/src/generated/api_manifest.rs` + `compiled_manifest.lock.json` — ManifestEntry rows for `int1e_sp/spsp/sigma/…_spinor` (`component_rank`, `forms`, `oracle_covered`). The lock is the source of truth; edits auto-sync both audit sides.
- `xtask/src/oracle_covered_update.rs` — the flip mechanism + deferral notes; SC#4's "refuse to flip a σ family whose only fixture was `skipped`" guard belongs here.

### Oracle / vendor parity infrastructure
- `crates/cintx-oracle/src/vendor_ffi.rs` — vendored libcint FFI (add `vendor_int1e_sp_spinor`).
- `crates/cintx-oracle/src/compare.rs` — oracle comparison, atol=1e-12.
- `crates/cintx-oracle/src/fixtures.rs` — `build_adversarial_spinor_fixture` (L209, the D-08 kappa=0 template); add `build_kappa_spinor_fixture` sibling (D-05).
- `crates/cintx-compat/src/transform.rs` — `CINTc2s_ket_spinor_si1` (L178) / `CINTc2s_iket_spinor_si1` (L225) delegating to `cart_to_spinor_si`; the covered single-block helper surface.

### Upstream reference (byte-authoritative)
- `libcint-master/src/cart2sph.c` — `c2s_si_1e` (L4947), `a_bra_cart2spinor_si` (L3920), `a_ket_cart2spinor`. The 4-block `gc_*` layout + bra-Pauli-mix / ket-ordinary split.
- `libcint-master/src/autocode/intor3.c` — `int1e_sp_spinor` / `int1e_spsp_spinor` drivers (`CINT1e_spinor_drv(..., &c2s_si_1e, ng)`). Authoritative for the `int1e_sp` gout + driver wiring.

### Skill
- `.claude/skills/spike-findings-cintx/SKILL.md` — spinor interleaved-complex layout (`rank*ni_sp*nj_sp*2`, re/im fastest, `ni_sp=4l+2` @ kappa=0; **kappa≠0 → 2l or 2l+2**), component-leading + ket-major. Load before touching c2s/output layout.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **The single-block si transform already exists and is covered** (`cart_to_spinor_si`, `c2spinor.rs:392`; helper `CINTc2s_ket_spinor_si1`). `cart_to_spinor_si_2d` builds on it exactly as `sf_2d` relates to the per-component sf step — new code is the bra+ket *driver*, not the σ-coupling math.
- `cart_to_spinor_sf_2d` (L531) is the structural template for `cart_to_spinor_si_2d` (bra step → ket step → `zcopy_ij` interleave).
- `spinor_len(l, kappa)` (L25) already handles GT (kappa<0), LT (kappa>0), and both (kappa=0) sizing — drives the kappa≠0 buffer sizing the D-05 fixture stresses.
- nabla/gout gradient machinery (`one_electron.rs` / `f12.rs` `nabla1*`) produces the ∇ cart blocks the σ·p assembler folds into `gc_x/gc_y/gc_z`.
- `center_4c1e.rs::test_device_matches_host_spsp` (L1878) — an existing spsp device/host harness to mine.

### Established Patterns
- **transforms (`c2spinor.rs`) are HOST fns** on the contracted cart staging post-kernel; **gout/nabla is DEVICE `#[cube]`** → si_2d = host (D-04), σ·p gout assembler = device `#[cube]` (D-03).
- New-family surface = manifest + RawApiId + kernel + vendor-FFI + oracle ONLY; no capi/legacy.
- Vendor parity double-gated: `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`; without both it silently skips (determinism-only). Add the no-silent-skip assertion (Phase 27 D-10 pattern).
- Interleaved `[re0,im0,re1,im1,…]` complex staging, column-major (j outer, i inner); oracle compares the flat buffer directly.

### Integration Points / Landmines
- **libcint `c2s_si_1e` mixes σ on the BRA only** (`a_bra_cart2spinor_si` over `gc_x/y/z/1`); the ket is the **ordinary** `a_ket_cart2spinor`. Do NOT symmetrize the ket.
- `gc_x/gc_y/gc_z/gc_1` are **4 contiguous blocks**, each size `nf = nfi*nfj` cart — the σ·p assembler output layout must match (confirm in spike).
- **kappa≠0 changes spinor sizing** (`di = 2l` or `2l+2`, NOT `4l+2`) — the new axis this phase stresses; buffer sizing MUST come from `spinor_len`, never a hardcoded `4l+2`.
- **KET→BRA transpose** landmine — own it inside `si_2d` (per B1 D-06); device cart blocks are KET-major.
- **nctr>1 column/row-major coeff transpose** (D-08 carryover) — the fixture keeps an nctr>1 shell.
- **component_rank truncation** — verify rank values in the lock for any flipped row.
</code_context>

<specifics>
## Specific Ideas

The user wants the same adversarial rigor as Gap B1: the kappa-bearing fixture must keep **non-square + nctr>1** AND add **genuine kappa≠0** — precisely the one thing B1's kappa=0 fixture could not test (the GT/LT-only `2l`/`2l+2` sizing). The reusable σ·p `#[cube]` module is to be **front-loaded now** because this phase IS the foundation for Groups 4/6 and the GIAO×σ slice of 5. The design spike is a **hard gate** — do not shortcut it; nail the `gc_*` 4-block stride/ordering and the bra-Pauli/ket-ordinary split against hand-checked vendor values before committing plan tasks. libcint `c2s_si_1e` is the byte-authoritative reference.
</specifics>

<deferred>
## Deferred Ideas

- **All Group-4 σ families beyond the `int1e_sp` vehicle** (`int1e_spsp/spnucsp/sprinvsp/srsr/sr/srnucsr/sigma`, 2e `spsp1/srsr1/ssp*/sps*/vsp*/spv*`) → **Phase 29**. They reuse the Phase-28 si_2d transform + reusable σ·p module; their kernels land in 29.
- **The `iket_si` (`*i` / imaginary, `c2s_si_1ei`) 2D variant and the 2e si transforms** (`c2s_si_2e1/2e1i/2e2/2e2i`) — needed for GIAO×σ (Phase 30) and gauge/Breit–Gaunt 2e (Phase 31). The single-block `iket_si` already exists; the 2D/2e si drivers are out of Phase 28 scope (Phase 28 = 1e `si_2d` + σ·p assembler foundation only).
- **GIAO×σ slice** (Phase 30) and **gauge/Breit–Gaunt 2e** (Phase 31) — both gate on this phase's σ·p machinery.
</deferred>

---

*Phase: 28-spin-included-c2s-si-transform-p-module-gap-b2*
*Context gathered: 2026-05-31*
