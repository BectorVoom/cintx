# Pitfalls Research — v1.4 Full libcint 6.1.3 Family Parity

**Domain:** Adding the ~140 remaining libcint 6.1.3 integral families to the existing cintx manifest→kernel→raw→safe-API→oracle pipeline (6 groups: remaining 1st-derivatives, Hessian/higher-order, position/multipole moments, relativistic spin-operator, GIAO/magnetic NMR, gauge/Breit–Gaunt 2e).
**Researched:** 2026-05-27
**Confidence:** HIGH for the source-grounded mechanisms (ng[] headroom, c2s_si 4-block input, c2s_zset0 imaginary zeroing, PTR_COMMON_ORIG reads, ALL_CINT vs ALL_CINT1E macro split, oracle-harness shape) — all read directly from `libcint-master` and the live cintx tree. MEDIUM for exact per-family component counts of the long tail (origj/adjacent families) which need a final enumeration pass.

> File note: `.planning/research/PITFALLS.md` is the immutable v1.2 record. This is the v1.4 milestone-scoped sibling. It builds on `STACK-v1.4.md`, `FEATURES-v1.4.md`, `ARCHITECTURE-v1.4.md`.

> Reading guide: pitfalls are ordered by how badly they would **silently pass a weak gate but fail true byte-identity**. The four most dangerous (P1–P4) all produce a green-looking artifact that is numerically wrong or vacuous. Each pitfall names the group(s) and the foundation/phase that must own it.

---

## Critical Pitfalls

### Pitfall 1: Reading imaginary GIAO/magnetic output (and σ families) as real f64 — silent zero or silent half-answer

**What goes wrong:**
Every Group-5 family (`int1e_ig*`, `g*`, `govlp`, `gnuc`, `a01gp`, `ia01p`, `int2e_g*`) and several Group-4/6 families are **complex-valued, and the physically meaningful content is the imaginary part**. libcint's drivers prefix these with `c2s_zset0` / `c2s_sf_1ei` / `c2s_si_*i` (the trailing `i` = imaginary). Verified in source: `CINTgout1e_int1e_igovlp` writes a pure cross-product `gout = -c[1]*s[2]+c[2]*s[1]` (the `i·(c×s)` antisymmetric form) and the driver routes through `c2s_zset0(out…)` which zeroes one component (`cart2sph.c:4737`). If cintx stages these into a real `[f64]` buffer and the oracle compares the real lane, the answer is silently zero (or only the real half), and `any_nonzero` sentinels and `max_abs_error` against a *also-zeroed* reference can both pass.

**Why it happens:**
Groups 1–3 (and all base families to date except spinor) are real f64. The team's mental model and the `assert_flat_buffer_contract` harness check (`compare.rs:278`) only enforce the complex-interleaved layout when `representation == "spinor"`. But GIAO **cart and sph** representations are *also* complex. A developer naturally reuses the real overlap-staging path, the kernel compiles, determinism tests pass, and the imaginary content never lands in the buffer.

**How to avoid:**
- Treat "complex output" as an explicit capability gate decided per family from the libcint driver routing (grep the `c2s_*` callee + `zset0` prefix), NOT from the representation string. The `complex_interleaved` field already exists on `OutputLayoutMetadata` (`planner.rs:60`) — set it for every `ig*`/`g*`/`a01*`/`sigma` family regardless of cart/sph/spinor.
- Extend `assert_flat_buffer_contract` so the interleaved-pair contract fires on `complex_interleaved == true` (not on `representation == "spinor"`). A complex cart/sph fixture that is staged as real-only must FAIL the contract, not pass.
- Stage GIAO output as interleaved re/im f64 pairs and have the vendor wrapper pass the same `f64` interleaved buffer to the `double complex *out` libcint symbol (the exact pattern already used by `vendor_int1e_ovlp_spinor`, `vendor_ffi.rs:1316`, doc-commented `ni_sp*nj_sp*2`).

**Warning signs:**
- A GIAO/magnetic kernel "passes" with an all-zero or suspiciously-small max error.
- The staging buffer length equals `n_components × ni × nj` (real) instead of `2 × n_components × ni × nj` (interleaved complex).
- The fixture's gauge origin is zero (see Pitfall 3) AND the family is GIAO — double-trivial pass.

**Phase to address:**
A dedicated **complex/imaginary-output capability phase** immediately before Group 5 (the orbital/spin-free GIAO subset). FEATURES-v1.4 flags this as one of the two cross-cutting unknowns. Group 4 `sigma`/SOC and Group 6 inherit the same machinery.

---

### Pitfall 2: Applying the scalar `cart_to_spinor_sf` transform to σ-operator families instead of the missing 4-block `c2s_si` transform

**What goes wrong:**
Every Group-4 relativistic spin-operator family (`spsp`, `spnucsp`, `sprinvsp`, `srsr`, `sigma`, `sp`), the GIAO×σ slice of Group 5 (`spg*`, `*_sa10sp`), and ALL of Group 6 (`int2e_gauge_r1/r2_*`, Gaunt `ssp/sps`) require the spin-**included** transform `c2s_si_1e` / `c2s_si_2e1i`/`c2s_si_2e2i`. Verified in `cart2sph.c:4947`: `c2s_si_1e` consumes a **4-block G-tensor** `gc_x, gc_y, gc_z, gc_1` (the three Pauli-σ component blocks plus the scalar block, each `nf*i_ctr*j_ctr` long) via `a_bra_cart2spinor_si`. cintx only has the **scalar single-block** `cart_to_spinor_sf_*` (`c2spinor.rs`). Reusing `sf` on a σ family silently produces a non-libcint spinor result — the σ-coupling is simply absent. The kernel is written and dispatchable, but the byte-identity gate rejects it *after the fact* (or, worse, passes if the fixture has no spinor shells — Pitfall 4).

**Why it happens:**
Phase 12 shipped a *real spinor transform* (scalar, Clebsch–Gordan). The natural assumption is "spinor support exists, register the σ family and route through it." The 4-block input contract is invisible unless you read `c2s_si_1e` and notice `gc_x/gc_y/gc_z/gc_1`. Confirmed in the breit spinor driver: `int2e_gauge_r1_ssp1ssp2_spinor` calls `CINT2e_spinor_drv(…, &c2s_si_2e1i, &c2s_si_2e2i)` (`breit1.c:211`), NOT the `sf` variant.

**How to avoid:**
- Land **Gap B2** (the `c2s_si_*` spin-included transform + a σ·p G-tensor assembler producing the 4 blocks `gc_x/y/z/1`) as its own foundation phase BEFORE any σ-operator kernel. Keep σ/spinor variants `UnsupportedApi` (the existing R5/D-03 guard, `one_electron.rs:901-908`) until the transform exists and is exercised end-to-end against a kappa-bearing fixture.
- Make the kernel emit 4 contiguous G-tensor blocks for σ families (mirroring the `gc_x/y/z/1` layout `c2s_si_1e` reads in order) and assert the block count at the transform boundary.
- Do NOT flip `oracle_covered` for any σ family until a real spinor fixture passes (Pitfall 4 + 9).

**Warning signs:**
- A `spsp`/`sigma`/`gauge_*` spinor kernel routes through `cart_to_spinor_sf_*`.
- The G-tensor passed to the transform is a single block, not four.
- Spinor byte-identity fails with a structured (non-noise) error pattern — the σ-cross terms are systematically wrong.

**Phase to address:**
**Gap B2 foundation phase** (largest single addition in v1.4) gating Groups 4, 6, and the GIAO×σ slice of 5. Sequence it after the cheaper cart/sph work (ARCHITECTURE-v1.4 build order steps 6–7).

---

### Pitfall 3: Not reading `PTR_COMMON_ORIG` (env[1..3]) for moments and GIAO — and validating only against a zero-gauge-origin fixture

**What goes wrong:**
The gauge/common origin `PTR_COMMON_ORIG = env[1..3]` is documented in `raw.rs:34` but **never read** by cintx. Decisive source finding: it is needed not only by the `_origj` moment variants but by the **plain** multipole moments too. `CINTgout1e_int1e_r` computes `drj[k] = envs->rj[k] - envs->env[PTR_COMMON_ORIG+k]` and feeds `G1E_RCJ` (= `CINTx1j_1e(…, drj, …)`, `g1e.h:57`). So `int1e_r`, `int1e_rr`, `int1e_r2`, `int1e_z`, …, every Group-3 moment with an `r` operator, AND every Group-5 GIAO family (gauge factor `c = rirj` referenced against the common origin) read this slot. If cintx leaves `PTR_COMMON_ORIG` unread (effectively origin = 0), the answer is wrong for any non-zero gauge origin — but **passes trivially on H2O/STO-3G, which uses the default zero origin**.

**Why it happens:**
ARCHITECTURE-v1.4 (and intuition) stated "non-origj moments use bra/ket centers only, no env slot." Source contradicts this: even plain `int1e_r` subtracts the common origin. Because the default fixture origin is zero, `rj - 0 = rj` and the kernel looks correct. This is the same class as the historic `PTR_RANGE_OMEGA=env[8]` global-slot gap and the unread-slot trap in the priors.

**How to avoid:**
- Implement **Gap A** (`common_orig: Option<[f64;3]>` on `OperatorEnvParams` + an `env[1..4]` read block in `eval_raw`, modeled verbatim on the Phase-21 `PTR_RINV_ORIG` block `raw.rs:599-616`) and wire it into the moment + GIAO G-tensor as the `drj` subtraction.
- Add a **non-zero gauge-origin fixture** (H2O/STO-3G with `env[PTR_COMMON_ORIG] != 0`) and gate every Group-3 moment and Group-5 GIAO parity on it. A zero-origin-only test is a vacuous gate for this slot.
- Validator gate: families that read the slot must error (typed) if the safe-API caller never set the origin, rather than silently defaulting to 0 (mirrors `validate_rinv_orig_env_params`).

**Warning signs:**
- A moment/GIAO kernel passes on H2O/STO-3G but the `eval_raw` env read map has no `PTR_COMMON_ORIG` block.
- `int1e_r` cart/sph parity passes but no fixture ever set a non-zero common origin.
- The safe-API builder has no `with_common_origin` setter.

**Phase to address:**
**Gap A foundation phase** (cheap, isolated, leads the milestone — ARCHITECTURE-v1.4 step 1). Gates Groups 3 and 5. The gauge-origin fixture is shared by both.

---

### Pitfall 4: Validating every group on H2O/STO-3G only — the fixture cannot exercise spinor shells or non-zero gauge origins

**What goes wrong:**
H2O/STO-3G has no kappa-bearing (spinor) shells and uses a zero gauge origin. Reusing `build_h2o_sto3g()` for Groups 4/6 means the σ/spinor path is **never executed** (the spinor representation is filtered out or returns `UnsupportedApi`, so the parity loop records a "skip" that looks like a pass). Reusing it for Groups 3/5 means `PTR_COMMON_ORIG` (Pitfall 3) is multiplied by zero. Two whole groups' physical correctness goes unverified while the manifest shows `oracle_covered=true`.

**Why it happens:**
The fixture is the established default; it is wired into the existing matrix path; adding a new fixture is more work. The `skipped` flag (`compare.rs:94`) is explicitly designed to record non-evaluated fixtures as "passing without numeric obligation" — convenient, but it means a missing spinor fixture turns into a green skip rather than a red gap.

**How to avoid:**
- Add **two new fixtures** before the dependent groups: (a) a **kappa-bearing relativistic fixture** (a molecule with spinor shells — e.g. a heavy atom in a minimal relativistic basis) for Groups 4/6 and the GIAO×σ slice; (b) the **non-zero gauge-origin fixture** for Groups 3/5.
- Make `oracle-covered-update` refuse to flip `oracle_covered=true` for a σ/spinor family whose only fixture was `skipped` (the doc comment at `compare.rs:90-94` already states "MUST NOT treat a skipped fixture as oracle-covered" — enforce it mechanically, not by convention).
- Assert per-group that the parity loop produced N>0 *evaluated* (non-skipped) fixtures for the group's headline families.

**Warning signs:**
- `oracle_covered=true` on a `*_spinor` family with `skipped=true` in the matrix report.
- Group 4/6 parity reports show "running 0 tests" or all-skips.
- No `fixtures.rs` entry with a spinor shell or non-zero `env[1]`.

**Phase to address:**
Fixture additions land **inside the Gap A phase** (gauge-origin fixture) and the **Gap B2 phase** (relativistic fixture), before the groups that consume them. ARCHITECTURE-v1.4 Anti-Pattern 3.

---

### Pitfall 5: Under-sizing the G-tensor angular-momentum headroom for higher-order derivatives and high-order moments

**What goes wrong:**
The derivative/moment recurrence requires the base G-tensor to be built at **elevated angular momentum**, and the elevation differs by order AND by which center carries the operator. Ground truth from libcint `ng[]` (first four elements = `li_inc, lj_inc, lk_inc, ll_inc` headroom):

| Family class | source `ng[]` | headroom | components |
|---|---|---|---|
| 1st-deriv (grad1) | `{1,0,0,0,1,1,1,3}` | bra +1 | 3 |
| Hessian ipip (hess) | `{2,0,0,0,2,1,1,9}` | bra +2 | 9 |
| 3rd-order (deriv3) | `{3,0,0,0,3,1,0,27}` | bra +3 | 27 |
| 4th-order ipiprinvipip (deriv4) | `{2,2,0,0,4,1,0,81}` | bra +2, ket +2 | 81 |
| **dipole `int1e_r`** (intor1) | `{0,1,0,0,1,1,1,3}` | **ket +1** | 3 |
| **hexadecapole `int1e_rrrr`** | `{0,4,0,0,4,1,1,81}` | **ket +4** | 81 |
| int2e ipip w/ aux (grad2) | `{2,1,1,1,5,4,4,3}` | **all four centers** | 3 |

Two distinct traps: (1) using the Phase-21 `li+1` bra-elevation for a 4th-order `{2,2,…}` family under-allocates the G-tensor and the HRR reads past the built range (garbage or panic); (2) **moments raise the KET (`lj`), not the bra** — copying the derivative bra-elevation logic for `int1e_r/rr/rrr/rrrr` builds headroom on the wrong index and produces a transposed/wrong result.

**Why it happens:**
The Phase-21 engine proved `li+1` (bra +1) works; the natural generalization is "raise the bra by order." But deriv4 raises *both* bra and ket by 2, and multipole moments raise *only the ket*. The `±2 j-level HRR headroom` prior already burned the team once on kinetic.

**How to avoid:**
- Drive the planner's G-tensor sizing from the per-family `ng[]` headroom tuple read from the source `_optimizer`, not from a single "order" scalar. Encode `(li_inc, lj_inc, lk_inc, ll_inc)` per family in the manifest/kernel and size `g_size` at `l + inc` per center.
- Add a unit assertion that the built G-tensor span ≥ the maximum index the gout reads (deriv4 reads g0..g15, each `g_size*3`).
- For moments specifically, write a regression that `int1e_r` headroom lands on the ket and that the result equals the transpose-consistent dipole.

**Warning signs:**
- A 3rd/4th-order or hexadecapole kernel panics on HRR index, or passes on s/p shells but fails on d/f (where elevated `l` exceeds the built range).
- Moment results are transposed relative to the vendor.
- The `if dst < staging.len()` scatter guard (WR-03) silently drops the high components instead of erroring (see Pitfall 8).

**Phase to address:**
**Group 2** (Hessian/higher-order) for the bra/ket multi-center elevation; **Group 3** for the ket-side moment elevation. Group 2 extends the Group-1 engine; the headroom tuple must be plumbed before either.

---

### Pitfall 6: Rys `nroots > 5` overflow on high-angular-momentum and high-order-derivative quartets

**What goes wrong:**
Higher-order derivatives (Group 2) elevate angular momentum by 2–3 on one or more centers, and relativistic σ·p families (Group 4) and Breit/Gaunt 2e (Group 6) push effective angular momentum higher still. The number of Rys roots scales with total angular momentum; cintx has a **fail-closed guard at `nroots > 5`** (the Phase-21 guard `two_electron.rs:642`) because the Wheeler/`nroots>=6` fallback is not implemented (pending todo `rys-nroots-ge6-wheeler-fallback`). For a `{2,2,…}` 4th-order family on d/f shells, or a Breit quartet, the quartet's root count exceeds 5 and the kernel returns `UnsupportedApi`.

**Why it happens:**
The guard is correct fail-closed behavior, but the families in Groups 2/4/6 hit it far more often than the Phase-21 first-derivatives did, because they raise angular momentum by more. A naive "register and expect parity" plan will see many `UnsupportedApi` returns and may mistake them for kernel bugs.

**How to avoid:**
- Decide explicitly per group whether `nroots>=6` must be supported. If the milestone's full-parity claim requires high-l quartets for these families, the `rys-nroots-ge6-wheeler-fallback` todo becomes a **hard prerequisite** for Groups 2/4/6 (not optional).
- If high-l is out of scope for v1.4, keep the fail-closed guard and document the `UnsupportedApi` envelope explicitly in the manifest/REQ so the gate expects it (matching the existing contract), and size fixtures to stay within nroots≤5.
- Add a determinism test that the guard triggers (returns the typed error) rather than producing wrong numbers, for a known high-l quartet.

**Warning signs:**
- Many `UnsupportedApi` returns for d/f quartets in Groups 2/4/6.
- A high-l fixture silently produces wrong numbers (guard bypassed) instead of erroring.

**Phase to address:**
Surfaced in **Group 2**; resolved either by the **Wheeler `nroots>=6` fallback** (own prerequisite phase if full high-l parity is required) or by an explicit scoped `UnsupportedApi` envelope. Re-checked in Groups 4 and 6.

---

### Pitfall 7: Component layout / F-order transpose on ×9/×27/×81 tensors and the σ 12-component Pauli pattern

**What goes wrong:**
libcint emits the component axis as **component-leading** (`[ncomp, …]` in column-major/F-order). Phase 21 already hit this for the 3-component gradient (`[3, nl, nk, nj, ni]`). For Group 2 the tensor jumps to 9/27/81 and the inner ordering matters: e.g. `int1e_rr`'s 9 components are `xx,xy,xz,yx,…` and the Hessian `ipip` 3×3 needs a specific column-major reorder (cintx's `gout_ipip1` already does a "column-major 3×3 reorder" per STACK-v1.4). For σ families the gout writes a **12-component Pauli pattern** (`int1e_sigma` writes `n*12` = 3 σ-directions × 4 spin-block components; `int1e_sp` writes `n*4`) — a layout entirely unlike the simple tensor families. Getting the component stride or the σ-block ordering wrong yields a result that is element-wise-permuted: the same numbers in the wrong slots, which a magnitude-only or unsorted comparison can miss but byte-identity rejects.

**Why it happens:**
The component-leading F-order is non-obvious and easy to invert; the higher-rank tensors compound it; and the σ 12-slot Pauli pattern is a special layout (`CINTgout1e_int1e_sigma` writes a fixed `±s` Pauli pattern, intor3.c) that does not match the derivative tensor convention at all.

**How to avoid:**
- Reuse the Phase-21 element-for-element byte-identity comparison as the layout gate (the test comment at `two_electron_ip1_parity.rs:14-17` confirms "element-for-element comparison IS the F-order layout gate"). Do NOT add any sort/abs-only comparison shortcut.
- For each new tensor family, copy the component ordering from the libcint gout (the `gout[n*9+k] = …` index map in intor1.c / hess.c) verbatim, not from intuition.
- For σ families, port the 12-slot Pauli pattern from `CINTgout1e_int1e_sigma` exactly and assert the block count at the transform boundary (ties to Pitfall 2).

**Warning signs:**
- Parity fails with the correct *set* of magnitudes but in permuted positions.
- A `rr`/`rrr` family passes on s-shells (where ordering is trivial) but fails on p/d.
- σ output length is not a multiple of 12 (1e) or the expected Pauli block size.

**Phase to address:**
**Group 2** (×9/27/81 ordering), **Group 3** (`rr`/`rrr`/`rrrr` Cartesian-component order), **Group 4** (σ 12-component Pauli pattern). Each group's parity test must be element-for-element.

---

### Pitfall 8: Silent partial-write scatter guards (`if dst < staging.len()`) masking component-count / headroom regressions

**What goes wrong:**
Every Phase-21 gradient scatter loop guards the write with `if dst < staging.len() { staging[dst] = … }` (WR-03 from the Phase-21 review: `two_electron.rs:783,824`, `center_3c2e.rs:497,531`, `one_electron.rs:1026,1050`). When `dst >= staging.len()` the value is **silently dropped**. For the v1.4 high-rank families (9/27/81 components), a planner/manifest `component_rank` mismatch — exactly the class of bug CR-01 was — produces a quietly truncated tensor (some components zero) that still passes the `any_nonzero` sentinels and may pass a tolerance check against a reference that the same bug also truncated. This directly violates the CLAUDE.md "no partial writes / fallible allocation + typed failure" contract.

**Why it happens:**
The guard was added defensively for the 3-component case and copy-propagated. As component counts grow 9×–27×, the chance of a sizing mismatch grows, and the guard converts a hard error into a silent zero.

**How to avoid:**
- Before the scatter loop, assert/return `BufferTooSmall` (typed) if `staging.len() < expected_component_leading_size` (e.g. `ncomp * di*dj*dk*dl`), then index `staging[dst]` unconditionally. The guard becomes a hard precondition, not a per-element silent drop.
- This is a prerequisite cleanup before the high-rank groups, because the failure mode is invisible at rank 3 and disastrous at rank 81.

**Warning signs:**
- A high-rank family passes determinism + sentinel but the component count in the manifest disagrees with the kernel's emitted count.
- Some tensor components are exactly zero where the vendor is nonzero.

**Phase to address:**
Cleanup at the start of **Group 2** (first high-rank group); enforced for all subsequent groups. Carry the WR-03 finding forward as a blocking item, not a deferred warning.

---

## Moderate Pitfalls

### Pitfall 9: Vendor FFI signature mismatch — complex `double complex *out` vs real `double *out`, and the ALL_CINT vs ALL_CINT1E optimizer-arg split

**What goes wrong:**
Two distinct signature traps for the oracle wrappers:
1. **Complex output:** σ/GIAO/Breit families return into `double complex *out`. The cintx vendor wrapper must pass an interleaved `f64` buffer of length `2 × ncomp × …` to that symbol (the `vendor_int1e_ovlp_spinor` pattern, `vendor_ffi.rs:1316`, doc `ni_sp*nj_sp*2`). Passing a real-sized buffer or comparing the real lane only re-triggers Pitfall 1 at the harness level.
2. **Optimizer-arg arity:** libcint has two wrapper macros (verified `misc.h:35,63`). `ALL_CINT(NAME)` generates `cNAME_cart/sph/spinor` **with an `opt` arg + `_optimizer` siblings** (used by `int2e_*`, `int3c2e_ip1/ip2`, `int2c2e_ip1`, breit/gauge). `ALL_CINT1E(NAME)` generates wrappers **without `opt` and without `_optimizer`** (used by `int1e_r`, `int1e_ipovlp/ipkin/ipnuc/iprinv`, most 1e moments). A **third rule**: ECP families (`ECPscalar_iprinv`) have **no cint* wrapper at all** (legacy.rs:285). Mapping a 1e family to the `ALL_CINT` shape (or vice-versa) yields a link error or a wrong-arity FFI call.

**How to avoid:**
- Per family, read the trailing macro line in the autocode (`ALL_CINT(int2c2e_ip1)` vs `ALL_CINT1E(int1e_r)`) and pick the matching wrapper shape + `_optimizer` presence. The existing `misc_wrapper_macro` test map (legacy.rs:378) is the precedent — extend it, do not guess.
- For complex families, model the vendor wrapper on `vendor_int1e_ovlp_spinor` and size the interleaved buffer `2×`.
- ECP-style families get no legacy cint* wrapper; route them through the safe-API/raw path only (the Phase-21 ECPscalar precedent).

**Warning signs:**
- Link error for `cNAME_optimizer` on an ALL_CINT1E family.
- A complex vendor wrapper buffer sized `ncomp×…` instead of `2×ncomp×…`.
- `BufferTooSmall` from the harness on a complex family.

**Phase to address:** Every group's oracle-harness task; the macro-shape map extension is a small per-group step.

### Pitfall 10: Symbols absent from `cint_funcs.h` need supplemental bindgen `extern` decls AND the autocode `.c` file added to `cc::Build`

**What goes wrong:**
Many v1.4 symbols are declared only in their autocode `.c`, not in `cint_funcs.h` (which has 570 declared symbols but not, e.g., the giao/cg 2e block or the gauge/Gaunt set). Two coupled steps are required and easy to half-do: (a) add the `.c` file to the oracle `cc::Build` chain (`build.rs:181-253`), and (b) add an `extern CINTIntegralFunction <symbol>;` to the supplemental header (`build.rs:265-345`) **and** extend the bindgen allowlist regex (`build.rs:358`). Doing only (a) → the symbol links but bindgen never generates the FFI binding (no Rust wrapper). Doing only (b) → bindgen generates a binding to an unresolved symbol → link error. The currently compiled set is missing `autocode/intor4.c` (giao/cg 2e + spin 2e), `deriv3.c`, `deriv4.c`, `gaunt1.c`, `breit1.c` — all needed by Groups 2/4/5/6.

**How to avoid:**
- For each new symbol: (1) confirm whether it is in `cint_funcs.h` (`grep`); if not, add the `extern CINTIntegralFunction` decl to the suppl header; (2) ensure its defining `.c` is in `cc::Build`; (3) add the bare symbol to the allowlist regex. Treat these three as an atomic unit per family.
- ARCHITECTURE-v1.4 already enumerates the missing `.c` files to add (`intor4.c`, `deriv3/4.c`, `gaunt1.c`, `breit1.c`).

**Warning signs:** Link error `undefined reference to <symbol>`; or the vendor wrapper references an FFI binding that does not exist in `oracle_bindings.rs`.

**Phase to address:** Groups 2 (`deriv3/4.c`), 4 (`intor4.c` spin 2e), 5 (`intor4.c` giao/cg 2e), 6 (`gaunt1.c`, `breit1.c`).

### Pitfall 11: The harness's self-comparing "upstream proxy" path gives a green parity artifact that never touched libcint

**What goes wrong:**
The matrix-driven parity loop diffs cintx's `eval_raw` against `eval_legacy_symbol`, but for the new families the legacy wrapper just calls `eval_raw` on the same `RawApiId` — so `raw_vs_upstream` is **cintx-vs-cintx** and can never catch a wrong-but-deterministic kernel (CR-02 from the Phase-21 review, `compare.rs:1271-1316`). The only genuine oracle is the `#[cfg(has_vendor_libcint)]` vendor block / the dedicated `*_parity.rs` tests. If a new family is added to the matrix path but no dedicated vendor test, the parity report is green and vacuous — the report label even overstated it as "vendored upstream compatibility proxy."

**How to avoid:**
- For every new family, the authoritative gate is a dedicated `tests/*_parity.rs` that calls a real `vendor_*` symbol under `CINTX_ORACLE_BUILD_VENDOR=1`. The matrix `raw_vs_upstream` is self-consistency only — label it as such and never let it stand in for vendor parity.
- Confirm each new family has N>0 *vendor* tests that actually execute (Pitfall 12).

**Warning signs:** A family is `oracle_covered=true` with only a matrix-path "pass" and no `vendor_*` reference; report label claims vendor parity for a self-comparing path.

**Phase to address:** Every group's oracle task; carry the CR-02 lesson forward.

### Pitfall 12: Vendor-gated tests silently skip — the double gate (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`) produces "running 0 tests"

**What goes wrong:**
Parity assertions compile out unless BOTH `--features cpu` and `CINTX_ORACLE_BUILD_VENDOR=1` are set (the `has_vendor_libcint`/`has_vendor_pyscf_nr_ecp` cfg gates). Without both, the binary reports `running 0 tests` — a silently vacuous pass (Phase-21 verification §human-verification). A CI lane that runs the standard `--test` gate without the vendor flag will "pass" all v1.4 families having tested nothing.

**How to avoid:**
- Each group's verification must confirm `running N tests` with N>0 under both flags, not just exit-0. The Phase-21 human-verification block is the template (it enumerates exact expected counts per binary).
- A pre-existing failure already lurks in the LIB-gated path: `CINTshells_cart_offset` reports `cintx=8 / vendor=0` (priors). Adding more vendor-gated tests near this surface risks inheriting/obscuring it — re-run the LIB tests explicitly under both flags per group and triage the offset mismatch separately.

**Warning signs:** `running 0 tests`; a green CI lane that never set `CINTX_ORACLE_BUILD_VENDOR=1`; the `CINTshells_cart_offset` 8-vs-0 mismatch resurfacing.

**Phase to address:** Every group's verification step; explicitly enumerate expected vendor-test counts.

---

## Minor Pitfalls

### Pitfall 13: Promoting half-registered `unstable::source::2e` families (`int2e_ipip1`, `int2e_ipvip1`) to stable without re-checking the source-only raw-api map

`int2e_ipip1_sph`/`int2e_ipvip1_sph` already appear in `compare.rs:314-315` as `RawApiId::Symbol(...)` and are registered only under `unstable::source::2e`. Group 2's full-parity claim promotes them to stable 2e Hessian families. The `source_only_raw_api_for_symbol` fallback (`compare.rs:354`) and the `unstable::source::` family-prefix branch (`compare.rs:369`) must be re-routed when the family is promoted, or the symbol resolves through the wrong (source-only) path. **Phase:** Group 2.

### Pitfall 14: Cart/sph σ-family variants are raw spin-free intermediates, not physical observables — do not over-claim oracle coverage

For Groups 4/6, libcint's cart/sph variant of a σ-coupled family emits only the raw spin-free intermediate (the physics lives in the spinor σ-coupling). Registering cart/sph for symbol-coverage parity is fine, but the physical-correctness gate is the spinor output. Treating the cart/sph numbers as a deliverable inflates the oracle surface with non-meaningful values (FEATURES-v1.4 anti-feature). **Phase:** Groups 4, 6.

### Pitfall 15: `ECPscalar_iprinv` coordinate-match selection (WR-06) diverges from vendor integer-index selection for degenerate centers

Carried from Phase 21: `select_iprinv_slots` matches the rinv origin by Euclidean distance (`ecp.rs:612`), not the vendor's integer atom index. Any v1.4 ECP-adjacent derivative family that reuses this selector inherits a byte-identity risk for co-located/degenerate ECP centers. **Phase:** any group reusing the ECP iprinv selector.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Reuse `cart_to_spinor_sf` for σ families | Kernel compiles & dispatches now | Silently wrong spinor output; oracle rejects after kernel written; blocks Groups 4/6 | **Never** — land Gap B2 first |
| Stage GIAO output as real f64 | Reuses overlap staging path | Silent zero/half-answer; passes weak gate (Pitfall 1) | **Never** — set `complex_interleaved` |
| Validate all groups on H2O/STO-3G | No new fixtures | Spinor + gauge-origin paths never exercised; green-but-vacuous coverage | **Never** for Groups 3/4/5/6 |
| Single "order" scalar for G-tensor headroom instead of the `ng[]` tuple | Simple planner change | Under-allocation on deriv4/multi-center + wrong-index elevation on moments | Only for pure single-center bra derivatives (Group 1) |
| Keep `if dst < staging.len()` scatter guards | Avoids touching working code | Silent partial writes at rank 9/27/81; violates no-partial-writes contract | **Never** at high rank — assert size, index unconditionally |
| Rely on the matrix `raw_vs_upstream` path as the oracle | No new test file per family | cintx-vs-cintx self-comparison; green artifact never touched libcint | **Never** — dedicated `vendor_*` test is authoritative |
| Defer the `nroots>=6` Wheeler fallback | Ships low-l families fast | High-l Group 2/4/6 quartets return `UnsupportedApi` instead of parity | Acceptable IF the milestone explicitly scopes out high-l for those families |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Vendored libcint FFI (complex families) | Pass real-sized `f64` buffer to `double complex *out` symbol | Pass `2×`-sized interleaved buffer (the `vendor_int1e_ovlp_spinor` pattern) |
| bindgen allowlist | Add `.c` to `cc::Build` but forget the suppl-header `extern` + allowlist regex (or vice-versa) | Treat (cc file, suppl extern, allowlist regex) as one atomic per-family change |
| `cint_funcs.h` | Assume every symbol is declared there | Many giao/cg/gauge/Gaunt symbols are `.c`-only → supplemental `extern CINTIntegralFunction` |
| ALL_CINT vs ALL_CINT1E legacy wrappers | Generate `_optimizer` siblings for a 1e family (or omit them for a 2e family) | Read the trailing macro line per family; extend the `misc_wrapper_macro` map; ECP gets no wrapper |
| `PTR_COMMON_ORIG` env slot | Assume only `_origj` variants read it | Plain `int1e_r/rr/r2/z` read it too (`drj = rj - PTR_COMMON_ORIG`); plumb Gap A for all of Group 3 |
| `oracle-covered-update` | Flip `oracle_covered=true` on a `skipped` spinor fixture | Refuse to flip on skipped fixtures (enforce the existing doc-comment convention mechanically) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| High-rank staging (×27/×81) blows the OOM-safe chunk plan | OOM / allocation failure at rank 81 not seen at rank 3 | Re-derive chunk-planner limits from `component_rank`; add OOM tests at rank 27/81 | Large basis × 4th-order derivative or hexadecapole |
| G-tensor built at `l+order` for many centers | Memory grows with elevated `l` on multiple centers (deriv4 `{2,2,…}`, grad2 `{2,1,1,1,…}`) | Size `g_size` from the per-center `ng[]` headroom tuple, not a global order | d/f shells with multi-center elevation |
| nroots growth with elevated angular momentum | `UnsupportedApi` at `nroots>5` for high-l quartets | Wheeler `nroots>=6` fallback OR scoped `UnsupportedApi` envelope | High-l Group 2/4/6 quartets |

## Security Mistakes

Not applicable — this is a numerical library milestone with no network/auth/user-input surface. The analogous correctness-integrity concern is the OOM-safe-stop contract (no partial writes), covered under Pitfall 8 and the Performance Traps table.

## "Looks Done But Isn't" Checklist

- [ ] **GIAO/magnetic family:** often missing the imaginary lane — verify `complex_interleaved=true` and that the interleaved-pair buffer contract fires for cart/sph (not just spinor).
- [ ] **σ-operator family:** often routed through scalar `cart_to_spinor_sf` — verify it consumes the 4-block `gc_x/y/z/1` G-tensor via the `c2s_si` transform and was exercised on a kappa-bearing fixture.
- [ ] **Multipole moment:** often validated only at zero gauge origin — verify `PTR_COMMON_ORIG` is read (Gap A) and a non-zero-origin fixture passes; verify headroom is on the **ket** (`ng[1]`), not the bra.
- [ ] **Higher-order derivative:** often uses bra-only `l+1` headroom — verify the per-center `ng[]` tuple (deriv4 raises bra+2 AND ket+2) and the scatter loop asserts buffer size (no silent `if dst < len`).
- [ ] **High-rank tensor (9/27/81):** often component-permuted — verify element-for-element byte-identity (no sort/abs shortcut) and the component ordering copied from the libcint gout index map.
- [ ] **New vendor symbol:** often half-wired — verify (cc `.c` file + suppl `extern` + allowlist regex) all three present, and a dedicated `vendor_*` test (not the self-comparing matrix path).
- [ ] **Vendor parity claim:** often vacuous — verify `running N tests` (N>0) under BOTH `--features cpu` and `CINTX_ORACLE_BUILD_VENDOR=1`, not just exit 0.
- [ ] **`oracle_covered=true`:** often flipped on a skipped spinor fixture — verify the fixture was actually *evaluated* (not `skipped=true`).

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Imaginary read as real (P1) | MEDIUM | Set `complex_interleaved`; widen `assert_flat_buffer_contract` to fire on the flag; re-stage interleaved; re-run vendor parity on non-zero-origin fixture |
| Wrong spinor transform (P2) | HIGH | Land Gap B2 (the `c2s_si` 4-block transform + σ G-tensor) as a foundation phase; keep affected families `UnsupportedApi` until it passes |
| Unread `PTR_COMMON_ORIG` (P3) | LOW | Add Gap A read block + validator (verbatim `PTR_RINV_ORIG` precedent); add gauge-origin fixture |
| H2O-only validation (P4) | LOW–MEDIUM | Add relativistic + gauge-origin fixtures; make `oracle-covered-update` reject skipped-fixture flips |
| Headroom under-size (P5) | MEDIUM | Plumb the `ng[]` headroom tuple per family; assert G-tensor span ≥ max gout index; moment ket-elevation regression |
| nroots>5 overflow (P6) | HIGH (if fallback needed) | Implement Wheeler `nroots>=6` fallback (the pending todo) OR scope high-l out with explicit `UnsupportedApi` envelope |
| Component permute (P7) | LOW–MEDIUM | Re-copy the gout index map from source; keep element-for-element comparison; assert σ block count |
| Silent partial write (P8) | LOW | Replace `if dst < len` guards with a size assertion + unconditional index before the scatter loop |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase / Group | Verification |
|---------|--------------------------|--------------|
| P1 Imaginary-as-real | Complex-output capability phase (before G5) | Cart/sph GIAO fixture fails if staged real-only; interleaved buffer length `2×ncomp×…` |
| P2 Wrong spinor transform | Gap B2 (before G4/G6) | σ family passes spinor parity on kappa-bearing fixture; G-tensor has 4 blocks |
| P3 Unread PTR_COMMON_ORIG | Gap A (before G3/G5) | Non-zero-origin fixture parity passes; `eval_raw` has env[1..3] read block |
| P4 H2O-only validation | Gap A (gauge fixture) + Gap B2 (relativistic fixture) | N>0 evaluated (non-skipped) fixtures per group; no `oracle_covered` on skipped |
| P5 Headroom under-size | G2 (multi-center) + G3 (ket moments) | G-tensor span ≥ max gout index; deriv4 `{2,2,…}` and `int1e_rrrr` ket-elevation regressions |
| P6 nroots>5 overflow | G2 (surfaces) + Wheeler fallback prerequisite OR scoped envelope | High-l quartet returns typed error or parity (per scope decision) |
| P7 Component permute | G2 (×9/27/81) + G3 (`rr/rrr`) + G4 (σ 12-comp) | Element-for-element byte-identity; gout index map matches source |
| P8 Silent partial write | Cleanup at start of G2 | Scatter loop asserts `staging.len() >= ncomp×…`; no per-element drop |
| P9 FFI signature mismatch | Every group's oracle task | Macro-shape map + interleaved complex buffer; ECP no-wrapper |
| P10 Missing suppl header/allowlist | G2/G4/G5/G6 oracle tasks | No undefined-reference link errors; FFI binding exists in `oracle_bindings.rs` |
| P11 Self-comparing oracle | Every group | Dedicated `vendor_*` test exists and executes; matrix path labeled self-consistency |
| P12 Vacuous vendor skip | Every group's verification | `running N tests` (N>0) under both flags; `CINTshells_cart_offset` 8-vs-0 triaged |
| P13 Unstable→stable promotion | G2 | Promoted `int2e_ipip1/ipvip1` route through stable (not source-only) raw-api map |
| P14 Cart/sph σ over-claim | G4/G6 | Oracle expectations gated to what libcint emits; spinor is the correctness gate |
| P15 ECP coord-match selection | Any ECP-deriv reuse | Unit test with two close-but-distinct ECP centers pins selection |

## Sources

- libcint 6.1.3 vendored source (HIGH — read directly):
  - `src/cart2sph.c` — `c2s_sf_1e` (single scalar block) vs `c2s_si_1e` (4-block `gc_x/gc_y/gc_z/gc_1`) at lines 4869/4947; `c2s_dset0`/`c2s_zset0` imaginary-zeroing at 4707/4737.
  - `src/autocode/grad1.c` `{1,0,0,0,1,1,1,3}`, `hess.c` `{2,0,0,0,2,1,1,9}`, `deriv3.c` `{3,0,0,0,3,1,0,27}`, `deriv4.c` `{2,2,0,0,4,1,0,81}` — `ng[]` headroom ground truth.
  - `src/autocode/intor1.c` — `int1e_r` `{0,1,0,0,…,3}` / `int1e_rrrr` `{0,4,0,0,…,81}` ket-side moment headroom; `CINTgout1e_int1e_r`/`int1e_r2` `drj = rj - env[PTR_COMMON_ORIG]`; `CINTgout1e_int1e_igovlp` imaginary cross-product `gout = -c[1]*s[2]+c[2]*s[1]` + `c2s_zset0` routing.
  - `src/autocode/grad2.c` `{2,1,1,1,5,4,4,3}` multi-center elevation; `breit1.c` — `int2e_gauge_r1_ssp1ssp2_spinor` routes `CINT2e_spinor_drv(…, &c2s_si_2e1i, &c2s_si_2e2i)` (line 211).
  - `src/g1e.h` — `G1E_RCJ`/`G1E_R0I` macros (`CINTx1j_1e(…, drj, …)`).
  - `src/misc.h:35,63` — `ALL_CINT` (with `_optimizer`) vs `ALL_CINT1E` (no opt) macro split.
- cintx live tree (HIGH — read directly):
  - `crates/cintx-oracle/src/compare.rs` — `assert_flat_buffer_contract` (spinor-only complex check, line 278), self-comparing matrix path, `raw_api_for_symbol`, `skipped` flag semantics, `CINTshells_cart_offset` surface.
  - `crates/cintx-oracle/build.rs` — `cc::Build` `.file()` chain (181-253), supplemental `extern CINTIntegralFunction` header (265-345), bindgen allowlist regex (358).
  - `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_int1e_ovlp_spinor` interleaved `f64`→`double complex` pattern (1316), real-`out` cart/sph wrappers.
  - `crates/cintx-compat/src/legacy.rs` — `all_cint1e_wrappers!`/`all_cint_wrappers!` macros, `misc_wrapper_macro` derivation map (378), ECP no-wrapper note (285).
- cintx planning (HIGH): `.planning/PROJECT.md` (v1.4 scope, prior pitfalls); `21-VERIFICATION.md` (vendor double-gate, WR-03 silent scatter, human-verification counts); `21-REVIEW.md` (CR-01 buffer-sizing, CR-02 self-comparing oracle, WR-03/WR-06); `STACK-v1.4.md`/`FEATURES-v1.4.md`/`ARCHITECTURE-v1.4.md` (Gaps A/B1/B2, group ordering, complex/imaginary capability).
- cintx priors (HIGH — milestone context): env global-slot collisions, kinetic D_j² ±2 j-level HRR headroom, F-order transpose, `rys-nroots-ge6-wheeler-fallback` pending todo, c2s_zset0 imaginary zeroing, missing `c2s_si` transform, spinor-gradient `UnsupportedApi` (R5/D-03).

---
*Pitfalls research for: v1.4 full libcint 6.1.3 family parity (6 family groups)*
*Researched: 2026-05-27*
