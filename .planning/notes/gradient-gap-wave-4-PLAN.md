# Wave 4 Execution Plan — Tier 4 + Tier 5 σ and remaining families

**Created:** 2026-08-22
**Parent:** `.planning/notes/gradient-family-gap-closure-PLAN.md` §4 "Wave 4"
**Milestone:** v1.4 — Full libcint 6.1.3 Family Parity
**Baseline commit:** `3829c35` (`feat(cubecl,oracle): close gradient family gaps …`)
**Audience:** an execution agent that follows instructions literally and does NOT infer.

---

## 0. STATUS — Wave 4 is COMPLETE (2026-08-22)

All 17 Wave-4 symbols are `oracle_covered = true`, plus two families this wave
inherited or uncovered. Every gate below is a non-`#[ignore]`d vendor test at
`atol = 1e-12` on a `d`-shell `nctr = 2` fixture.

| Task | Symbols | Outcome |
|---|---|---|
| **W4-00** | — | 2e σ test tightened `2e-11 → 1e-12`, fixture raised `p → d` (`nctr=2`), per-family reporting. The 1e σ file was already at `1e-12`. |
| **W4-01/02** | 6 × 1e σ | Were already green; re-verified on the tightened gate. |
| **W4-03** | `int2e_ipspsp1`, `ip1spsp2`, `ipspsp1spsp2` (+ `int2e_spsp2`) | **Fixed (RC-2)** and flipped. |
| **W4-04** | `int2e_ipsrsr1`, `ip1srsr2`, `ipsrsr1srsr2` | **Fixed (RC-1)** and flipped. |
| **W4-05** | `int3c2e_ipspsp1` | **Implemented**: new `cart_to_spinor_si_3c2e1` transform + launcher. |
| **W4-06** | `int2e_ip1v_r1`, `ip1v_rc1`, `ipvg1_xp1`, `ipvg2_xp1` | **Implemented**: cart + sph + spinor (12 rows). |
| **Collateral** | `int2e_spinor` | **Shipped-API defect found and fixed** — see §0.2. |

### 0.1 The two Wave-4 root causes, as diagnosed and fixed

**RC-2 — `cart_to_spinor_sf_2e1` was missing the KET→BRA transpose.**
The leaf `cart_to_spinor_sf_2d` reads BRA-major and does not own the transpose
(unlike `cart_to_spinor_si_2d`, which does). `sf_2e1` forwarded its KET-major input
straight through, so it returned the `i↔j` transpose of the correct block. This was
invisible while `sf_2e1` was only ever paired with `sf_2e2` and surfaced only through
the `(sf_2e1, si_2e2)` pairing — the one pairing no gate covered. Fixing it turned
`int2e_spsp2`, `int2e_ip1spsp2` and `int2e_ip1srsr2` green in one step.

**RC-1 — the `bra_derivative_from_shifted` shortcut is invalid for σ·r.**
It synthesized `∇_i` by evaluating the base σ family at `li ± 1`. That identity holds
for σ·p but not for σ·r, whose operator is itself a bra-index raise. Kinds 0/2/3/5 now
use explicit cascades transcribed from `grad2.c`, and the shortcut has been deleted.
Two new pieces support this:
  * `rel2e_leg_masks` — the generic `NLEG`-leg → `3^NLEG`-term `s[]` table. Verified
    term-by-term against the vendored 27-term and 243-term tables, so the 243-line
    autocoded table is generated rather than transcribed.
  * `gout_rel2e_rank243` — the 32-block machine for the 2-sided σ⊗σ derivatives.

Kinds 0 and 2 (`ipspsp1`, `ipspsp1spsp2`) were green BEFORE this change and were
deliberately routed through the new machinery as its cross-check. They stayed green.

### 0.2 Collateral: `int2e_spinor` was wrong for every `l > 0` shell

`cart_to_spinor_sf_4d` had the identical missing transpose. Its documented input
contract (`kernels/two_electron.rs:4732`) is i-fastest, but it forwarded each `(k,l)`
slice straight to the BRA-major `cart_to_spinor_sf_2d`.

`int2e_spinor` — `Stability::Stable`, `oracle_covered = true`, shipped — therefore
returned the `i↔j` transpose of the correct spinor block: exact for an all-`s` quartet
(`nci == ncj == 1`) and wrong by ~3e-3 for anything higher. The existing coverage only
exercised `s` shells, so nothing caught it.

Fixed, and gated by `crates/cintx-oracle/tests/two_electron_spinor_orientation.rs`,
which sweeps non-square quartets where the transpose cannot cancel. Residuals went
`3.1e-3 → 1.6e-17`. `int2e_breit_*` and the F12 spinor families share `sf_4d` and are
covered by the same fix.

### 0.3 Corrections to the parent plan, confirmed against the vendored C

* **Only `int2e_ip1v_rc1` reads `PTR_COMMON_ORIG`.** The parent plan's W4-06 note
  ("reuse the Phase-22 `PTR_COMMON_ORIG` plumbing") is wrong for the other three:
  `ip1v_r1` uses a plain stride shift (`G2E_R_J`) and both `xp1` families raise about a
  BASIS CENTRE (`G2E_R0I` / `G2E_R0K`). The launcher now reads the origin only for
  `ip1v_rc1` (`Gauge2eKind::uses_common_origin`), and
  `only_rc1_depends_on_the_gauge_origin` asserts that moving the gauge origin changes
  `rc1` and leaves the other three bit-identical.
* **The W4-06 families are not σ and not spinor-only.** `ng[5] == ng[6] == 1`; their
  spinor drivers use `c2s_sf_2e1 + c2s_sf_2e2`. They ship cart + sph + spinor.
* **`int2e_ipvg2_xp1_spinor` carries a −1 phase.** Alone among the four it uses the
  imaginary-ket pair `c2s_sf_2e1i + c2s_sf_2e2i` (`intor2.c:1004`) — `i * i = -1`
  against the plain pair, the same convention `kernels/unstable/breit.rs:1944`
  documents.
* **W4-05 is a new transform, not a dispatch arm** (as this plan predicted). Its gout
  is byte-for-byte `int2e_ipspsp1`'s with a phantom `l`-shell, so it reuses
  `gout_ip_sigma(0, …)`; the new code is `cart_to_spinor_si_3c2e1`.
* **libcint 6.1.3 declares these symbols in `cint_funcs.h`.** The parent plan's §1.1
  premise (33 symbols absent from the header) does not hold for this vendored copy —
  bindgen sees all of them, and `crates/cintx-oracle/build.rs` already allowlisted
  them, so W0-03 needed no work for W4-05/W4-06 beyond the `vendor_ffi` wrappers.

### 0.4 What this wave did NOT do

* The other 52 entries in the parity gate's `unsupported_libcint_families` list are
  Wave 5 / Phase 30–31 scope and were deliberately left alone (parent plan R-06).
* `Gap B`'s 11 unverified spinor rows (parent plan §1.3) remain Wave 5 work. Note that
  the `sf_4d` fix in §0.2 is likely to move several of them.

---

## 0b. ORIGINAL PLAN (retained for provenance)

Everything from here down is the plan as written BEFORE execution, kept so the
diagnosis in §0 can be checked against what was predicted. Where it and §0 disagree,
**§0 is the current state**.

## 0c. PREMISE CHECK — Wave 4 is unblocked, and it is already ~half done

The parent plan says *"Do not start Wave 4 before Phase 29 is closed."*

**Phase 29 is closed** (`.planning/ROADMAP.md:33`, completed 2026-06-01, 6/6). Its own
parity suite is green today:

```
$ CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test rel_2e_sigma_parity
test result: ok. 18 passed; 0 failed; 0 ignored
```

Wave 4 is therefore unblocked. It is also **not** a greenfield wave: commit `3829c35`
landed manifest rows, dispatch arms, kernels and oracle wiring for **12 of the 17
symbols**. Six are proven; six are wired but **numerically wrong**; five were never
started.

**Do not re-implement the 12. Read §1 before writing any code.**

---

## 1. VERIFIED STATE — measured 2026-08-22, not inferred

Every cell below was produced by running the code, not by reading it.

### 1.1 W4-01 / W4-02 — 1e σ gradients (6 symbols): **DONE**

Manifest: 6 spinor rows, `oracle_covered=true`
(`api_manifest.csv:419-424`). Dispatch: `kernels/sigma_1e.rs:73-78` (kind ids 13–18),
`kernels/deriv34.rs:1592-1610` (FamilySpec), rinv-origin arm at
`kernels/one_electron.rs:10669`. Vendor wrappers: `vendor_ffi.rs:76-81`.

Measured residual vs vendored libcint, d-shell (`ANG_OF=2`) `nctr=2` fixture:

| Symbol | rank | max_abs | ≤1e-12? |
|---|---:|---:|---|
| `int1e_ipspnucsp_spinor` | 3 | 4.121e-13 | yes |
| `int1e_ipsprinvsp_spinor` | 3 | 1.421e-14 | yes |
| `int1e_ipipspnucsp_spinor` | 9 | 8.882e-15 | yes |
| `int1e_ipipsprinvsp_spinor` | 9 | 1.332e-15 | yes |
| `int1e_ipspnucspip_spinor` | 9 | 2.043e-14 | yes |
| `int1e_ipsprinvspip_spinor` | 9 | 2.776e-15 | yes |

`crates/cintx-oracle/tests/gradient_gap_tier4_1e_sigma.rs` passes, not `#[ignore]`d.
**Residual work is hygiene only — see W4-00 below.**

### 1.2 W4-03 / W4-04 — 2e σ gradients (6 symbols): **3 green, 3 red**

All six have manifest rows (`api_manifest.csv:425-430`), dispatch arms
(`two_electron.rs:3019-3024`) and vendor wrappers (`vendor_ffi.rs:149-154`).
All six carry `oracle_covered=false`. Measured per-family (p-shell `nctr=2` quartet):

| Symbol | `gout_ip_sigma` kind | (e1, e2) transform | max_abs | verdict |
|---|---:|---|---:|---|
| `int2e_ipspsp1_spinor` | 0 | (Si, Sf) | 2.168e-17 | **GREEN** |
| `int2e_ipspsp1spsp2_spinor` | 2 | (Si, Si) | 9.714e-17 | **GREEN** |
| `int2e_ip1spsp2_spinor` | 1 | (Sf, Si) | 2.449e-3 | RED |
| `int2e_ip1srsr2_spinor` | 4 | (Sf, Si) | 6.815e-4 | RED |
| `int2e_ipsrsr1_spinor` | 3 | (Si, Sf) | 9.739e-4 | RED |
| `int2e_ipsrsr1srsr2_spinor` | 5 | (Si, Si) | 2.796e-3 | RED |

> Reproduce: the checked-in test `gradient_gap_tier4_2e_sigma.rs` asserts inside a loop
> and therefore aborts on the first failure, hiding the other five. W4-00 fixes that.

### 1.3 Collateral discovery — `int2e_spsp2_spinor` is broken and uncovered

`int2e_spsp2_spinor` is a **Phase 29 REL-03 family**, not a Wave 4 family. It carries
`oracle_covered=false` (`api_manifest.csv` last row), it is **absent from
`rel_2e_sigma_parity.rs`'s family list**, and it fails:

```
int2e_spsp2_spinor: max_abs = 2.366e-3     (vs ip1spsp2's 2.449e-3 — same defect)
```

Phase 29 shipped 16 of its 17 2e σ rows green and left this one out of its own suite.
**Wave 4 inherits it**, because `int2e_ip1spsp2 = int2e_spsp2 + one ∇ on e1` and cannot
be made green while its base is wrong. It is in scope for W4-03.

### 1.4 W4-05 / W4-06 — 5 symbols: **NOT STARTED**

No manifest rows, no dispatch arms, no `vendor_ffi` wrappers. Only the bindgen
allowlist regex (`crates/cintx-oracle/build.rs:405`) already names them, and
`intor2.c` / `int3c2e.c` are already in the `cc::Build` file list
(`build.rs:242`, `:255`) — so **W0-03 is done for these five; do not redo it.**

All five appear in today's parity-gate `unsupported_libcint_families` list
(57 entries total; the other 52 belong to Wave 5 and Phases 30–31 — **do not touch them**):

```
$ cd xtask && cargo run -q -- manifest-audit
$ python3 -c "import json;d=json.load(open('/tmp/cintx_artifacts/cintx_phase_04_manifest_audit.json'));print(d['libcint_export_parity']['unsupported_count'])"
57
```

---

## 2. ROOT CAUSE — the six red families have exactly two causes, not six

This section is the load-bearing part of the plan. Both causes were isolated by
correlating the pass/fail split against the code structure. **Read it before touching
`f12.rs`.**

### RC-1 — the `bra_derivative_from_shifted` shortcut is invalid for σ·**r**

`f12.rs:2780-2792` computes kinds 0, 2, 3, 5 **without transcribing the vendored
cascade**. It evaluates the *base* σ family at `li+1` and `li−1` and synthesizes `∇_i`
from the pair:

```rust
let base = |l: usize| match kind {
    0 => gout_spsp1(g, shape, l, lj, lk, ll, ai, aj),
    2 => gout_spsp1spsp2(...),
    3 => gout_srsr1(g, shape, l, lj, lk, ll),
    5 => gout_srsr1srsr2(...),
    ...
};
let plus  = base(li + 1);
let minus = li.checked_sub(1).map(base);
bra_derivative_from_shifted(li, lj, lk, ll, ncomp, ai, &plus, minus.as_deref())
```

That identity holds for σ·**p** (kinds 0, 2 — both GREEN) because the operator carries
no explicit `r_i`. It **fails for σ·r** (kinds 3, 5 — both RED) because the σ·r operator
*is* a bra-index raise, so shifting `li` perturbs the operator as well as the basis
function. The vendored code makes the ordering explicit — `D_I` is applied **before**
the `R_I` fold:

```c
/* CINTgout2e_int2e_ipsrsr1, grad2.c:861 */
G2E_R_J(g1, g0, i_l+2, j_l+0, k_l, l_l);
G2E_D_I(g2, g0, i_l+1, j_l,   k_l, l_l);
G2E_D_I(g3, g1, i_l+1, j_l,   k_l, l_l);
G2E_R_I(g4, g0, i_l+0, j_l,   k_l, l_l);   /* R_I applied to the ALREADY-differentiated */
G2E_R_I(g5, g1, i_l+0, j_l,   k_l, l_l);   /* blocks — not reconstructible from base(li±1) */
G2E_R_I(g6, g2, i_l+0, j_l,   k_l, l_l);
G2E_R_I(g7, g3, i_l+0, j_l,   k_l, l_l);
```

**Fix:** kinds 3 and 5 must get explicit `Rel2eStep` cascades exactly like the kind 1|4
branch at `f12.rs:2710-2778`. `Rel2eOp::Ri/Rj/Rk/Rl` already implement the plain stride
shift that `G2E_R_I/J/K/L` means (`f12.rs:2413-2424`) — the primitives are correct; only
the composition is missing. `ipsrsr1srsr2` needs a **32-block** cascade
(`grad2.c:1089`, g1…g31); budget for it.

### RC-2 — the `(c2s_sf_2e1, c2s_si_2e2)` transform pairing was never proven

Exactly three families in the whole codebase pair a **spin-free e1** transform with a
**σ e2** transform: `spsp2`, `ip1spsp2`, `ip1srsr2` (`two_electron.rs:3015`, `:3020`,
`:3023`). **All three are RED. Every other pairing is green.**

- `(Si, Sf)` — proven by Phase 29's D-03 micro-test (`si_2e_transform_parity.rs`).
- `(Si, Si)` — proven by `spsp1spsp2` / `srsr1srsr2` in `rel_2e_sigma_parity.rs`.
- `(SiI, SiI)` — proven by the four `ssp/sps` families.
- `(Sf, Si)` — **never gated by anything.**

The kind-1 cascade at `f12.rs:2712-2760` structurally matches vendored
`CINTgout2e_int2e_ip1spsp2` (`grad2.c:297`: `D_L`, `D_K`, `D_K∘g1`, then `D_I` of
g0…g3) — so the cascade is not the suspect. The base `int2e_spsp2`
(`gout_spsp2`, `f12.rs:2535`, a different function with a 3-block cascade) fails at the
**same magnitude**. The single structure shared by all three failures is the `(Sf, Si)`
pairing.

**Fix order:** prove `(Sf, Si)` on the *base* family `int2e_spsp2` first, exactly as
Phase 29 proved `(Si, Sf)` on `int2e_spsp1` before wiring anything onto it (29-04, the
D-03 gate). `ip1spsp2` and `ip1srsr2` then follow for free — or, if they do not, the
remaining delta is provably cascade-local and cheap to chase.

### Consequence for scheduling

| Symbol | RC-1 | RC-2 |
|---|:-:|:-:|
| `int2e_spsp2` (Phase-29 leftover) | | ✗ |
| `int2e_ip1spsp2` | | ✗ |
| `int2e_ip1srsr2` | | ✗ |
| `int2e_ipsrsr1` | ✗ | |
| `int2e_ipsrsr1srsr2` | ✗ | |

RC-1 and RC-2 are independent and can be worked in parallel by two agents.

### Downstream impact

`pyscf/grad/dhf.py` needs `int1e_ipspnucsp`, `int1e_ipsprinvsp`, `int2e_ipspsp1`,
`int2e_ip1spsp2`, `int2e_ipspsp1spsp2`. Four of those five are green today.
**RC-2 alone gates `pyscf.grad.dhf`.** W4-04 (`srsr`), W4-05 and W4-06 are
PARITY-01-only — no PySCF consumer is waiting on them.

---

## 3. TASKS

Standing rules RULE 1–7 of the parent plan apply unchanged. RULE 4 (`atol=1e-12`,
never claim unproven coverage) is the one this wave has already bent — see W4-00.

### W4-00 — Test hygiene (blocking, ~1h). Do this first.

Three defects in the checked-in Wave-4 tests, all of which weaken the gate:

1. **`ATOL = 2e-11` is looser than RULE 4's `1e-12`** —
   `gradient_gap_tier4_1e_sigma.rs:10` and `gradient_gap_tier4_2e_sigma.rs:10`.
   Tighten both to `1e-12`. §1.1 shows all six 1e families already clear it, so this
   costs nothing and closes a real hole.
2. **The 2e fixture is a p-shell quartet (`ANG_OF = 1`), not the d-shell `nctr>1`
   fixture W0-04 mandates** (`gradient_gap_tier4_2e_sigma.rs:35`). The 1e fixture is
   already correct (`ANG_OF = 2`). Raise the 2e fixture to `ANG_OF = 2` and re-measure
   §1.2 — **expect the RED residuals to move, and check whether either GREEN family
   regresses.** A d-shell quartet at `ng={2,1,0,0,…}` gives
   `nroots = (4+3+2+2)/2 + 1 = 6 > MAX_DEVICE_NROOTS`, so this also exercises the
   host-route path that the p-shell fixture never reaches.
3. **Both tests assert inside a `for` loop**, so the first failure hides every later
   family. Collect `(symbol, max_abs)` for all families, then assert once with the full
   table in the message. This is what made §1.2 expensive to produce; do not leave the
   next agent to redo it.

**Gate:** both test files at `ATOL = 1e-12`, d-shell `nctr=2` fixtures, per-family
reporting; the six 1e families still green.

### W4-03 — `int2e_ip1spsp2` + `int2e_spsp2` (RC-2). **Highest priority — unblocks `pyscf.grad.dhf`.**

`int2e_ipspsp1` and `int2e_ipspsp1spsp2` need no code — flip them to
`oracle_covered=true` once W4-00's tightened, d-shell test is green (RULE 7: edit
`api_manifest.csv`, regenerate `api_manifest.rs`, do not hand-edit).

For the RC-2 pair:

1. **Write the `(Sf, Si)` micro-test first, on the base family.** Model it on
   `crates/cintx-oracle/tests/si_2e_transform_parity.rs` (Phase 29's D-03 gate), but for
   `int2e_spsp2_spinor` = `c2s_sf_2e1 + c2s_si_2e2`. It must be byte-identical at
   `1e-12` **before** any gradient family is wired onto the pairing. This is the
   blocking gate for W4-03; treat it exactly as 29-04 was treated.
2. Fix whatever it surfaces. Read `libcint-master/src/cart2sph.c` `c2s_si_2e2` against
   the cintx `E2Transform::Si` implementation; the σ e2 block ordering (`ox/oy/oz/o1`)
   and the imaginary-unit placement are the two things Phase 30-01d showed can hide a
   uniform sub-percent residual.
3. Re-measure `int2e_ip1spsp2` and `int2e_ip1srsr2`. If either still fails, the
   remaining delta is cascade-local: diff cintx's kind-1/kind-4 `Rel2eStep` table
   (`f12.rs:2712-2760`) term-by-term against `CINTgout2e_int2e_ip1spsp2`
   (`grad2.c:297`) and `CINTgout2e_int2e_ip1srsr2` (`grad2.c:975`).
4. Flip `int2e_spsp2_spinor`, `int2e_ip1spsp2_spinor`, `int2e_ip1srsr2_spinor` to
   `oracle_covered=true`, and **add `int2e_spsp2_spinor` to `rel_2e_sigma_parity.rs`'s
   family list** so the Phase-29 suite covers all 17 of its rows.

**Gate:** `int2e_spsp2`, `int2e_ipspsp1`, `int2e_ip1spsp2`, `int2e_ipspsp1spsp2`,
`int2e_ip1srsr2` green at `1e-12` on the d-shell fixture; `pyscf/grad/dhf.py` symbol
set satisfiable.

### W4-04 — `int2e_ipsrsr1`, `int2e_ipsrsr1srsr2` (RC-1)

Replace the `bra_derivative_from_shifted` shortcut for kinds 3 and 5 with explicit
`Rel2eStep` cascades transcribed from the vendored C. **Do not attempt to repair the
shortcut** — it is structurally wrong for σ·r, not off by a factor.

| Kind | Symbol | Vendored gout | Blocks |
|---:|---|---|---:|
| 3 | `int2e_ipsrsr1` | `grad2.c:861` | 8 (g0…g7) |
| 5 | `int2e_ipsrsr1srsr2` | `grad2.c:1089` | 32 (g0…g31) |

Both cascades are reproduced in §2/RC-1 and can be read directly with:
```bash
awk '/void CINTgout2e_int2e_ipsrsr1\(/,/^double s\[/' libcint-master/src/autocode/grad2.c
awk '/void CINTgout2e_int2e_ipsrsr1srsr2\(/,/^double s\[/' libcint-master/src/autocode/grad2.c
```
Keep kinds 0 and 2 on the shortcut — they are green and the shortcut is valid for σ·p.
Note that `gout_srsr1` (`f12.rs:1950`) and `gout_srsr1srsr2` (`f12.rs:2283`) stay as-is;
they are the *non-derivative* families and both are green.

**Gate:** both green at `1e-12`, d-shell fixture; `oracle_covered=true`.

### W4-05 — `int3c2e_ipspsp1` (rank 3, σ on e1)

Not started. Vendored facts (verified):

```
ng[] = {2, 1, 0, 0, 3, 4, 1, 3}          int3c2e.c:751
gout: CINTgout2e_int3c2e_ipspsp1         int3c2e.c:668
cart   → CINT3c2e_drv(..., &c2s_cart_3c2e1, 0)      :754
sph    → CINT3c2e_drv(..., &c2s_sph_3c2e1,  0)      :762
spinor → CINT3c2e_spinor_drv(..., &c2s_si_3c2e1, 0) :770
cascade: D_J(g1,g0,i+2,j+0,k,0); D_I(g2,g0,i+1,..); D_I(g3,g1,i+1,..);
         D_I(g4..g7, g0..g3, i+0, ..)
```

Two things the parent plan does not say:

- **cintx has no `c2s_si_3c2e1` equivalent.** `center_3c2e.rs` imports only
  `cart_to_spinor_sf_3c2e` and `cart_to_spinor_sf_derivative_3c2e`
  (`center_3c2e.rs:27`). A σ 3-center spinor transform is a **new deliverable** of the
  same class as Phase 29's `si_2e1`. Size this as a plan of its own, not a dispatch arm.
- **Declare it spinor-only.** Every σ family in the manifest is spinor-only
  (`int1e_sp`, `int1e_spsp`, `int2e_spsp1`, `int2e_srsr1`, … — verified across the
  whole CSV). Follow the precedent; do not add cart/sph rows for a σ family.

`nroots` at d-shells = `(4+3+2+0)/2 + 1 = 5` (at the cap); at f-shells = 7. **Host-route**
via the `deriv34.rs` / `gout_*` host pattern that the 2e σ families already use.

**Gate:** `int3c2e_ipspsp1_spinor` green at `1e-12`; new si_3c2e1 transform has its own
micro-test, per the D-03 precedent.

### W4-06 — `int2e_ip1v_r1`, `ip1v_rc1`, `ipvg1_xp1`, `ipvg2_xp1` (rank 9)

Not started. **The parent plan is wrong about this group in two ways — correct it.**

> Parent plan W4-06: *"2e Rys + gauge origin — reuse the Phase-22 `PTR_COMMON_ORIG`
> plumbing"*

**Correction 1 — these are not σ families and they are not spinor-only.** All four use
`c2s_cart_2e1` / `c2s_sph_2e1` and, for spinor, `c2s_sf_2e1 + c2s_sf_2e2` — **spin-free**
(`intor2.c:804`, `:825`, `:846`). `ng[5] = ng[6] = 1`. They are ordinary scalar 2e Rys
families in **cart + sph + spinor**, closest sibling `int2e_ip1`.

**Correction 2 — only ONE of the four uses the common origin.**

| Symbol | `ng[]` | gout | cascade head | common origin? | cf |
|---|---|---|---|---|---|
| `int2e_ip1v_r1` | `{1,2,0,0,2,1,1,9}` | `intor2.c:610` | `G2E_R_J` (plain stride shift) | **no** | — |
| `int2e_ip1v_rc1` | `{1,2,0,0,2,1,1,9}` | `intor2.c:522` | `G2E_RCJ` (`drj = rj − R_C`) | **yes** | — |
| `int2e_ipvg1_xp1` | `{2,1,0,0,3,1,1,9}` | `intor2.c:694` | `G2E_R0I` (`envs->ri`) | **no** | ×0.5 |
| `int2e_ipvg2_xp1` | `{1,1,1,0,3,1,1,9}` | `intor2.c:851` | `G2E_R0K` (`envs->rk`) | **no** | ×0.5 |

`G2E_R_J` is `f = g + envs->g_stride_j` (`g2e.h:107`) — a stride shift, no origin.
`G2E_R0I/R0K` shift about the **basis centers** `ri`/`rk` (`g2e.h:93`, `:95`), not the
gauge origin. Only `G2E_RCJ` (`g2e.h:99`) reads `PTR_COMMON_ORIG`. So the Phase-22
plumbing is needed for `ip1v_rc1` **only**; wiring it into the other three would be a
silent wrong answer.

The two `xp1` families additionally need a **cross-product fold** with coefficients
`c[] = ri − rj` (`intor2.c:709-713`) — there is no `xp1` machinery anywhere in
`crates/cintx-cubecl/src/kernels/` today. `Giao2eKind` (`two_electron.rs:2314`) covers
`G1/Ig1/Gg1/G1g2` and is the right neighbour to extend, but the cross-product fold is
new code. Transcribe the 9-component `gout[n*9+…]` expressions verbatim
(`intor2.c:759-767`); do not re-derive them.

**All four exceed the device Rys cap at d-shells** — `(3+4+2+2)/2+1 = 6`,
`(4+3+2+2)/2+1 = 6`, `(3+3+3+2)/2+1 = 6` — so **host-route from the start** (`deriv34.rs`
pattern). Do not build a device kernel and discover this at the d-shell gate.

**Gate:** four families × cart + sph + spinor green at `1e-12`; twelve new manifest rows.

---

## 4. WAVE 4 EXIT GATE

1. All 17 Wave-4 symbols `oracle_covered=true`, plus the inherited
   `int2e_spsp2_spinor` (18 rows).
2. Every `vendor_*` test non-`#[ignore]`d, at `ATOL = 1e-12`, on a d-shell `nctr=2`
   fixture.
3. `cd xtask && cargo run -q -- manifest-audit --check-lock` reports
   `status: ok` with `uncovered_count` unchanged-or-lower and
   `unsupported_count` reduced from **57 → 52** (the five W4-05/W4-06 symbols leave the
   list; the remaining 52 are Wave 5 / Phase 30–31 scope — **do not chase them here**,
   parent plan R-06).
4. `pyscf/grad/dhf.py` symbol requirements satisfiable (achieved at W4-03, well before
   the full wave closes — report it to the `pyscf_rs` handoff note as soon as it lands).
5. Phase 29's `rel_2e_sigma_parity.rs` still green **and** now covers
   `int2e_spsp2_spinor`.

---

## 5. RISKS SPECIFIC TO THIS WAVE

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| W4-R1 | W4-00's d-shell fixture regresses `ipspsp1`/`ipspsp1spsp2`, which are green only on the p-shell fixture | **MAJOR** | Measure before touching RC-1/RC-2 — the d-shell quartet crosses `MAX_DEVICE_NROOTS` and exercises a code path the current fixture never reaches |
| W4-R2 | RC-2 is a transform bug, and Phase 30-01d is the precedent for a σ-transform residual eating multiple days | **MAJOR** | Gate on the base-family micro-test (`int2e_spsp2`) before wiring gradients; keep `oracle_covered=false` rather than over-claiming (RULE 4) |
| W4-R3 | The 32-block `ipsrsr1srsr2` cascade is transcribed with an index slip that only shows at high `l` | MAJOR | Transcribe mechanically from `grad2.c:1089`; add an f-shell case to that family's test |
| W4-R4 | W4-05's `si_3c2e1` transform is scoped as "a dispatch arm" and blows up | MAJOR | It is a new transform, not an arm — plan it as its own unit with its own micro-test, per §3/W4-05 |
| W4-R5 | `PTR_COMMON_ORIG` gets wired into all four W4-06 families | **MAJOR** | §3/W4-06 Correction 2 — only `ip1v_rc1`. A wrongly-applied origin is a silent wrong number, not an error |
| W4-R6 | 12 new W4-06 manifest rows shift `OperatorId` positional indices | MAJOR | Parent-plan R-04: append rows; re-run `ecp_operator_ids_match_constants`; symbol-name dispatch only (RULE 5) |

---

## 6. ORDER

```
W4-00  (hygiene, blocking, ~1h)
   ├── W4-03  (RC-2)  ← highest value: unblocks pyscf.grad.dhf
   └── W4-04  (RC-1)  ← independent of W4-03; can run in parallel
W4-05  (new si_3c2e1 transform + family)   — parity only
W4-06  (4 families × 3 forms, host-routed) — parity only
```

W4-03 and W4-04 are mutually independent (different `gout_ip_sigma` branches, different
root causes) and are the only tasks with a downstream consumer. W4-05 and W4-06 are
PARITY-01 work and may be deferred behind Wave 5's spinor verifications without blocking
anything outside cintx.

---

## 7. REPRODUCTION COMMANDS

```bash
cd /home/user/Documents/workspace/cintx

# §1.1 / §1.2 residuals
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test gradient_gap_tier4_1e_sigma --test gradient_gap_tier4_2e_sigma

# Phase 29 regression guard (must stay 18/18)
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test rel_2e_sigma_parity

# Parity-gate state
cd xtask && cargo run -q -- manifest-audit --check-lock
# report: /tmp/cintx_artifacts/cintx_phase_04_manifest_audit.json
#   .libcint_export_parity.unsupported_count      -> 57 today, 52 at wave exit
#   .oracle_coverage.uncovered_count              -> reads the compiled manifest lock

# Vendored transcription targets
awk '/void CINTgout2e_int2e_ipsrsr1\(/,/^double s\[/'       libcint-master/src/autocode/grad2.c
awk '/void CINTgout2e_int2e_ipsrsr1srsr2\(/,/^double s\[/'  libcint-master/src/autocode/grad2.c
awk '/void CINTgout2e_int2e_ip1spsp2\(/,/^double s\[/'      libcint-master/src/autocode/grad2.c
awk '/void CINTgout2e_int2e_spsp2\(/,/^double s\[/'         libcint-master/src/autocode/dkb.c
sed -n '522,700p'  libcint-master/src/autocode/intor2.c    # ip1v_rc1 + ip1v_r1
sed -n '694,850p'  libcint-master/src/autocode/intor2.c    # ipvg1_xp1
sed -n '851,1010p' libcint-master/src/autocode/intor2.c    # ipvg2_xp1
sed -n '668,780p'  libcint-master/src/autocode/int3c2e.c   # int3c2e_ipspsp1
```
