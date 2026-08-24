# Wave 5 Execution Plan — Gap B spinor rows, Tier 6, and the parity gate

**Created:** 2026-08-22
**Parent:** `.planning/notes/gradient-family-gap-closure-PLAN.md` §4 "Wave 5"
**Sibling:** `.planning/notes/gradient-gap-wave-4-PLAN.md` (Wave 4 — COMPLETE)
**Milestone:** v1.4 — Full libcint 6.1.3 Family Parity
**Baseline commit:** `3829c35` + the uncommitted Wave-4 working tree
**Audience:** an execution agent that follows instructions literally and does NOT infer.

---

## 0. STATUS — executed 2026-08-22

`cargo run -q -- manifest-audit --check-lock` is **`status: ok` for the first time**,
and honestly so: `uncovered_count = 0`, with every remaining unproven row carrying an
explicit policy and a named owner.

| Task | Outcome |
|---|---|
| **W5-00** | **DONE.** `unsupported_policy` threaded through the lock → `build.rs` → `ManifestEntry`; audit split into `uncovered` / `fail_closed` / `no_upstream_oracle`; `policy_contradictions` fails unconditionally. Guard test `manifest_fail_closed_policy.rs` asserts the correspondence in both directions. |
| **W5-01** | **PREMISE DISPROVEN — see §0.1.** The five rows cannot be oracle-gated at all. Recorded as `no_upstream_oracle`; `oracle_covered` stays `false` per RULE 4. |
| **W5-02** | **DONE.** Aux-k contraction added to the shared arity-3 spinor derivative transform. `int3c2e_ip1_spinor` / `ip2_spinor` **re-proven at `nctr_k = 2`** — their previous coverage claim could not have exercised that path. |
| **W5-03** | **NOT STARTED** (11 rows). |
| **W5-04** | **DECIDED — permanent v1.4 deferral.** See §0.2. |
| **W5-05** | **NOT STARTED** (7 families). |
| **W5-06** | **DONE.** `int1e_pnucp` + `int1e_prinvp`, **cart + sph + spinor**, byte-identical at `1e-12` across s/p/d shells. |
| **W5-07** | Baseline mechanism landed and live; hand-back plan written. ROADMAP/CHANGELOG pending. |

### 0.1 The W5-01 finding: there is no oracle to prove those rows against

The five rows §1.1 called "test and flip" sit behind **unconditional stubs inside
libcint 6.1.3 itself**:

* `CINT3c1e_spinor_drv` — `fprintf` + `exit(1)` (`src/cint3c1e.c:450-455`), taking out
  `int3c1e_spinor`, `int3c1e_ip1_spinor`, `int3c1e_iprinv_spinor`;
* `int2c2e_ip1_spinor` / `ip2_spinor` / `ip1ip2_spinor` — write nothing and **return 0**
  (`src/autocode/int3c2e.c:384`, `:462`, `:1366`). These fail **silently**, so a naive
  vendor test passes against an all-zero buffer. The byte-identity assertion in the new
  test only caught it because it asserts the reference is non-zero first.

`int2c2e_spinor` is NOT affected: `CINT2c2e_spinor_drv` stubs only for
`ncomp_e1 > 1 || ncomp_e2 > 1` (`cint2c2e.c:297-300`) and the base family has
`ncomp == 1`. Its existing coverage claim is sound — my first classifier pass wrongly
flagged it, and re-reading the guard corrected that.

Consequence: a **third** manifest state was needed. `no_upstream_oracle` means *cintx
evaluates this, and byte-identity is unobtainable at any effort*. Six rows carry it.
Corroboration (determinism + non-degeneracy + shared-transform provenance) is asserted
in `gradient_gap_wave5_groupa.rs` and labelled explicitly as **not** oracle proof.

### 0.2 W5-04 decided: `int1e_ecp_iprinv_spinor` is permanently deferred for v1.4

There is **no oracle anywhere**, not merely no implementation: libcint has no ECP at
all, and PySCF's `nr_ecp_deriv.c` contains **zero** spinor code (only `nr_ecp.c`'s
`ECPso_spinor`, a different operator). The row could never flip to
`oracle_covered = true` under RULE 4 even if implemented, so implementing it was the
wrong call. Recorded in the manifest with `owner = "deferred-v1.4"` and the full reason.

### 0.3 Corrections to this plan's own §1, found by executing it

* **§1.1 Group A is not "test and flip"** — it is unprovable. See §0.1.
* **`int1e_pnucp_spinor` / `int1e_prinvp_spinor` are provable and now proven.** I first
  registered them `fail_closed`; the W5-00 guard test immediately failed with *"marked
  fail_closed but RETURNED DATA"*, because the deriv34 spinor fold already handles them
  and `CINT1e_spinor_drv` is a real driver. The policy was dropped and all three
  representations proven. **This is the guard doing exactly what it was built for.**
* **`int1e_prinvp` needed origin plumbing neither existing predicate covered.** Its name
  contains neither `iprinv` nor the `int1e_rinv_`/`int1e_drinv_` prefix, so
  `is_iprinv_family_symbol` (`raw.rs`) and `validate_rinv_orig_env_params`
  (`validator.rs`) both missed it and the kernel was reached with `rinv_orig = None`.
  Both now name it explicitly.
* **The deriv34 machine generalised without modification.** `pnucp`/`prinvp` needed only
  a `FamilySpec` (3 ops, a 9-entry 2-leg `s` table, one dot term) — no new engine, no new
  contraction path. RULE 2 held exactly as written.

### 0.4 What remains

**W5-03 (11 spinor rows)** and **W5-05 (7 families)** are not started. W5-05's remaining
families need new device kernels or a new `R_I`/`R_J` op in the deriv34 `Op` enum
(`int1e_iprinvr` and the three rank-81 lresc families carry an `r` operator the enum
cannot express today), plus ket-side overlap kernels for `int1e_ovlpip` / `int1e_kinip`.
Neither was started rather than half-landed, because an unproven kernel behind a
`Stability::Stable` row is the exact defect this wave exists to eliminate.

---

## 0b. PREMISE CHECK (written BEFORE execution; retained for provenance)

Everything from here down is the plan as written before any code was touched. Where it
and §0 disagree, **§0 is the current state** — in particular §1.1's "Group A" and the
"52 unsupported" count are both superseded.

### 0b.1 The parent plan's Wave 5 is stale in three separate ways

Everything in §1 was **measured on 2026-08-22 by running the code**, not read off the
plan. Reproduction commands are in §7.

The parent plan (§1.3, §4/W5-01) says:

> "Gap B — 11 declared rows with `oracle_covered: false`. **These have kernels
> already** — the work is a vendor test per row plus whatever transform bug each one
> surfaces."

All three load-bearing claims in that sentence are wrong:

| Parent-plan claim | Measured reality |
|---|---|
| **11** uncovered rows | **50** uncovered `stability=stable` rows (68 in the CSV, of which 18 are `unstable_source` and out of lock scope) |
| "These have **kernels already**" | **45 of the 50 return a typed `UnsupportedApi`.** Only **5** evaluate at all. The work is *implementation*, not verification. |
| W5-01 is "a vendor test per row" | 45 rows need a spinor kernel or transform first; ~46 of the 50 also need a bindgen allowlist entry and a `vendor_ffi` wrapper that do not exist yet |

And the Tier-6 list in §1.3/W5-03 is **missing three families**: `int1e_rinvipiprip`
(lresc.c), `int1e_ovlpip` and `int1e_kinip` (grad1.c) are all `ALL_CINT1E`-exported
derivative families absent from the manifest and absent from the parent plan's tables.

**Two further facts that change scheduling:**

1. **The manifest audit is RED today.** `cargo run -q -- manifest-audit --check-lock`
   exits non-zero with `status: "failed"` and `uncovered_count: 50`. This is not a
   Wave-5 regression — it is the pre-existing state — but it means Wave 5 cannot use
   "audit green" as an incremental signal until W5-06 lands.
2. **W5-04's exit condition is not achievable by Wave 5.** `unsupported_count` is 52,
   and only **9** of those 52 are derivative families belonging to this plan. The other
   **43** are Phase-30 (GIAO×σ) and Phase-31 (Breit/Gaunt/gauge) scope. See §1.4.

---

## 1. VERIFIED STATE — measured 2026-08-22

### 1.1 The 50 uncovered rows, classified by what actually happens when you call them

Probe: a `d`-shell (`ANG_OF=2`) `nctr=2` fixture with non-zero `PTR_COMMON_ORIG` and
`PTR_RINV_ORIG`, `eval_raw(RawApiId::Symbol(sym), …)` per row.

| Group | Rows | Behaviour today | Rejection site |
|---|---:|---|---|
| **A — evaluable** | **5** | returns numbers | — |
| **B — this plan's own debt** | **11** | typed `UnsupportedApi` | `two_electron.rs:1463/1730/2025`, `center_2c2e.rs`, `center_3c2e.rs:3119-3122`, `one_electron.rs:9717` |
| **C — Phase 24 / 26 leftovers** | **32** | typed `UnsupportedApi` | `one_electron.rs:9378/9467/9597/9855/9952`, `two_electron.rs:2729` |
| **D — Phase 30 scope** | **1** | typed `UnsupportedApi` | `one_electron.rs` (`spgsa01`) |
| **E — ECP** | **1** | typed `UnsupportedApi` (R5) | `ecp.rs:2047` |

#### Group A — the only 5 rows that are "test and flip"

| Symbol | rank | Caveat |
|---|---:|---|
| `int3c1e_spinor` | 1 | none |
| `int2c2e_ip1_spinor` | 3 | none — `vendor_ffi` wrapper already exists |
| `int2c2e_ip2_spinor` | 3 | none |
| `int3c1e_ip1_spinor` | 3 | **fails closed for `nctr_k > 1`** (`center_3c1e.rs:1182`) |
| `int3c1e_iprinv_spinor` | 3 | same `nctr_k > 1` limit |

#### Group B — 11 derivative spinor rows this plan is responsible for

| Symbol | rank | Origin |
|---|---:|---|
| `int1e_drinv_spinor` | 3 | parent §1.3 Gap B |
| `int2e_ip1_spinor` | 3 | parent §1.3 Gap B |
| `int2e_ip2_spinor` | 3 | parent §1.3 Gap B |
| `int2e_ipip1_spinor` | 9 | parent §1.3 Gap B |
| `int2e_ipvip1_spinor` | 9 | parent §1.3 Gap B |
| `int2e_ip1ip2_spinor` | 9 | parent §1.3 Gap B |
| `int2e_ipip1ipip2_spinor` | 81 | Phase 25 leftover |
| `int2e_ipvip1ipvip2_spinor` | 81 | **created by W1-03** |
| `int2c2e_ip1ip2_spinor` | 9 | **created by W2-01** |
| `int3c2e_ipvip1_spinor` | 9 | **created by W2-02** |
| `int3c2e_ip1ip2_spinor` | 9 | **created by W2-03** |

Waves 1–2 each registered a spinor row alongside their cart/sph rows and left it
`oracle_covered=false`. That is four rows of debt this plan created and must retire.

#### Group C — 32 rows that belong to Phases 24 and 26, not to this plan

These are **deliberate, documented deferrals**, not defects. `one_electron.rs:9599`
says so verbatim: *"Spinor moment reps are registered for surface completeness but not
implemented: fail typed, never partial (D-09)."* The Phase-26 sites carry the same note
under D-11.

| Sub-group | Rows | Symbols |
|---|---:|---|
| 1e moment | 14 | `int1e_{r,rr,rrr,rrrr,r2,r4,z,zz}_spinor` + the six `_origj` forms |
| 1e rinv | 1 | `int1e_rinv_spinor` |
| 1e p4 | 1 | `int1e_p4_spinor` |
| 1e irp | 1 | `int1e_irp_spinor` |
| 1e GIAO | 11 | `govlp, gnuc, igovlp, ignuc, igkin, a01gp, ia01p, cg_irxp, giao_irjxp, cg_a11part, giao_a11part` |
| 2e GIAO | 4 | `int2e_{g1,ig1,gg1,g1g2}_spinor` |

### 1.2 A shipped coverage claim that is narrower than it looks

`int3c2e_ip1_spinor` and `int3c2e_ip2_spinor` carry `oracle_covered = true`
(`api_manifest.csv:23`, `:277`) **but fail closed for `nctr_k > 1`**
(`center_3c2e.rs:2657`, `:3035`). The wrapper
`cart_to_spinor_sf_derivative_3c2e` indexes device blocks as
`[(ci*n_ctr_j+cj)][comp][k][j][i]` — a single spherical aux-k axis per `(ci,cj)`
sub-block — and does not handle a contracted aux-k axis.

W0-04 of the parent plan mandates a `nctr > 1` fixture. Those two rows were proven on a
fixture that cannot have had `nctr_k > 1`. This is a RULE-4 honesty gap on **shipped,
`Stability::Stable`, `oracle_covered=true` API**, and it shares one root cause with two
of the Group-A rows. Fixing the aux-k contraction once retires all four.

### 1.3 The manifest cannot express "declared but fail-closed"

45 rows say `stability="stable"`, `forms="spinor"`, `oracle_covered=false` while the
kernel returns `UnsupportedApi`. The parent plan flags exactly this state for **one**
row (W5-02: *"do not leave it as a bare rejection with a `Stability::Stable` manifest
row — that combination is the one state the audit cannot express"*).

It is not one row. It is 45, and it is why `manifest-audit --check-lock` is red.
`Stability` today is `Stable | Optional | UnstableSource | Other(_)`
(`crates/cintx-ops/src/resolver.rs:30`) — there is no value that means *"this symbol and
representation are public API, fail closed by design, and not yet implemented."*

**Note on RULE 7 (parent plan §2.1) — it is backwards.** The CSV is *generated*, not
edited: `crates/cintx-ops/build.rs:201-202` writes **both** `api_manifest.rs` and
`api_manifest.csv` from `crates/cintx-ops/generated/compiled_manifest.lock.json`. The
lock JSON is the source of truth. Edit the lock, rebuild `cintx-ops`, then verify the
two generated files moved.

### 1.4 Gap C — the 52 unsupported symbols split 9 / 43

Classified by the `ALL_CINT*` export site in `libcint-master/src/`:

| Source file | Count | Owner | Symbols |
|---|---:|---|---|
| `autocode/lresc.c` | 4 | **Wave 5** | `int1e_iprinvr`, `int1e_iprinviprip`, `int1e_ipiprinvrip`, `int1e_rinvipiprip` |
| `autocode/hess.c` | 1 | **Wave 5** | `int1e_iprip` |
| `autocode/grad1.c` | 2 | **Wave 5** | `int1e_ovlpip`, `int1e_kinip` |
| `autocode/intor1.c` (X2C base) | 2 | **Wave 5 (recommended)** | `int1e_pnucp`, `int1e_prinvp` |
| `autocode/intor1.c` (GIAO/property) | 8 | Phase 30 | `ggovlp, ggnuc, ggkin, grjxp, irpr, irrp, pnucxp, prinvxp` |
| `autocode/intor2.c` | 3 | Phase 30 | `int1e_inuc_rxp`, `int1e_inuc_rcxp`, `int2e_p1vxp1` |
| `autocode/intor3.c` | 4 | Phase 30 | `sa01sp, sprsp, spsigmasp, srsp` |
| `autocode/intor4.c` | 9 | Phase 30 | `cg_sa10sp1, giao_sa10sp1, …, pp1, pp2, pp1pp2, spgsp1, …` |
| `autocode/int3c2e.c` | 5 | Phase 30/31 | `ig1, pvp1, pvxp1, spsp1, spsp1ip2` |
| `autocode/gaunt1.c` | 3 | Phase 31 | `cg_ssa10ssp2, giao_ssa10ssp2, gssp1ssp2` |
| `autocode/breit1.c` | 8 | Phase 31 | the `gauge_r1_*` / `gauge_r2_*` set |
| `autocode/dkb.c` | 2 | Phase 31 | `int1e_spnuc`, `int1e_spspsp` |

**`int1e_pnucp` and `int1e_prinvp` deserve a note.** Wave 3 shipped their *derivatives*
(`int1e_ippnucp`, `int1e_ipprinvp`, and the four `pip`/`ipip` forms) while the **base**
families are still absent. `pyscf/x2c/x2c.py` calls `int1e_pnucp` directly; the X2C
symbol set W3's gate claimed to satisfy is therefore only satisfiable for the *gradient*,
not for the underlying X2C Hamiltonian. Both are `ng={1,1,0,0,2,1,0,1}`, rank 1, on the
nuclear-Rys and rinv-Rys engines Wave 3 already extended. They are the cheapest families
in this plan.

### 1.5 What is already wired and must NOT be redone

* **`lresc.c` is already in the oracle `cc::Build`** (`crates/cintx-oracle/build.rs:280`),
  and all four lresc/hess Tier-6 symbols are **already in the bindgen allowlist**
  (`build.rs:405`, the `(…|int1e_iprinvr|int1e_iprip|int1e_iprinviprip|int1e_ipiprinvrip)_(cart|sph|spinor)`
  group). Parent-plan risk R-05 is closed. Only `vendor_ffi` wrappers are missing.
* Same for the four Group-B rows created by Waves 1–2: `int2e_ipvip1ipvip2`,
  `int2c2e_ip1ip2`, `int3c2e_ipvip1`, `int3c2e_ip1ip2` are allowlisted in all three
  representations at `build.rs:405`.
* `int1e_ovlpip`, `int1e_kinip`, `int1e_pnucp`, `int1e_prinvp` are **not** allowlisted;
  `intor1.c` and `grad1.c` are both already in the `cc::Build` list (`:232`, `:271`).
* Group-C spinor symbols (`int1e_r_spinor`, `int1e_govlp_spinor`, …) are allowlisted
  only in `_sph|_cart` form at `build.rs:401`. Their `_spinor` variants need appending.

---

## 2. SCOPE DECISION — read this before doing anything

Taken literally, the parent plan's Wave 5 now means **50 spinor implementations + 9 new
families + closing a 52-symbol gate**. That is not one wave; Phases 24 and 26 each
deferred their spinor slice deliberately, and absorbing both here would silently
re-scope two closed phases into this plan.

**Recommendation — Wave 5 owns 28 rows and hands back 33:**

| In scope for Wave 5 | Rows / families |
|---|---:|
| Group A — test and flip | 5 rows |
| Group B — this plan's own derivative debt | 11 rows |
| Group E — ECP `iprinv` spinor decision | 1 row |
| §1.2 aux-k contraction honesty fix | 2 shipped rows re-proven |
| Tier 6 + omitted derivative families | 7 new families |
| X2C base families (recommended) | 2 new families |
| Manifest honesty + gate baseline | — |

| Handed back, with a written disposition | Rows |
|---|---:|
| Group C → a new "Phase 24/26 spinor completion" plan | 32 |
| Group D `int1e_spgsa01_spinor` → Phase 30 | 1 |
| 43 non-derivative unsupported symbols → Phases 30/31 | — |

**If the milestone owner instead wants Wave 5 to absorb Group C, say so explicitly and
re-plan it as its own wave** — it is three distinct transform classes (moment, GIAO
overlap-engine, GIAO nuclear-engine) and, on the Phase 30-01d precedent, is the largest
single block of work left in v1.4. Do not absorb it silently by treating W5-01 as "all
uncovered rows".

---

## 3. TASKS

Standing rules RULE 1–7 of the parent plan apply, **except RULE 7, which §1.3 corrects**:
the manifest source of truth is `crates/cintx-ops/generated/compiled_manifest.lock.json`,
not the CSV.

### W5-00 — Make the audit tell the truth (blocking, do this first)

The gate is red and cannot distinguish "not yet proven" from "fail-closed by design".
Until it can, no other task has a usable green signal.

1. Add a representation-level `unsupported_policy` field to the compiled manifest lock
   schema, carrying a reason string and the owning phase — e.g.
   `{"policy": "fail_closed", "reason": "D-09 spinor moment deferral", "owner": "phase-24"}`.
   Thread it through `crates/cintx-ops/build.rs` into both generated files and into
   `ManifestEntry` / `resolver.rs`.
2. Populate it for all 45 fail-closed rows from §1.1 groups B/C/D/E. **The reason string
   must name the rejection site** (`one_electron.rs:9597`, `ecp.rs:2047`, …) so the next
   agent can find the code from the manifest alone.
3. `xtask::manifest_audit::check_oracle_coverage` (`xtask/src/manifest_audit.rs:342`)
   currently counts every `stability == "stable" && !oracle_covered` row. Change it to
   report three buckets: `uncovered` (implemented, unproven), `fail_closed` (declared,
   `unsupported_policy` set), and `undeclared`. Only `uncovered` fails the gate.
4. Add a test asserting that **every** row carrying `unsupported_policy: fail_closed`
   actually returns `UnsupportedApi` when called, and that no row *without* it does.
   This is the structural guard: it makes the parent plan's W0-06 fail-open class of
   defect impossible to reintroduce, and it catches the reverse — a row that quietly
   starts working while still marked fail-closed.

**Gate:** `cargo run -q -- manifest-audit --check-lock` exits 0 with
`oracle_coverage.uncovered_count` reflecting only genuinely-unproven rows, and the new
round-trip test green.

### W5-01 — Group A: five rows that only need a test (small, high confidence)

1. `vendor_ffi` wrappers + `build.rs:401` allowlist entries for `int3c1e_spinor`
   (exists), `int2c2e_ip1_spinor` (exists), `int2c2e_ip2_spinor`,
   `int3c1e_ip1_spinor` (exists), `int3c1e_iprinv_spinor` (exists).
2. One test file `crates/cintx-oracle/tests/gradient_gap_wave5_groupa.rs`, `d`-shell
   `nctr=2`, `ATOL = 1e-12`, **per-family reporting with a single terminal assert**
   (the W4-00 pattern — do not assert inside the loop).
3. Flip the five rows in the lock; regenerate.

For the two `int3c1e_*` rows the fixture must use `nctr_k = 1` **until W5-02 lands**,
and the test must carry a comment saying so with a pointer to W5-02. Do not quietly ship
an `nctr_k = 1` fixture as if it satisfied W0-04.

**Gate:** five rows `oracle_covered=true`; the two 3c1e rows re-run at `nctr_k = 2` after
W5-02 without edits beyond the fixture constant.

### W5-02 — Contracted aux-k in the spinor derivative transform (§1.2)

One root cause, four rows.

`cart_to_spinor_sf_derivative_3c2e` (and the 3c1e sibling behind
`center_3c1e.rs:1182`) assumes one spherical aux-k axis per `(ci, cj)` sub-block. Give
it the aux-k contraction axis. Then:

* remove the `n_ctr_k > 1` rejections at `center_3c2e.rs:2657`, `:3035` and
  `center_3c1e.rs:1182`;
* **re-prove `int3c2e_ip1_spinor` and `int3c2e_ip2_spinor` at `nctr_k = 2`** — they
  claim `oracle_covered=true` today on a fixture that cannot have exercised this path;
* raise the W5-01 3c1e fixture to `nctr_k = 2`.

**Gate:** four rows green at `1e-12` with `nctr_i = nctr_j = nctr_k = 2`; no
`nctr_k > 1` rejection remains in any 3-center spinor path.

### W5-03 — Group B: 11 derivative spinor rows

The largest correctness task in the wave. All 11 need a spinor kernel path, not a test.

Sequence by transform class, cheapest proof first:

| Step | Rows | Notes |
|---|---|---|
| **W5-03a** | `int2e_ip1_spinor`, `int2e_ip2_spinor` (rank 3) | `two_electron.rs:1463`, `:1730`. The thinnest 2e spinor derivative and the proof vehicle for the rest. Wave 4 §0.2 fixed the missing KET→BRA transpose in `cart_to_spinor_sf_4d`; **re-read that fix before writing anything** — it is the reason these may be much closer than they look. |
| **W5-03b** | `int2e_ipip1`, `ipvip1`, `ip1ip2` (rank 9), `ipip1ipip2`, `ipvip1ipvip2` (rank 81) | `two_electron.rs:2025` rejects the whole `Hess2eKind` set at once; `:2285` is an `unreachable!` that must be replaced, not deleted. Rank 81 is host-routed already for cart/sph — reuse that route. |
| **W5-03c** | `int2c2e_ip1ip2_spinor`, `int3c2e_ipvip1_spinor`, `int3c2e_ip1ip2_spinor` | `center_3c2e.rs:3119-3122`. Depends on W5-02 (aux-k). |
| **W5-03d** | `int1e_drinv_spinor` | `one_electron.rs:9717`. Rank 3, rinv engine, no Rys sharing with the above — independent, can run in parallel. |

Per row: bindgen allowlist entry, `vendor_ffi` wrapper, dispatch/transform work, a
non-`#[ignore]`d vendor test at `1e-12` on the `d`-shell `nctr=2` fixture, then flip.

**Budget generously.** Phase 30-01d and Wave 4's RC-1/RC-2 are both precedents for a
single spinor family hiding a multi-day residual hunt. RULE 4 stands: leave
`oracle_covered=false` and the row `fail_closed` rather than over-claim.

**Gate:** 11 rows `oracle_covered=true`, `unsupported_policy` removed from each.

### W5-04 — Group E: `int1e_ecp_iprinv_spinor` — decide, do not defer again

`ecp.rs:2047` rejects it under the Phase-21 R5 deferral. Two acceptable outcomes, one
unacceptable one.

* **Implement it** — spinor ECP gradient path, vendor-gated against PySCF's
  `nr_ecp_deriv.c` (already in the oracle `cc::Build` at `build.rs:453`), flip to
  `oracle_covered=true`.
* **Or declare it permanently fail-closed** — `unsupported_policy` from W5-00 with
  `owner: "phase-21-R5"` and a reason naming `ecp.rs:2047`, plus a CHANGELOG line
  saying the spinor ECP gradient is out of scope for v1.4.

**Not acceptable:** leaving it as a bare rejection behind a `Stability::Stable` row —
that is the exact state W5-00 exists to eliminate.

Note the probe hit `AS_NECPBAS` validation before reaching the R5 rejection, so any test
here needs a real ECP fixture — reuse `crates/cintx-oracle/tests/ecp_iprinv_parity.rs`'s.

### W5-05 — Tier 6 and the three families the parent plan missed

Seven new families, cart + sph + spinor. All ride engines cintx already owns (parent
plan §3); RULE 2 applies — new code is one `gout` + one dispatch arm + manifest rows +
one vendor test each.

| Symbol | source | `ng[]` | rank | drv type | engine |
|---|---|---|---:|---:|---|
| `int1e_ovlpip` | `grad1.c` | `{0,1,0,0,1,1,1,3}` | 3 | 0 | 1e overlap-deriv, **ket-side ∇** |
| `int1e_kinip` | `grad1.c` | `{0,3,0,0,3,1,1,3}` | 3 | 0 | 1e overlap-deriv, ket-side |
| `int1e_iprinvr` | `lresc.c` | `{1,1,0,0,2,1,0,9}` | 9 | 1 | 1e rinv-Rys |
| `int1e_iprip` | `hess.c` | `{1,2,0,0,3,1,1,27}` | 27 | 0 | 1e overlap-deriv |
| `int1e_iprinviprip` | `lresc.c` | `{1,3,0,0,4,1,0,81}` | 81 | 1 | 1e rinv-Rys |
| `int1e_ipiprinvrip` | `lresc.c` | `{2,2,0,0,4,1,0,81}` | 81 | 1 | 1e rinv-Rys |
| `int1e_rinvipiprip` | `lresc.c` | `{0,4,0,0,4,1,0,81}` | 81 | 1 | 1e rinv-Rys, **ket-only ∇⁴** |

`int1e_ovlpip` and `int1e_kinip` are **ket-side** gradients — the mirror of the shipped
`int1e_ipovlp` / `int1e_ipkin`, not the both-side `int1e_ipovlpip`. Do not assume the
bra-side arm can be reused with a sign flip; transcribe `CINTgout1e_int1e_ovlpip` from
`grad1.c`.

**`nroots` per §3 of the parent plan, computed for `li = lj = 2`:**
`iprinvr` → 4 (device OK); `iprip` → 4; `iprinviprip`, `ipiprinvrip`, `rinvipiprip` →
**5, at `MAX_DEVICE_NROOTS`**. At `f`-shells all three rank-81 families reach 6 and
exceed it. **Host-route all three rank-81 families from the start** via the
`kernels/deriv34.rs` pattern, and add an `f`-shell case to their tests — do not build a
device kernel and discover this at the gate (parent-plan R-02).

Oracle wiring for all five lresc/hess symbols is **already done** (§1.5) — only
`vendor_ffi` wrappers are needed. `int1e_ovlpip` / `int1e_kinip` need allowlist entries.

**Gate:** seven families × cart + sph green at `1e-12` on the `d`-shell `nctr=2` fixture,
plus an `f`-shell case for the three rank-81 families. Spinor rows for these families
ship `fail_closed` under W5-00 unless separately proven — declare that explicitly rather
than registering silent `oracle_covered=false` rows and repeating the Group-B mistake.

### W5-06 — X2C base families `int1e_pnucp` / `int1e_prinvp` (recommended, §1.4)

Both `ng={1,1,0,0,2,1,0,1}`, rank 1, `CINT1e_drv(…, 2)` and `(…, 1)` respectively —
the nuclear-Rys and rinv-Rys engines Wave 3 already extended for their derivatives.
`nroots = 4` at `d`-shells, device-resident. Two `gout` transcriptions from `intor1.c`,
two dispatch arms, two vendor tests.

This closes the gap between W3's claimed X2C symbol coverage and what `pyscf/x2c/x2c.py`
actually calls.

**Gate:** both families cart + sph green at `1e-12`; `pyscf/x2c/x2c.py` symbol
requirements satisfiable, not just `sfx2c1e_grad.py`'s.

### W5-07 — Close what this plan can close of the parity gate; hand back the rest

1. `unsupported_count` drops **52 → 43** (W5-05's seven plus W5-06's two leave the list).
2. **Do not flip `CINTX_PARITY_STRICT=1` on by default** — the parent plan's W5-04 asks
   for that, but 43 symbols legitimately belong to Phases 30 and 31, so a default-strict
   gate would be permanently red and would train everyone to ignore it. Instead:
   * add a dated, explicitly-enumerated `parity_baseline` allowlist to
     `xtask/src/manifest_audit.rs` holding exactly those 43 symbols, each tagged with its
     owning phase from §1.4;
   * make the audit **fail** (unconditionally, no env var) if `unsupported_libcint_families`
     contains anything **not** on that baseline, or if the baseline contains a symbol that
     is no longer unsupported. That makes a future Gap-A-shaped omission mechanically
     impossible — which is what Gap C was actually about — while leaving Phase 31 the job
     of emptying the baseline.
   * `CINTX_PARITY_STRICT=1` keeps its current meaning (list must be *empty*) and becomes
     Phase 31's exit gate.
3. Write the Group-C hand-back: a new
   `.planning/notes/phase-24-26-spinor-completion-PLAN.md` enumerating the 32 rows from
   §1.1/Group C with their rejection sites and transform classes. Reference it from the
   `unsupported_policy` reason strings so the two artefacts cannot drift.
4. Update `.planning/ROADMAP.md` Phase 31 success criterion 4 to reference the widened
   reference set and the baseline mechanism. Refresh `CHANGELOG.md`.
5. Append to `.planning/notes/phase-21-pyscf-rs-handoff.md`: the X2C base families from
   W5-06 and, if W5-04 lands as a permanent deferral, the spinor-ECP-gradient limitation.

---

## 4. WAVE 5 EXIT GATE

1. `cargo run -q -- manifest-audit --check-lock` exits **0**, with every remaining
   uncovered row either proven or carrying an `unsupported_policy` naming its owner.
2. Group A (5) + Group B (11) rows `oracle_covered=true`; §1.2's two shipped 3c2e rows
   re-proven at `nctr_k = 2`.
3. W5-05's seven and W5-06's two families green cart + sph at `1e-12`, `f`-shell case
   present for every rank-81 family.
4. `unsupported_count` = 43, all 43 on the dated `parity_baseline`, and the audit fails
   on any symbol outside it.
5. `int1e_ecp_iprinv_spinor` has a decision recorded in the manifest and the CHANGELOG —
   implemented or explicitly deferred, not silent.
6. Group C's 32 rows have a written owner and a plan file. **Wave 5 does not implement
   them.**
7. Every prior suite still green: `rel_2e_sigma_parity` (18/18),
   `two_electron_spinor_orientation`, `gradient_gap_tier{1,2,3,4,5}*`.

---

## 5. RISKS

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| W5-R1 | W5-00's schema change touches every generated manifest row and breaks the `compiled_manifest.lock.json` ↔ generated-files agreement tests | **MAJOR** | Land W5-00 alone, on its own commit, with `cargo test -p cintx-ops` green before any family work. It is the wave's only cross-cutting change. |
| W5-R2 | Group B's 11 spinor rows hide multi-day residual hunts (Phase 30-01d, Wave 4 RC-1/RC-2) | **MAJOR** | Prove `int2e_ip1_spinor` first as the class's proof vehicle. Keep unproven rows `fail_closed` rather than over-claim (RULE 4). |
| W5-R3 | W5-02's aux-k rework silently changes `int3c2e_ip1/ip2_spinor` output at `nctr_k=1` — rows that ship today | **MAJOR** | Pin the current `nctr_k=1` values in a regression test **before** touching the transform; the fix must be additive at `nctr_k=1`. |
| W5-R4 | Rank-81 Tier-6 families built as device kernels, then found to exceed `MAX_DEVICE_NROOTS` at `f`-shells | MAJOR | §W5-05 — host-route from the start; `f`-shell case in the test, not added later. |
| W5-R5 | Nine new families × three representations shift `OperatorId` positional indices | MAJOR | Parent-plan R-04: **append** rows; symbol-name dispatch only (RULE 5); re-run `ecp_operator_ids_match_constants`. |
| W5-R6 | Wave 5 quietly absorbs Group C and doubles in size | **MAJOR** | §2 is the contract. Group C is handed back in writing (W5-07.3) before Wave 5 can close. |
| W5-R7 | `int1e_ovlpip`/`int1e_kinip` implemented as sign-flipped bra-side arms | MINOR | They are ket-side (`ng[1]` headroom, `ng[0]=0`). Transcribe from `grad1.c` per RULE 1. |
| W5-R8 | The `parity_baseline` allowlist becomes a dumping ground | MINOR | Each entry carries an owning phase; the audit fails if a baseline symbol becomes supported, forcing the list to shrink monotonically. |

---

## 6. ORDER

```
W5-00  (manifest honesty + audit buckets)   ← blocking, own commit
   ├── W5-01  (Group A, 5 rows)             ← smallest real win
   ├── W5-02  (aux-k contraction)           ← unblocks W5-01's 3c1e fixture + W5-03c
   │      └── W5-03c (2c2e/3c2e Hessian spinor)
   ├── W5-03a → W5-03b                      ← 2e spinor gradient then Hessian
   ├── W5-03d (drinv spinor)                ← independent
   ├── W5-05  (Tier 6 + grad1/hess families) ← independent of all spinor work
   ├── W5-06  (X2C base families)           ← independent, cheapest, has a consumer
   └── W5-04  (ECP decision)                ← independent
W5-07  (gate baseline + hand-backs)         ← last
```

W5-05 and W5-06 touch only cart/sph on 1e engines and share no code with the spinor
tasks — they are the natural parallel track. W5-06 is the single highest value-per-hour
item in the wave (two rank-1 families, a live PySCF consumer, engines already extended).

---

## 7. REPRODUCTION COMMANDS

```bash
cd /home/user/Documents/workspace/cintx

# §1.1 — the 50 uncovered stable rows (this is the authoritative list)
cd xtask && cargo run -q -- manifest-audit --check-lock; cd ..
python3 -c "import json;d=json.load(open('/tmp/cintx_artifacts/cintx_phase_04_manifest_audit.json'));\
print(d['status'], d['oracle_coverage']['uncovered_count']);\
print('\n'.join(d['oracle_coverage']['uncovered_stable_entries']))"

# §1.4 — the 52 unsupported symbols and their export sites
python3 -c "import json;d=json.load(open('/tmp/cintx_artifacts/cintx_phase_04_manifest_audit.json'));\
print('\n'.join(d['libcint_export_parity']['unsupported_libcint_families']))"
for s in int1e_iprinvr int1e_iprip int1e_ovlpip int1e_pnucp; do
  echo "$s -> $(grep -rl "ALL_CINT1\?E\?($s)" libcint-master/src/)"; done

# §1.1 — reproduce the evaluability probe (write it, run it, DELETE it)
#   model on crates/cintx-oracle/tests/gradient_gap_tier5_gauge2e.rs; call
#   eval_raw(RawApiId::Symbol(sym), …) per row and print Ok/Err, do not assert.

# §1.2 — the aux-k limit
sed -n '2645,2665p' crates/cintx-cubecl/src/kernels/center_3c2e.rs
grep -n 'nctr_k>1' crates/cintx-cubecl/src/kernels/center_3c1e.rs

# W5-05 vendored transcription targets
awk '/void CINTgout1e_int1e_iprinvr\(/,/^FINT int1e_iprinvr/'    libcint-master/src/autocode/lresc.c
awk '/void CINTgout1e_int1e_rinvipiprip\(/,/^FINT/'              libcint-master/src/autocode/lresc.c
awk '/void CINTgout1e_int1e_iprip\(/,/^FINT/'                    libcint-master/src/autocode/hess.c
awk '/void CINTgout1e_int1e_ovlpip\(/,/^FINT/'                   libcint-master/src/autocode/grad1.c
awk '/void CINTgout1e_int1e_kinip\(/,/^FINT/'                    libcint-master/src/autocode/grad1.c
awk '/void CINTgout1e_int1e_pnucp\(/,/^FINT/'                    libcint-master/src/autocode/intor1.c
awk '/void CINTgout1e_int1e_prinvp\(/,/^FINT/'                   libcint-master/src/autocode/intor1.c

# Regression guards that must stay green
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
  --test rel_2e_sigma_parity --test two_electron_spinor_orientation \
  --test gradient_gap_tier1 --test gradient_gap_tier1_2e --test gradient_gap_tier2 \
  --test gradient_gap_tier3 --test gradient_gap_tier4_1e_sigma \
  --test gradient_gap_tier4_2e_sigma --test gradient_gap_tier5_3c2e_sigma \
  --test gradient_gap_tier5_gauge2e
```
