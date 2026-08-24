# Phase 24 / 26 Spinor Completion — the 32 rows Wave 5 handed back

**Created:** 2026-08-22 (Wave 5, task W5-07)
**Parent:** `.planning/notes/gradient-gap-wave-5-PLAN.md` §2 (scope decision)
**Milestone:** v1.4 — Full libcint 6.1.3 Family Parity
**Status:** NOT STARTED — this file exists so the rows have a written owner, not
because work has begun.

---

## 0. Why this file exists

Wave 5 opened expecting 11 unproven spinor rows. It measured **50**, of which
**45 return a typed `UnsupportedApi`** — they have no spinor kernel at all.

Of those 45, **32 are not Wave 5's debt.** They are deliberate, documented
deferrals taken when Phases 24 and 26 closed:

> `one_electron.rs:9599` — *"Spinor moment reps are registered for surface
> completeness but not implemented: fail typed, never partial (D-09)."*

The Phase-26 sites carry the same note under D-11. Absorbing them into Wave 5
would have silently re-scoped two closed phases, so Wave 5 marked each row
`unsupported_policy = fail_closed` with `owner = "phase-24-26-spinor-completion"`
and stopped. **This file is that owner.**

The manifest and this file are cross-referenced deliberately: every row below
carries a `reason` string in `crates/cintx-ops/generated/compiled_manifest.lock.json`
naming its rejection site, so the two artefacts cannot drift apart silently.

---

## 1. The 32 rows, by rejection site

All are `stability = "stable"`, `forms = ["spinor"]`, `oracle_covered = false`,
`unsupported_policy.policy = "fail_closed"`.

### 1.1 Phase 24 — position / moment families (17 rows)

| Rejection site | Rows | Symbols |
|---|---:|---|
| `one_electron.rs:9597` (Cluster A moment) | 14 | `int1e_{r,rr,rrr,rrrr,r2,r4,z,zz}_spinor` + `int1e_{r,rr,r2,r4,z,zz}_origj_spinor` |
| `one_electron.rs:9717` (Cluster B rinv) | 1 | `int1e_rinv_spinor` |
| `one_electron.rs:9855` (Cluster C p4) | 1 | `int1e_p4_spinor` |
| `one_electron.rs:9952` (Cluster D irp) | 1 | `int1e_irp_spinor` |

Component ranks span 1 → 81 (`int1e_rrrr_spinor`). The `_origj` variants share
the base family's kernel and differ only in the origin source, so they should
come essentially free once the base spinor fold lands.

### 1.2 Phase 26 — GIAO families (15 rows)

| Rejection site | Rows | Symbols |
|---|---:|---|
| `one_electron.rs:9378` (GIAO-01 overlap engine, D-11) | 5 | `int1e_{govlp,igovlp,igkin,cg_irxp,giao_irjxp}_spinor` |
| `one_electron.rs:9467` (GIAO-01 nuclear engine, D-11) | 6 | `int1e_{gnuc,ignuc,ia01p,a01gp,cg_a11part,giao_a11part}_spinor` |
| `two_electron.rs:2729` (2e GIAO) | 4 | `int2e_{g1,ig1,gg1,g1g2}_spinor` |

These emit **purely imaginary** components in cart/sph (Phase 26's
complex-interleaved capability). The spinor fold must preserve that, so the
`re = 0 / im = value` convention at the existing sites is load-bearing context.

---

## 2. What makes this a phase and not a task

Three distinct transform classes, not one:

1. **Moment fold** — rank 1…81 on the overlap engine, no Rys.
2. **GIAO overlap-engine fold** — rank 3/9, imaginary output, reads
   `PTR_COMMON_ORIG` for `cg_irxp`.
3. **GIAO nuclear-engine fold** — rank 3/9, Rys atom-sum, and the four
   `ia01p`/`a01gp`/`*_a11part` families evaluate at the **single rinv center**
   (`env[PTR_RINV_ORIG]`, charge +1), not atom-summed — see the note at
   `crates/cintx-compat/src/raw.rs` `is_giao_rinv_center_symbol`.

On the Phase 30-01d precedent a single σ-transform residual consumed multiple
days. Budget accordingly; do not size this from the row count.

---

## 3. Before starting: check the oracle actually exists

**This is the trap Wave 5 fell into and it will bite here too.**

Wave 5 set out to "just write a test and flip" five rows, and discovered their
vendored drivers are **unconditional stubs inside libcint 6.1.3 itself**:

* `CINT3c1e_spinor_drv` — `fprintf` + `exit(1)` (`src/cint3c1e.c:450-455`);
* `int2c2e_ip1/ip2/ip1ip2_spinor` — write nothing and **return 0**
  (`src/autocode/int3c2e.c:384`, `:462`, `:1366`). These fail *silently*, so a
  naive vendor test passes against an all-zero buffer.

Those six rows are now `unsupported_policy = "no_upstream_oracle"`.

**For every row in §1, before writing any kernel, confirm the vendored driver is
real:**

```bash
cd /home/user/Documents/workspace/cintx/libcint-master
# 1. Find the driver the _spinor entry point calls.
awk '/CACHE_SIZE_T int1e_r_spinor\(/,/^}/' src/autocode/intor1.c
# 2. Confirm that driver is not a stub (and, if it is, whether the stub is
#    GUARDED — CINT2c2e_spinor_drv stubs only for ncomp>1, so the base family works).
grep -n "not implemented" -B 8 src/cint1e.c src/cint2e.c
```

A row whose driver is stubbed must be marked `no_upstream_oracle`, NOT
implemented-and-claimed. `oracle_covered` stays `false` either way.

---

## 4. Definition of done

Per row, unchanged from the parent plan §2.2, plus one addition:

1. Vendored driver confirmed real (§3) — **or** the row is marked
   `no_upstream_oracle` and closed without a kernel.
2. Spinor fold implemented; the rejection site's guard removed.
3. `vendor_<symbol>_spinor` wrapper + bindgen allowlist entry
   (`crates/cintx-oracle/build.rs:401` — the Group-C `_spinor` variants are NOT
   yet allowlisted; only their `_sph|_cart` forms are).
4. Non-`#[ignore]`d vendor test at `atol = 1e-12` on a `d`-shell `nctr = 2`
   fixture with a non-zero gauge origin.
5. `oracle_covered = true` **and** `unsupported_policy` removed — the audit fails
   if a row claims both (`policy_contradictions`).
6. `cargo run -q -- manifest-audit --check-lock` still `status: ok`.

`crates/cintx-oracle/tests/manifest_fail_closed_policy.rs` enforces the
correspondence in both directions: a row marked `fail_closed` that starts
returning data fails the test, and so does a row that claims coverage while
still carrying a policy.

---

## 5. Suggested order

```
1. int1e_r_spinor           — rank 3, thinnest moment; proof vehicle for §2 class 1
2. the remaining 13 moment rows + the 3 origj-sharing siblings
3. int1e_rinv_spinor, int1e_p4_spinor, int1e_irp_spinor
4. int1e_govlp_spinor       — proof vehicle for §2 class 2 (imaginary output)
5. the remaining 4 GIAO overlap-engine rows
6. int1e_gnuc_spinor        — proof vehicle for §2 class 3
7. the remaining 5 GIAO nuclear-engine rows
8. the 4 int2e GIAO rows
```

Steps 1, 4, 6 are the three real unknowns. Everything else in each group follows
its group's proof vehicle.
