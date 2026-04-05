# Phase 15: Oracle Tolerance Unification & Manifest Lock Closure - Research

**Researched:** 2026-04-06
**Domain:** Oracle parity infrastructure, manifest lock regeneration, CI gate closure
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** The oracle parity comparison (`compare.rs`) marks `oracle_covered=true` on each manifest entry that passes at atol=1e-12. Coverage is objective — only entries that actually passed get the flag.

**D-02:** `oracle_covered` persists in the committed `compiled_manifest.lock.json`. CI verifies the claim matches actual parity results. Drift between the flag and real oracle status = CI failure.

**D-03:** Any stable entry that fails oracle parity at 1e-12 is treated as a kernel bug to be fixed, not a tolerance to be loosened (per ORAC-01). Block until fixed.

**D-04:** Replace the explicit `tolerance_for_family()` match arms with a catch-all default returning `UNIFIED_ATOL`. The match arms become documentation, not gatekeeping. New families never cause "missing tolerance" errors.

**D-05:** Replace `PHASE4_ORACLE_FAMILIES` hardcoded list with manifest-driven oracle eligibility — derive oracle-eligible families from the manifest lock itself (any entry with `stability: stable` or `stability: optional`). No family allow-list to maintain.

**D-06:** Keep the manifest lock as a single file (`compiled_manifest.lock.json`) with all profiles. Regeneration is atomic — one xtask command regenerates the whole lock after oracle passes.

**D-07:** Unstable-source profile stays separate per Phase 14 D-02. Regeneration covers the four standard profiles (base, with-f12, with-4c1e, with-f12+with-4c1e); unstable-source handled by nightly CI only.

**D-08:** Regeneration happens AFTER oracle parity is confirmed, not before (per ROADMAP SC3).

**D-09:** Use GitHub Actions matrix strategy over the four profiles. Each profile runs as a parallel job. All must pass for the gate to succeed.

**D-10:** The manifest-audit CI gate validates both: (1) no lock drift AND (2) every `stability: stable` entry has `oracle_covered=true`. A single uncovered stable entry fails the gate.

### Claude's Discretion

- Internal ordering of oracle audit across families/profiles
- Whether to run tolerance audit as a standalone xtask or integrate into existing parity commands
- How to structure the oracle_covered write-back into the manifest lock (post-run xtask update vs inline during comparison)
- Exact matrix job naming and artifact handling in CI workflow

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ORAC-01 | Oracle tolerance unified to atol=1e-12 for every family with no per-family exceptions | `tolerance_for_family()` already uses `UNIFIED_ATOL=1e-12` for all matched families; gap is the `other => bail!` arm. Refactor to catch-all removes the blocking error for new families. |
| ORAC-02 | Four-profile manifest lock regenerated covering all implemented APIs | Lock regeneration is driven by `build.rs` reading `compiled_manifest.lock.json`. The oracle_covered write-back step must update that JSON before regenerating to bake the flags into the compiled manifest. |
| ORAC-03 | CI oracle-parity gate passes all four profiles at atol=1e-12 under --features cpu with mismatch_count==0; every stability:Stable manifest entry has oracle_covered:true | Two CI changes required: (1) matrix strategy over four profiles for oracle_parity_gate job; (2) add oracle_covered completeness check to manifest-audit job. |
| ORAC-04 | Existing base families (1e, 2e, 2c2e, 3c1e, 3c2e) pass oracle at tightened atol=1e-12 | These families are already exercised by oracle-compare. The gate tightening is a test of current kernel correctness, not new implementation work. |
</phase_requirements>

---

## Summary

Phase 15 is primarily an infrastructure and verification closure phase — no new kernel families are implemented. The work falls into four distinct tracks: (1) refactor `tolerance_for_family()` to eliminate the catch-all bail, (2) add an xtask oracle_covered write-back command that updates `compiled_manifest.lock.json` after oracle passes, (3) update the manifest-audit xtask and CI gate to validate `oracle_covered` completeness, and (4) switch the CI `oracle_parity_gate` job to a GitHub Actions matrix strategy.

The `compare.rs` already has `UNIFIED_ATOL = 1e-12` and `UNIFIED_RTOL = 1e-12` as module-level constants. The tolerance constants are already unified. The gap is entirely in coverage tracking and CI gate shape. All 22 stable operator symbols and 76 stable helper/transform/optimizer/legacy symbols currently have `oracle_covered` either absent (None) or `false`. The 10 F12/STG/YP optional symbols were manually set to `true` in Phase 13. Phase 15 automates this for all remaining stable entries.

The `PHASE4_ORACLE_FAMILIES` constant hardcodes `&["1e", "2e", "2c2e", "3c1e", "3c2e", "4c1e"]` and is referenced in both `fixtures.rs` and `manifest_audit.rs`. Replacing it with manifest-driven derivation eliminates a maintenance hazard but requires coordinated changes in both files.

**Primary recommendation:** Write-back oracle_covered into the lock via a new `xtask oracle-covered-update` sub-command that reads per-profile parity JSON artifacts and stamps `oracle_covered=true` for every symbol that passed. Run it after `oracle-compare` succeeds. Then regenerate the manifest, verify with `manifest-audit --check-lock --require-coverage`, and commit.

---

## Current State Inventory (verified from source)

### compiled_manifest.lock.json (130 entries total)

| Stability | oracle_covered=true | oracle_covered=false | oracle_covered=absent |
|-----------|---------------------|---------------------|----------------------|
| stable (98) | 0 | 76 (helper/xform/opt/legacy) | 22 (operator symbols) |
| optional (12) | 10 (F12/STG/YP) | 2 (int4c1e_cart, int4c1e_sph) | 0 |
| unstable_source (20) | 0 | 20 | 0 |

The 22 stable operators with absent `oracle_covered` are the nine 1e symbols, three 2e symbols, three 2c2e symbols, four 3c1e symbols, and three 3c2e symbols. All of these have `raw_api_for_symbol()` mappings in `compare.rs` and `eval_legacy_symbol()` dispatch entries. They are already exercised when `oracle-compare` runs.

The 76 stable helper/transform/optimizer/legacy entries have `oracle_covered=false`. They are validated by `verify_helper_surface_coverage()` which is called at the start of `build_profile_parity_report()`. This means they ARE oracle-exercised at runtime — the flag just hasn't been written back.

The 2 optional 4c1e entries (`int4c1e_cart`, `int4c1e_sph`) have `oracle_covered=false`. These are in scope for the four-profile run (with-4c1e profile). They have raw API mappings (`RawApiId::INT4C1E_CART`, `RawApiId::INT4C1E_SPH`) and eval_legacy_symbol dispatch. These need to pass at atol=1e-12 and have their flag set.

### tolerance_for_family() current shape

```rust
// crates/cintx-oracle/src/compare.rs line 127
pub fn tolerance_for_family(family: &str) -> Result<FamilyTolerance> {
    let static_family: &'static str = match family {
        "1e" => "1e",
        "2e" => "2e",
        "unstable::source::2e" => "unstable::source::2e",
        "2c2e" => "2c2e",
        "3c2e" => "3c2e",
        "3c1e" => "3c1e",
        "4c1e" => "4c1e",
        other => bail!("missing family tolerance for `{other}`"),  // BLOCKS new families
    };
    Ok(FamilyTolerance { family: static_family, atol: UNIFIED_ATOL, rtol: UNIFIED_RTOL, zero_threshold: ZERO_THRESHOLD })
}
```

All matched arms return the same `FamilyTolerance` values. The match arms exist to provide a valid `'static str` for the `family` field. This can be replaced with a catch-all that uses a leaked or pre-known string. The missing unstable::source::* families (breit, grids, origi, origk, ssc) will trigger the bail today if they ever enter the parity loop — but for the 4 standard profiles, `stability_is_included()` in `fixtures.rs` filters them out before they reach `tolerance_for_family()`. The bail matters for the nightly unstable-source run.

### PHASE4_ORACLE_FAMILIES usage

```rust
// crates/cintx-oracle/src/fixtures.rs line 156
pub const PHASE4_ORACLE_FAMILIES: &[&str] = &["1e", "2e", "2c2e", "3c1e", "3c2e", "4c1e"];

fn is_phase4_oracle_family(family: &str) -> bool {
    PHASE4_ORACLE_FAMILIES.contains(&family) || family.starts_with("unstable::source::")
}
```

This constant is used in `fixtures.rs` (oracle eligibility) and `manifest_audit.rs` (audit scope). D-05 says replace with manifest-driven derivation. The replacement: any family with at least one entry where `stability` is `"stable"` or `"optional"` in the lock is oracle-eligible. The `starts_with("unstable::source::")` guard can remain for the opt-in path.

### CI oracle_parity_gate current shape

The current `oracle_parity_gate` job in `compat-governance-pr.yml` runs a single job:

```yaml
- name: Run oracle parity gate
  run: |
    CINTX_BACKEND=cpu cargo run --manifest-path xtask/Cargo.toml -- oracle-compare \
      --profiles "${CINTX_REQUIRED_PROFILES}" --include-unstable-source false
```

Where `CINTX_REQUIRED_PROFILES: base,with-f12,with-4c1e,with-f12+with-4c1e`.

This runs all four profiles sequentially in one job. D-09 says switch to matrix strategy for parallel execution. The matrix strategy splits each profile into an independent job — all four must pass for the gate to succeed. The xtask `oracle-compare` command already handles single-profile execution when passed `--profiles base` etc.

### manifest-audit current oracle_covered gap

`manifest_audit.rs` currently does NOT validate `oracle_covered`. The `run_manifest_audit()` function checks symbol drift and profile scope, but has no code touching `oracle_covered`. D-10 requires adding: after confirming all four profiles pass, verify every entry with `stability == "stable"` has `oracle_covered == true`.

---

## Architecture Patterns

### Pattern 1: oracle_covered write-back (new xtask sub-command)

**What:** After `oracle-compare` runs and writes per-profile JSON parity reports, a second xtask command reads those reports and stamps `oracle_covered=true` in `compiled_manifest.lock.json` for every symbol that passed across all relevant profiles.

**When to use:** Run explicitly after `oracle-compare` succeeds, before manifest lock regeneration.

**Approach — post-run xtask update:**

```rust
// xtask/src/oracle_covered_update.rs (new file)
pub fn run_oracle_covered_update(profiles: &[String]) -> Result<()> {
    // 1. Load compiled_manifest.lock.json as serde_json::Value
    let lock_path = Path::new("crates/cintx-ops/generated/compiled_manifest.lock.json");
    let mut lock: Value = serde_json::from_str(&fs::read_to_string(lock_path)?)?;

    // 2. Collect symbols that passed in every profile they appear in
    //    by reading the per-profile parity JSON artifacts
    let mut covered_symbols: BTreeSet<String> = BTreeSet::new();
    for profile in profiles {
        let artifact = load_parity_artifact_for_profile(profile)?;
        for fixture in artifact["fixtures"].as_array().unwrap_or(&[]) {
            let within = fixture["raw_vs_upstream"]["within_tolerance"].as_bool().unwrap_or(false);
            if within {
                covered_symbols.insert(fixture["symbol"].as_str().unwrap_or("").to_owned());
            }
        }
    }

    // 3. Stamp oracle_covered=true for covered operator symbols
    for entry in lock["entries"].as_array_mut().unwrap() {
        let sym = entry["id"]["symbol"].as_str().unwrap_or("");
        if covered_symbols.contains(sym) {
            entry["oracle_covered"] = json!(true);
        }
    }

    // 4. For helper/transform entries: stamp based on helper_legacy_parity passing
    //    (verify_helper_surface_coverage already confirms these)
    // 5. Write back
    fs::write(lock_path, serde_json::to_vec_pretty(&lock)?)?;
    Ok(())
}
```

**Alternative — inline during comparison:** `build_profile_parity_report()` could return covered symbol names and `run_oracle_compare()` could update the lock directly. This is tighter coupling but fewer moving parts. The CONTEXT leaves this to discretion.

**Recommended:** Post-run xtask sub-command (`oracle-covered-update`). Keeps compare and write-back separable — oracle-compare can still fail-close without touching the lock.

### Pattern 2: CI matrix strategy for oracle_parity_gate

**What:** Replace the single `oracle_parity_gate` job with a matrix over four profiles.

```yaml
oracle_parity_gate:
    name: oracle_parity_gate (${{ matrix.profile }})
    runs-on: ubuntu-latest
    strategy:
        fail-fast: false
        matrix:
            profile: [base, with-f12, with-4c1e, with-f12+with-4c1e]
    steps:
        # ... checkout, toolchain, cache ...
        - name: Run oracle parity gate for profile ${{ matrix.profile }}
          run: |
              CINTX_BACKEND=cpu cargo run --manifest-path xtask/Cargo.toml -- oracle-compare \
                --profiles "${{ matrix.profile }}" --include-unstable-source false
```

`fail-fast: false` ensures all four profiles report independently rather than aborting on first failure.

The xtask `validate_required_profile_scope()` currently requires all four profiles when running the standard set. A single-profile matrix call needs the validation to accept a single profile. Options: (a) relax the validation to accept subsets, (b) add a `--single-profile` flag to bypass the "must be all four" check, (c) accept that the matrix always passes all four profiles in parallel and each just validates its own.

**Simplest:** Accept a single profile string to `oracle-compare` and skip the "must have all four" check when only one profile is passed. The CI env `CINTX_REQUIRED_PROFILES` can stay as a comma-separated validation reference, but the matrix job passes its own profile.

### Pattern 3: manifest-audit oracle_covered completeness check

**What:** Add a check in `run_manifest_audit()` (or as a new flag `--require-coverage`) that reads every lock entry with `stability == "stable"` and fails if any has `oracle_covered != true`.

```rust
// In manifest_audit.rs
fn check_oracle_coverage(lock_root: &Value) -> Vec<String> {
    let entries = lock_root["entries"].as_array().unwrap_or(&[]);
    let mut uncovered = Vec::new();
    for entry in entries {
        if entry.get("stability").and_then(Value::as_str) == Some("stable") {
            let covered = entry.get("oracle_covered").and_then(Value::as_bool).unwrap_or(false);
            if !covered {
                let sym = entry["id"]["symbol"].as_str().unwrap_or("?");
                uncovered.push(sym.to_owned());
            }
        }
    }
    uncovered
}
```

Integrate into `run_manifest_audit()` when `check_lock` is true. The fail condition: `should_fail = check_lock && (has_symbol_drift || has_profile_scope_mismatch || !uncovered_stable.is_empty())`.

### Pattern 4: tolerance_for_family() catch-all refactor

**What:** Replace `other => bail!(...)` with a catch-all that leaks the string or uses a predefined default.

```rust
pub fn tolerance_for_family(family: &str) -> FamilyTolerance {
    // All families use unified tolerance — match arms are documentation only.
    let static_family: &'static str = match family {
        "1e" => "1e",
        "2e" => "2e",
        "2c2e" => "2c2e",
        "3c2e" => "3c2e",
        "3c1e" => "3c1e",
        "4c1e" => "4c1e",
        "unstable::source::2e" => "unstable::source::2e",
        // Catch-all: any new family uses unified tolerance.
        // Leak to satisfy 'static; occurs at most once per unique family string.
        _ => Box::leak(family.to_owned().into_boxed_str()),
    };
    FamilyTolerance { family: static_family, atol: UNIFIED_ATOL, rtol: UNIFIED_RTOL, zero_threshold: ZERO_THRESHOLD }
}
```

**Note:** Change the return type from `Result<FamilyTolerance>` to `FamilyTolerance` — callers currently do `match tolerance_for_family(...)` with a mismatch error path. All call sites in `build_profile_parity_report()` need updating. The `push_mismatch` for `"missing_tolerance"` becomes dead code and should be removed.

**Alternative for 'static:** Since families are a closed set per manifest, use a pre-interned set. But `Box::leak` is fine for a fixed population of ~10 families.

### Pattern 5: PHASE4_ORACLE_FAMILIES manifest-driven replacement

**What:** Replace the hardcoded `PHASE4_ORACLE_FAMILIES` constant with a function that reads oracle-eligible families from the parsed lock.

```rust
// In fixtures.rs — replaces PHASE4_ORACLE_FAMILIES constant
fn manifest_oracle_families() -> BTreeSet<String> {
    let root: Value = serde_json::from_str(COMPILED_MANIFEST_LOCK_JSON).expect("lock parse");
    root["entries"].as_array().unwrap_or(&[]).iter()
        .filter_map(|e| {
            let stab = e.get("stability").and_then(Value::as_str).unwrap_or("");
            if matches!(stab, "stable" | "optional") {
                e.get("id").and_then(|id| id.get("family")).and_then(Value::as_str).map(|s| s.to_owned())
            } else {
                None
            }
        })
        .collect()
}

fn is_oracle_eligible_family(family: &str) -> bool {
    manifest_oracle_families().contains(family) || family.starts_with("unstable::source::")
}
```

**Performance note:** `manifest_oracle_families()` parses the lock JSON on each call. Since it's used in test/xtask paths (not hot paths), lazy_static or once_cell is optional but clean if called frequently.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parallel CI matrix | Custom fan-out bash scripts | GitHub Actions `strategy.matrix` | Native first-class support, proper failure reporting |
| JSON lock update | Custom serializer | `serde_json::Value` mutation + `to_vec_pretty` | Already used throughout xtask; preserves key ordering |
| Tolerance comparison | New tolerance struct | Existing `FamilyTolerance` + `diff_summary()` | Already handles atol/rtol/zero_threshold correctly |
| oracle_covered verification | Separate audit crate | Add to existing `manifest_audit.rs` | Keeps all lock-integrity checks in one place |

---

## Common Pitfalls

### Pitfall 1: Regenerating the lock before oracle passes
**What goes wrong:** The lock is regenerated (via `cargo build` triggering `build.rs`) with stale `oracle_covered=false` flags. CI passes the structure check but the coverage check fails because the flags were written before parity was confirmed.
**Why it happens:** `build.rs` runs on any cargo invocation that touches `cintx-ops`. Simply building the crate after editing the JSON is enough to regenerate.
**How to avoid:** The oracle_covered write-back step must happen BEFORE the cargo build that regenerates the derived Rust. Sequence: run oracle-compare → oracle-covered-update (edits JSON) → cargo build (regenerates .rs from JSON) → commit.
**Warning signs:** `oracle_covered` in the generated `api_manifest.rs` is `false` despite oracle having passed.

### Pitfall 2: validate_required_profile_scope blocks single-profile matrix runs
**What goes wrong:** The matrix job passes `--profiles base` but `validate_required_profile_scope()` requires all four profiles and bails.
**Why it happens:** The function was written to enforce the full four-profile contract. A single-profile matrix call is a legitimate new mode.
**How to avoid:** Relax validation to accept either: (a) all four standard profiles, or (b) any single known profile. Add a `--single-profile` flag or detect that only one profile was passed and allow it.

### Pitfall 3: Helper/transform entries counted in oracle_covered gap but not oracle-compared
**What goes wrong:** The check adds all stable entries to the gap count, including helpers, but the parity report only covers operator symbols — helpers never appear in the per-fixture results.
**Why it happens:** `build_profile_parity_report()` calls `verify_helper_surface_coverage()` as an all-or-nothing check, not per-symbol. The result is a boolean pass/fail, not a per-symbol covered flag.
**How to avoid:** When writing back oracle_covered for helpers: treat a passing `verify_helper_surface_coverage()` call as covering all 17 helper + 7 transform + 7 optimizer helper_kind entries collectively. Stamp all of them at once if the helper gate passes. This matches Phase 11 (HELP-01 through HELP-04 complete) which already validated these.

### Pitfall 4: tolerance_for_family caller sites not updated after signature change
**What goes wrong:** Changing `tolerance_for_family` from `Result<FamilyTolerance>` to `FamilyTolerance` causes compile errors at every call site in `build_profile_parity_report()` that pattern-matches on the Result.
**Why it happens:** The current call pattern: `match tolerance_for_family(&fixture.family) { Ok(value) => value, Err(error) => { push_mismatch... continue; } }`.
**How to avoid:** Update all call sites in the same commit as the signature change. The `push_mismatch` for `"missing_tolerance"` kind becomes dead code to remove.

### Pitfall 5: Lock JSON key ordering changes on write-back
**What goes wrong:** After `serde_json::to_vec_pretty` writes the lock, git diff shows reordering of fields in every entry, causing noise in the manifest-audit diff check.
**Why it happens:** `serde_json::Value` does not preserve insertion order by default; it uses a `Map` which is BTreeMap-backed (alphabetical order) in the default feature set.
**How to avoid:** serde_json with `preserve_order` feature (indexmap-backed Map) OR ensure the existing lock already uses alphabetical key order. Inspect the current lock: the keys are `id`, `profiles`, `compiled_in_profiles`, `stability`, `category`, `arity`, `forms`, `component_rank`, `feature_flag`, `declared_in`, `oracle_covered`, `helper_kind`, `canonical_family` — NOT alphabetical. Adding `preserve_order` to the xtask `serde_json` dependency is the cleanest fix.

### Pitfall 6: 4c1e oracle_covered=false is acceptable vs not
**What goes wrong:** The 4c1e entries have `stability: optional` (not `stability: stable`). D-10 requires `oracle_covered=true` for every `stability: stable` entry — not optional. If someone interprets this as requiring all non-stable entries too, it blocks on 4c1e spinor which returns `UnsupportedApi` unconditionally.
**How to avoid:** Scope the `oracle_covered` completeness check to `stability == "stable"` only. The 4c1e optional entries should have their flag updated when oracle confirms cart/sph pass (they have raw API mappings), but spinor stays `false`/absent legitimately.

---

## Code Examples

### tolerance_for_family catch-all (verified pattern from compare.rs)
```rust
// Source: crates/cintx-oracle/src/compare.rs
// Change: remove Result wrapper, add catch-all match arm
pub fn tolerance_for_family(family: &str) -> FamilyTolerance {
    let static_family: &'static str = match family {
        "1e" => "1e",
        "2e" => "2e",
        "2c2e" => "2c2e",
        "3c2e" => "3c2e",
        "3c1e" => "3c1e",
        "4c1e" => "4c1e",
        "unstable::source::2e" => "unstable::source::2e",
        _ => Box::leak(family.to_owned().into_boxed_str()),
    };
    FamilyTolerance {
        family: static_family,
        atol: UNIFIED_ATOL,
        rtol: UNIFIED_RTOL,
        zero_threshold: ZERO_THRESHOLD,
    }
}
```

### oracle-compare single-profile validation relaxation
```rust
// Source: xtask/src/oracle_update.rs
// Current: validate_required_profile_scope requires all 4 profiles
// Change: allow single-profile call (for CI matrix)
fn validate_profile_scope(profiles: &[String]) -> Result<Vec<String>> {
    if profiles.len() == 1 {
        let p = &profiles[0];
        if ALL_KNOWN_PROFILES.contains(&p.as_str()) {
            return Ok(profiles.to_vec());
        }
        bail!("unknown profile `{p}`");
    }
    // ... existing 4-profile validation for bulk runs ...
}
```

### CI matrix strategy for oracle_parity_gate
```yaml
# Source: .github/workflows/compat-governance-pr.yml
oracle_parity_gate:
    name: oracle_parity_gate (${{ matrix.profile }})
    runs-on: ubuntu-latest
    strategy:
        fail-fast: false
        matrix:
            profile: [base, with-f12, with-4c1e, "with-f12+with-4c1e"]
    steps:
        - uses: actions/checkout@v4
        - name: Resolve pinned Rust channel
          id: rust
          run: |
              python <<'PY'
              import os, tomllib
              from pathlib import Path
              data = tomllib.loads(Path("rust-toolchain.toml").read_text())
              channel = data.get("toolchain", {}).get("channel")
              if not channel:
                  raise SystemExit("failed to resolve channel from rust-toolchain.toml")
              with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                  fh.write(f"channel={channel}\n")
              PY
        - uses: dtolnay/rust-toolchain@master
          with:
              toolchain: ${{ steps.rust.outputs.channel }}
        - uses: Swatinem/rust-cache@v2
        - name: Run oracle parity gate
          run: |
              CINTX_BACKEND=cpu cargo run --manifest-path xtask/Cargo.toml -- \
                oracle-compare --profiles "${{ matrix.profile }}" \
                --include-unstable-source false
```

### manifest-audit oracle_covered completeness check
```rust
// Source: xtask/src/manifest_audit.rs (addition)
fn collect_uncovered_stable_symbols(lock_root: &Value) -> Vec<String> {
    lock_root["entries"].as_array().unwrap_or(&[])
        .iter()
        .filter(|entry| {
            entry.get("stability").and_then(Value::as_str) == Some("stable")
        })
        .filter(|entry| {
            entry.get("oracle_covered").and_then(Value::as_bool) != Some(true)
        })
        .filter_map(|entry| {
            entry.get("id").and_then(|id| id.get("symbol")).and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-family tolerance constants (1e-11, 1e-9, etc.) | Single `UNIFIED_ATOL = 1e-12` | Phase 10/13 | Tighter, uniform gate |
| Hardcoded family allow-list | Manifest-driven eligibility (D-05) | Phase 15 (target) | No maintenance hazard on new families |
| Manual oracle_covered updates | Post-oracle xtask write-back (D-01) | Phase 15 (target) | Automated, objective, audit-verified |
| Sequential four-profile CI job | Matrix parallel jobs (D-09) | Phase 15 (target) | Faster CI, independent failure reporting |

---

## Environment Availability

Phase 15 is code/CI changes only. The oracle runs under `--features cpu` (CubeCL CPU backend), which does not require GPU hardware and is available on standard `ubuntu-latest` runners.

| Dependency | Required By | Available | Notes |
|------------|------------|-----------|-------|
| `cargo --features cpu` | oracle-compare, all CI gates | Yes | cpu feature confirmed working in Phases 10-14 |
| `CINTX_ORACLE_BUILD_VENDOR=1` | Vendored libcint numeric oracle | Optional | Not required for `oracle-compare` xtask; only for `has_vendor_libcint` cfg path |
| `/mnt/data` | Artifact persistence | CI only | Falls back to `CINTX_ARTIFACT_DIR`/`/tmp/cintx_artifacts` per existing convention |
| GitHub Actions matrix | CI parallel jobs | Yes | `strategy.matrix` is standard GHA feature |

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test + cargo nextest |
| Config file | none (uses cargo test defaults) |
| Quick run command | `cargo test -p cintx-oracle --features cpu -- oracle_parity --test-threads=1` |
| Full suite command | `CINTX_BACKEND=cpu cargo run --manifest-path xtask/Cargo.toml -- oracle-compare --profiles base,with-f12,with-4c1e,with-f12+with-4c1e --include-unstable-source false` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ORAC-01 | `tolerance_for_family()` returns `UNIFIED_ATOL` for all families, no bail | unit | `cargo test -p cintx-oracle --features cpu -- tolerance_for_family` | ❌ Wave 0 |
| ORAC-01 | catch-all arm handles unknown family without panic | unit | `cargo test -p cintx-oracle --features cpu -- tolerance_catchall` | ❌ Wave 0 |
| ORAC-02 | `compiled_manifest.lock.json` has oracle_covered=true for all stable operator symbols after write-back | integration | `cargo run --manifest-path xtask/Cargo.toml -- manifest-audit --profiles base,with-f12,with-4c1e,with-f12+with-4c1e --check-lock` | ✅ (needs oracle_covered check added) |
| ORAC-03 | CI oracle_parity_gate matrix passes with mismatch_count==0 for each profile | smoke | `CINTX_BACKEND=cpu cargo run --manifest-path xtask/Cargo.toml -- oracle-compare --profiles base --include-unstable-source false` | ✅ |
| ORAC-04 | Base families 1e/2e/2c2e/3c1e/3c2e pass oracle at 1e-12 | integration | `CINTX_BACKEND=cpu cargo run --manifest-path xtask/Cargo.toml -- oracle-compare --profiles base --include-unstable-source false` | ✅ |

### Sampling Rate
- **Per task commit:** `cargo test -p cintx-oracle --features cpu -- --test-threads=1`
- **Per wave merge:** `CINTX_BACKEND=cpu cargo run --manifest-path xtask/Cargo.toml -- oracle-compare --profiles base,with-f12,with-4c1e,with-f12+with-4c1e --include-unstable-source false`
- **Phase gate:** Full oracle-compare across all four profiles, manifest-audit with oracle_covered check, zero mismatches

### Wave 0 Gaps
- [ ] Unit tests for `tolerance_for_family()` catch-all behavior — covers ORAC-01
- [ ] Unit test for `manifest_audit::check_oracle_coverage()` — covers ORAC-02/ORAC-03

---

## Open Questions

1. **serde_json key ordering in oracle_covered write-back**
   - What we know: current lock uses non-alphabetical key order; `serde_json` default (BTreeMap) will reorder alphabetically on write
   - What's unclear: whether the manifest-audit diff check will treat key reordering as a structural drift failure
   - Recommendation: add `preserve_order` feature to `serde_json` in xtask's `Cargo.toml` before implementing the write-back, then verify the round-trip produces no diff

2. **Scope of oracle_covered for helper/transform/legacy entries**
   - What we know: CONTEXT D-10 says "every stability: stable entry has oracle_covered: true"; 76 of 98 stable entries are helpers/transforms/optimizers/legacy; these are validated collectively by `verify_helper_surface_coverage()`
   - What's unclear: whether the planner should treat helpers as covered by the existing HELP-01/HELP-04 gates (already complete per REQUIREMENTS.md) or require the lock flags to match
   - Recommendation: stamp all helper/transform/optimizer/legacy stable entries as `oracle_covered=true` in the write-back, gated on `verify_helper_surface_coverage()` passing. This is consistent with HELP-01-04 being complete.

3. **int4c1e_cart / int4c1e_sph oracle at 1e-12**
   - What we know: 4c1e has `oracle_covered=false` currently; `int4c1e_cart` and `int4c1e_sph` have raw API mappings and eval_legacy dispatch; 4C1E-01/04 requirements are marked complete
   - What's unclear: whether these two symbols currently pass at 1e-12 in the existing oracle compare run (they should per 4C1E-04)
   - Recommendation: include them in the write-back if they pass; do not special-case them

---

## Project Constraints (from CLAUDE.md)

- Error handling: public library uses `thiserror` v2; xtask/oracle uses `anyhow` — this phase touches xtask code only, so `anyhow` throughout
- Artifacts to `/mnt/data` with `CINTX_ARTIFACT_DIR` fallback — all new xtask artifact writes must follow this pattern
- Verification: Full API coverage claims must be backed by the compiled manifest lock — oracle_covered in the lock IS the coverage claim; the write-back closes this contract
- `CINTX_BACKEND=cpu` required for oracle CI runs — all new CI jobs must set this env var
- `Cargo.lock` committed, CI uses `cargo --locked` — any new xtask deps must be locked

---

## Sources

### Primary (HIGH confidence)
- `crates/cintx-oracle/src/compare.rs` — UNIFIED_ATOL/RTOL constants at lines 21-22; `tolerance_for_family()` at line 127; `build_profile_parity_report()` at line 990; `raw_api_for_symbol()` at line 206; `eval_legacy_symbol()` at line 249
- `crates/cintx-oracle/src/fixtures.rs` — `PHASE4_ORACLE_FAMILIES` at line 156; `is_phase4_oracle_family()` at line 382; `stability_is_included()` at line 386; `build_profile_representation_matrix()` at line 539
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — 130 entries; 98 stable (0 with oracle_covered=true); 10 optional with oracle_covered=true (F12/YP); 2 optional with oracle_covered=false (4c1e)
- `crates/cintx-ops/build.rs` — reads oracle_covered from lock at line 129; writes it into generated `api_manifest.rs`
- `xtask/src/oracle_update.rs` — `run_oracle_compare()` at line 34; `validate_required_profile_scope()` at line 195
- `xtask/src/manifest_audit.rs` — `run_manifest_audit()` at line 18; no oracle_covered check present (confirmed gap)
- `.github/workflows/compat-governance-pr.yml` — current `oracle_parity_gate` job at line 73 (sequential, not matrix); `manifest_drift_gate` at line 37

### Secondary (MEDIUM confidence)
- `.planning/phases/15-oracle-tolerance-unification-manifest-lock-closure/15-CONTEXT.md` — all locked decisions verified against source code
- `.planning/REQUIREMENTS.md` — ORAC-01 through ORAC-04 requirement definitions

---

## Metadata

**Confidence breakdown:**
- Current state (oracle_covered gap, tolerance_for_family shape, PHASE4_ORACLE_FAMILIES usage): HIGH — verified by reading source and running Python queries against the lock
- Architecture patterns (write-back xtask, matrix CI, manifest-audit extension): HIGH — based on existing code patterns in the same files
- Pitfalls (serde_json key ordering, single-profile validation): MEDIUM — observed from code reading; serde_json behavior is well-documented but not locally tested for round-trip

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable infrastructure; no fast-moving dependencies)
