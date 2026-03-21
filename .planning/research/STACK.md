# Stack Research

**Domain:** Rust crate test-governance policy, CI enforcement, and auditable reporting
**Researched:** 2026-03-21
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Rust toolchain (stable + pinned nightly) | stable + `nightly-YYYY-MM-DD` | Baseline compilation/tests + nightly-only verification (Miri, fuzzing, some coverage modes) | Miri and other verification tooling require nightly; pinning nightly keeps audits reproducible. citeturn1search2 |
| GitHub Actions + rust-toolchain action | `actions-rs/toolchain@v1` | CI orchestration and toolchain provisioning | Widely used Rust CI surface with explicit toolchain control. citeturn5search3 |
| RustSec advisory enforcement | `cargo-audit` 0.22.1 + `cargo-deny` 0.18.9 | Dependency vulnerability + license/policy enforcement | RustSec is the canonical advisory DB; cargo-audit and cargo-deny are standard enforcement tools. citeturn3search9turn5search6turn2search7 |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `proptest` | 1.7.0 | Property-based tests with shrinking | Mandatory baseline for input-space exploration beyond example tests. citeturn4search2 |
| `cargo-mutants` | 26.0.0 | Mutation testing to detect fake/weak tests | Mandatory baseline to guard against tests that don’t fail when code is wrong. citeturn3search2 |
| `cargo-llvm-cov` | 0.6.18 | Coverage reports (line/region/branch) | Baseline visibility signal (not a sufficiency gate). citeturn5search1 |
| `cargo-hack` | 0.6.43 | Feature-matrix testing, MSRV range checks | Mandatory for feature-flag governance and MSRV drift detection. citeturn3search0turn0search2 |
| `trybuild` | 1.0.114 | Compile-fail/UI tests for user-facing diagnostics | Required when crates expose compile-time contracts or macro diagnostics. citeturn5search8 |
| `ui_test` | 0.30.1 | Compiler-output regression harness | Use when you need rustc-style UI test semantics. citeturn1search7 |
| `loom` | 0.7 | Deterministic concurrency permutation testing | Required for atomics/concurrency risk classes. citeturn0search3 |
| `Miri` | nightly component | Undefined-behavior detection for unsafe code | Required whenever unsafe code exists; catches UB in tests. citeturn1search2 |
| `cargo-fuzz` | 0.13.1 | Fuzzing via libFuzzer | Required for parsers/decoders/formatters or hostile inputs. citeturn0search5turn0search8 |
| `Kani` | latest (pin release in CI) | Bounded model checking for critical invariants | Required for high-value invariants where bounded proofs are feasible. citeturn0search7 |
| `cargo-nextest` | 0.9.111 | Parallel test execution + JUnit/JSON reporting | Use to scale test suites and produce auditable machine-readable reports. citeturn2search4 |
| `cargo-auditable` | 0.7.0 | Embed dependency metadata into binaries | Use when you need post-build provenance for audits. citeturn4search0 |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `cargo test` | Baseline unit/integration/regression tests | Baseline gate; not sufficient for assurance alone. |
| `cargo miri test` | Unsafe UB checks | Use nightly; run as separate CI lane. citeturn1search2 |
| `cargo mutants` | Mutation testing | Use incremental PR mode + full mainline/nightly sweeps. citeturn3search2 |
| `cargo hack` | Feature and MSRV matrix | Use `--each-feature` or `--feature-powerset` based on size. citeturn0search2 |
| `cargo llvm-cov` | Coverage artifacts | Produce LCOV/HTML artifacts for audit trail. citeturn5search1 |
| `cargo audit` / `cargo deny` | Supply-chain enforcement | Use CI gates + scheduled audits. citeturn5search6turn2search7 |

## Installation

```bash
# Core (tooling binaries)
cargo install --locked cargo-mutants cargo-llvm-cov cargo-hack cargo-fuzz cargo-nextest cargo-audit cargo-deny cargo-auditable

# Nightly components
rustup toolchain install nightly
rustup +nightly component add miri

# Supporting test crates (dev-dependencies)
cargo add -D proptest trybuild ui_test loom
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `cargo-mutants` | `mutagen` | Use only if you need manual mutation control and can accept higher setup overhead. citeturn3search2turn0search4 |
| `trybuild` | `ui_test` | Use `ui_test` if you need rustc-style UI test semantics; otherwise `trybuild` is simpler for crate-level compile-fail tests. citeturn1search7turn5search8 |
| `cargo-nextest` | `cargo test` only | Use `cargo test` alone for tiny crates with short runtimes; otherwise nextest improves throughput and reporting. citeturn2search4 |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Coverage-only gates | Coverage does not prove correctness or spec conformance | Use mutation tests + property tests + targeted verification. |
| Unpinned nightly toolchains | Non-reproducible audits and flaky CI | Pin nightly by date in `rust-toolchain.toml`. citeturn1search2 |
| Outdated fuzzing wrappers (e.g., `cargo-libafl`) | Marked as outdated and behind `cargo-fuzz` | Use `cargo-fuzz` unless you have a specific LibAFL requirement. citeturn6view0turn0search8 |

## Stack Patterns by Variant

**If the crate contains unsafe code or FFI:**
- Add `Miri` to PR/nightly gates.
- Because it detects UB that `cargo test` cannot. citeturn1search2

**If the crate is concurrent or uses atomics:**
- Add `loom` model tests.
- Because it explores interleavings deterministically. citeturn0search3

**If the crate parses external input or has serialization formats:**
- Add `cargo-fuzz` fuzz targets.
- Because libFuzzer-based fuzzing finds hostile-input bugs. citeturn0search8

**If the crate exposes compile-time constraints or macros:**
- Add `trybuild` or `ui_test`.
- Because compile-fail diagnostics are part of the public contract. citeturn5search8turn1search7

**If the crate has critical invariants:**
- Add `Kani` proofs for bounded verification.
- Because model checking can prove properties that tests may miss. citeturn0search7

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `cargo-llvm-cov` 0.6.18 | Rust nightly (branch coverage optional) | Branch coverage requires nightly. citeturn5search1 |
| `cargo-fuzz` 0.13.1 | Nightly + libFuzzer | Uses libFuzzer via cargo-fuzz. citeturn0search5turn0search8 |
| `Miri` (nightly component) | Pinned nightly toolchain | Install via rustup nightly; supports `cargo miri test`. citeturn1search2 |

## Sources

- https://github.com/rust-lang/miri — UB detection + nightly install details citeturn1search2
- https://github.com/taiki-e/cargo-hack — feature matrix + MSRV range tooling citeturn0search2
- https://lib.rs/crates/cargo-llvm-cov — cargo-llvm-cov release info + branch coverage note citeturn5search1
- https://docs.rs/crate/cargo-hack/0.2.0 — cargo-hack version history citeturn3search0
- https://github.com/sourcefrog/cargo-mutants — cargo-mutants usage + CI integration citeturn3search2
- https://lib.rs/crates/cargo-fuzz — cargo-fuzz version info citeturn0search5
- https://rust-fuzz.github.io/book/cargo-fuzz.html — cargo-fuzz usage + libFuzzer binding citeturn0search8
- https://model-checking.github.io/kani/ — Kani overview + workflow citeturn0search7
- https://docs.rs/crate/cargo-audit/latest — cargo-audit version info + usage citeturn5search6
- https://www.mail-archive.com/package-announce%40lists.fedoraproject.org/msg440085.html — cargo-deny version reference citeturn2search7
- https://lib.rs/crates/proptest — proptest version info citeturn4search2
- https://github.com/tokio-rs/loom — loom quickstart version pin citeturn0search3
- https://www.mail-archive.com/package-announce%40lists.fedoraproject.org/msg441946.html — trybuild version reference citeturn5search8
- https://lib.rs/crates/ui_test — ui_test version info citeturn1search7
- https://github.com/nextest-rs/nextest — cargo-nextest releases citeturn2search4
- https://lib.rs/crates/cargo-auditable — cargo-auditable version info citeturn4search0
- https://github.com/marketplace/actions/rust-toolchain — GitHub Actions rust-toolchain action citeturn5search3
- https://github.com/rustsec/rustsec — RustSec advisory tooling citeturn3search9

---
*Stack research for: Rust crate test-governance policy, CI enforcement, and auditable reporting*
*Researched: 2026-03-21*
