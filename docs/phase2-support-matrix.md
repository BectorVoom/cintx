# Phase 2 CPU Support Matrix

This document defines the exact Phase 2 CPU baseline compatibility envelope and links each supported row to executable evidence.

## Stable-Family Envelope (In Scope)

All rows below are required and covered by automated tests. There are no exclusions inside this matrix.

| Family | Operator | Representations | Evidence Tests | Requirement Links |
|---|---|---|---|---|
| `1e` | `Overlap` | `cart`, `sph`, `spinor` | `tests/phase2_cpu_execution_matrix.rs`, `tests/phase2_safe_raw_equivalence.rs`, `tests/phase2_oracle_tolerance.rs` | `COMP-01`, `SAFE-03`, `RAW-01`, `EXEC-01` |
| `2e` | `ElectronRepulsion` | `cart`, `sph`, `spinor` | `tests/phase2_cpu_execution_matrix.rs`, `tests/phase2_safe_raw_equivalence.rs`, `tests/phase2_oracle_tolerance.rs` | `COMP-01`, `SAFE-03`, `RAW-01`, `EXEC-01` |
| `2c2e` | `ElectronRepulsion` | `cart`, `sph`, `spinor` | `tests/phase2_cpu_execution_matrix.rs`, `tests/phase2_safe_raw_equivalence.rs`, `tests/phase2_oracle_tolerance.rs` | `COMP-01`, `SAFE-03`, `RAW-01`, `EXEC-01` |
| `3c1e` | `Kinetic` | `cart`, `sph`, `spinor` | `tests/phase2_cpu_execution_matrix.rs`, `tests/phase2_safe_raw_equivalence.rs`, `tests/phase2_oracle_tolerance.rs` | `COMP-01`, `SAFE-03`, `RAW-01`, `EXEC-01` |
| `3c2e` | `ElectronRepulsion` | `cart`, `sph`, `spinor` | `tests/phase2_cpu_execution_matrix.rs`, `tests/phase2_safe_raw_equivalence.rs`, `tests/phase2_oracle_tolerance.rs` | `COMP-01`, `SAFE-03`, `RAW-01`, `EXEC-01` |

## Explicit `3c1e` Spinor Handling

- `3c1e + spinor` is a first-class supported row in the stable-family matrix.
- Routing is validated through the dedicated `ThreeCenterOneElectronSpinor` adapter path.
- Oracle and safe/raw equivalence tests include this row as a mandatory execution gate.

## Out-of-Phase Envelope Expectations

Rows outside the matrix above must return typed unsupported errors (`LibcintRsError::UnsupportedApi { api: "cpu.route", .. }`).

| Family | Operator | Representation | Expected Result |
|---|---|---|---|
| `1e` | `Kinetic` | `cart` | Typed unsupported (`cpu.route`) |
| `1e` | `NuclearAttraction` | `spinor` | Typed unsupported (`cpu.route`) |
| `2e` | `Overlap` | `sph` | Typed unsupported (`cpu.route`) |
| `3c2e` | `Kinetic` | `spinor` | Typed unsupported (`cpu.route`) |

## Requirement Trace to Evidence

| Requirement | Evidence |
|---|---|
| `COMP-01` | Stable-family matrix execution + oracle tolerance (`phase2_cpu_execution_matrix`, `phase2_oracle_tolerance`) |
| `RAW-01` | Raw compat query/execute exercised for every supported matrix row (`phase2_cpu_execution_matrix`, `phase2_safe_raw_equivalence`) |
| `RAW-02` | Raw query-then-execute flow validated per row (`phase2_cpu_execution_matrix`, `phase2_safe_raw_equivalence`) |
| `RAW-03` | Typed dims/buffer/no-partial-write failure gates (`phase2_raw_failure_semantics`) |
| `SAFE-03` | Safe evaluate/evaluate_into layout and numeric parity gates (`phase2_safe_raw_equivalence`, `phase2_oracle_tolerance`) |
| `MEM-01` | Memory-limit contract gates (`phase2_memory_contracts`) |
| `MEM-02` | Allocation-failure typed contract gates (`phase2_memory_contracts`, `phase2_raw_failure_semantics`) |
| `EXEC-01` | CPU backend dispatch used for all stable-family rows (`phase2_cpu_execution_matrix`) |

## Phase 3 Governance Ownership

Phase 2 support rows are governed by Phase 3 blocking CI gates before merge and release.

| Governance Scope | Workflow | Key Jobs | Requirements |
|---|---|---|---|
| PR blocking gates | `.github/workflows/compat-governance-pr.yml` | `helper_parity_gate`, `manifest_governance_gate`, `core_regression_gate` | `COMP-02`, `COMP-03`, `COMP-04`, `VERI-01`, `VERI-02` |
| Release blocking gates | `.github/workflows/compat-governance-release.yml` | `helper_parity_release_gate`, `manifest_release_gate`, `oracle_profile_release_gate`, `optimizer_equivalence_release_gate` | `COMP-02`, `COMP-03`, `COMP-04`, `RAW-04`, `VERI-01`, `VERI-02`, `VERI-03` |

Detailed gate policy and command-level traceability are documented in `docs/phase3-governance-gates.md`.
