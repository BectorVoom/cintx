---
phase: quick-260531-aw1
status: human_needed
verified_by: orchestrator (direct vendor-gate execution)
date: 2026-05-31
---

# Quick Task 260531-aw1 — Verification

**Goal:** Force-port all remaining host-side calculation in `crates/cintx-cubecl/src/math/` (`eigh.rs`, `rys_wheeler.rs`) to CubeCL `#[cube]` kernels, without regressing the byte-identical libcint vendor parity locked by Phase 25 FND-02.

**Verdict:** Parity bar fully met; on-device coverage maximal (eigh + Rys nroots 8..12). One narrow band (Rys nroots **6,7** production dispatch) held on the host path under the plan's documented parity-honest escape hatch — flagged for user decision. Hence `human_needed`, not a silent pass.

## Verification method

All gates were run **directly by the orchestrator** in the executor's worktree before merge (authoritative — actual `cargo`, not LSP). The LSP reported E0282/E0283/E0599 errors in `eigh.rs`/cubecl crate files; these were confirmed to be **cubecl `#[cube]` proc-macro false positives** — `cargo build` exits 0.

## must_haves — evidence

| must_have (truth/artifact) | Status | Evidence |
|---|---|---|
| `eigh::cint_diagonalize` runs as a `#[cube]` CPU kernel, bit-identical to host reference | ✅ MET | `eigh.rs` has 15 `#[cube]` kernels; `eigh_device_matches_host`, `eigh_mrrr_tridiag_12x12` green (63/63 in-crate). MAXDIFF=0 over 2000 random tridiagonals (per SUMMARY). |
| FMA fidelity probe proves CubeCL 0.10.0 CpuRuntime fuses `fma` | ✅ MET | `math::rys_wheeler::tests::fma_probe` green. Verdict: **FUSED** (no Dekker-split fallback needed). |
| Rys nroots 8..12 (double-double) run on-device | ✅ MET | `#[cube]` `DdDev` double-double Jacobi/Laguerre/Schmidt kernels using device `fma`. Byte-identical to host dd path. |
| In-crate vendor reference-table regression test (nroots 6..12) | ✅ MET | `rys_roots_host_nroots6to12_matches_libcint` green; gold captured from libcint vendor harness; tolerance split atol=1e-12 (6-7) / max(atol,rtol·\|ref\|) (8-12) mirroring `rys_nroots_sweep_parity.rs:38-42`. |
| Vendor `rys_nroots_sweep` parity preserved at documented split | ✅ MET | `--test rys_nroots_sweep_parity` under `CINTX_ORACLE_BUILD_VENDOR=1 --features cpu`: 3 passed, parity body executed (not skipped). |
| 29/29 family parity preserved byte-identically | ✅ MET | See gate table below — all 7 family binaries green under the vendor gate. |
| No tolerance loosened below baseline; no reference value edited | ✅ MET | Confirmed in diff + SUMMARY; baseline = documented split, not flat 1e-12. |
| Rys nroots 6,7 production dispatch on-device | ⚠️ DEVIATION | Device kernels written + bit-identical in isolation, but production dispatch kept **host** (escape hatch). See deviation below. |
| Diff confined to `crates/cintx-cubecl/src/math/` | ✅ MET | Merge touched only `eigh.rs`, `rys.rs`, `rys_wheeler.rs`. No manifest/RawApiId/capi/FFI changes. |

## Vendor parity gate (orchestrator-run, `CINTX_ORACLE_BUILD_VENDOR=1 --features cpu`)

| Binary | Result |
|---|---|
| `rys_nroots_sweep_parity` | 3/3 ✅ (parity body ran) |
| `center_2c2e_parity` | 2/2 ✅ |
| `center_3c1e_parity` | 2/2 ✅ |
| `deriv34_parity` | 14/14 ✅ |
| `hess1e_ipip_parity` | 8/8 ✅ |
| `hess2e_parity` | 2/2 ✅ (487s) |
| `hess_multicenter_ipip_parity` | 2/2 ✅ |
| `int2c2e_ip_parity` | 4/4 ✅ |
| In-crate `cintx-cubecl --lib math::` | 63/63 ✅ |

Build: `cargo build -p cintx-cubecl --features cpu` exit 0 (29 warnings, all benign dead-code/unused). Post-merge rebuild on `fix/general-contraction-nctr-1e`: exit 0.

## Deviation flagged for user (why `human_needed`)

**Rys nroots 6,7 production dispatch stays host.** Routing the f64 nroots-6,7 `#[cube]` device kernels through the family hot path reproducibly breaks `hess2e_parity` by ~1e-11 at the largest components — even though the kernels are **bit-identical to host in isolation** (`rys_nroots_sweep` + reference table byte-identical). Forced-rebuild bisection traced it to a CubeCL `CpuRuntime` **launch perturbing subsequent host g-tensor accumulation** (an FP-environment side effect), not a numerics error. The device kernels are retained in-module; only the family-critical dispatch routes 6,7 to host. eigh + nroots 8..12 are on-device. No tolerance was loosened.

**User decision:** accept the documented escape hatch as final, OR open a follow-up to root-cause/eliminate the CubeCL CpuRuntime FP-environment side effect (or batch/quarantine the launches) so the 6,7 kernels can also join the family-critical path.

## Conclusion

The sacred bar — byte-identical libcint 6.1.3 vendor parity — is **fully preserved (29/29)**. The force-port is substantially achieved (eigh + all of nroots 8..12 on-device) with one narrow, well-evidenced, parity-honest host carve-out for nroots 6,7. Recommend accepting as complete-with-deviation or scheduling the FP-environment follow-up.
