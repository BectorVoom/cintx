# Route Coverage Manifest Specification for libcint Rust Reimplementation

- Document version: 0.1
- Purpose: Define the internal manifest that governs production execution routing and prevents proliferation of ad-hoc bypasses
- Scope: Internal runtime routing only; this file does **not** replace `compiled_manifest.lock.json`

## 1. Positioning

`compiled_manifest.lock.json` remains the source of truth for the supported public ABI/API denominator.
`route_coverage_manifest.lock.json` is a second lock file used only to govern production execution routing inside the Rust implementation.

This separation is intentional:

- Public compatibility inventory answers **what must exist**
- Route coverage inventory answers **how supported combinations are allowed to execute**

## 2. Required Invariants

1. Every policy-supported `stable` family/representation/profile row has exactly one production route.
2. Every policy-supported `optional` row has exactly one production route when its feature is enabled.
3. Every unsupported-by-policy row is explicit as `status = "unsupported_policy"`.
4. Safe API, raw compat API, and C ABI shim all resolve through the same route ID.
5. No production route may exist only as an ad-hoc family-specific bypass outside the shared resolver.

## 3. Schema

| field | type | meaning |
|---|---|---|
| `route_id` | string | Unique production route key |
| `canonical_family` | string | Family name normalized to the compiled manifest |
| `representation` | enum | `cart`, `sph`, `spinor` |
| `surface_group` | enum | `safe`, `raw`, `capi`, `all` |
| `feature_flag` | string | `none`, `with-f12`, `with-4c1e`, `unstable-source-api`, etc. |
| `stability` | enum | `stable`, `optional`, `unstable_source` |
| `support_predicate` | string | `always`, `Validated4C1E`, `sph_only_f12`, etc. |
| `route_kind` | enum | `direct_kernel`, `transform_from_cart`, `composed_workaround`, `unsupported_policy` |
| `backend_set` | array[string] | Allowed backends for this route |
| `entry_kernel` | string | Backend kernel or planner node that owns execution |
| `transform_chain` | array[string] | Pre/post transforms required by the route |
| `writer_contract` | string | Flat layout / logical tensor writer contract |
| `optimizer_mode` | enum | `supported`, `ignored_but_invariant`, `not_applicable` |
| `parity_gate` | string | Mandatory parity/oracle suite for the route |
| `status` | enum | `implemented`, `planned`, `unsupported_policy` |
| `notes` | string | Optional human-readable remarks |

## 4. Example

```json
{
  "route_id": "int1e_ovlp.cart.cpu.direct.v1",
  "canonical_family": "int1e_ovlp",
  "representation": "cart",
  "surface_group": "all",
  "feature_flag": "none",
  "stability": "stable",
  "support_predicate": "always",
  "route_kind": "direct_kernel",
  "backend_set": ["cpu"],
  "entry_kernel": "cpu::one_e::overlap_cartesian",
  "transform_chain": [],
  "writer_contract": "libcint_flat_col_major_1e",
  "optimizer_mode": "not_applicable",
  "parity_gate": "tests/one_e_overlap_cartesian_wrapper_parity.rs",
  "status": "implemented",
  "notes": "Shared by safe/raw/capi through runtime::route_resolver"
}
```

## 5. Generation Procedure

1. Read `compiled_manifest.lock.json`
2. Expand by representation, feature profile, and policy predicate
3. Emit one row for each supported route or explicit `unsupported_policy` row
4. Fail generation if a stable supported row is missing or duplicated
5. Write `crates/libcint-ops/generated/route_coverage_manifest.lock.json`

## 6. CI / Release-Gate Usage

- `route-audit`: validates uniqueness, completeness, and policy coverage
- `route-equivalence`: proves safe/raw/capi resolve to the same `route_id`
- `bypass-lint`: fails if facade, compat, or executor entry layers add family-specific direct execution that is not declared by `entry_kernel`
- release gate: route lock diff must be zero unless explicitly approved in the same PR as the implementation/parity update

## 7. Non-Goals

- It does not define public API coverage
- It does not replace oracle parity suites
- It does not permit calling libcint in the production execution path
