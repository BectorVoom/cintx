# Resolved Build Error: spinor writer signature drift in project using `cubecl`

## Summary

The CPU-feature build failed while extending the shared 1e component writer for
spinor output. A broad source patch inserted the new kappa parameters into the
neighboring `contract_nuclear` signature and a GIAO call instead of the intended
writer signature. The fix made the signature and its four call sites explicit.

## Impact

`cintx-cubecl` and the Tier-1 vendor parity test could not compile.

## Environment

- Project: cintx
- OS/architecture: Linux x86_64
- Rust: workspace-pinned toolchain
- CubeCL: 0.10.0
- Backend: CubeCL CPU

## Command That Failed

```bash
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test gradient_gap_tier1
```

## Observed Error

```text
cannot find value `kappa_i` in this scope
this function takes 14 arguments but 16 arguments were supplied
```

## Root Cause

Repeated `li, lj` argument sequences made a context-light patch ambiguous. It
modified `contract_nuclear` and `write_giao_complex_staging` while leaving
`write_component_leading_staging` without the parameters its new spinor arm used.

## Resolution

1. Removed the accidental kappa parameters from `contract_nuclear`.
2. Added `kappa_i: i16` and `kappa_j: i16` to
   `write_component_leading_staging`.
3. Passed shell kappas at each component-writer call only.
4. Removed the accidental GIAO call arguments.

## Verification

```bash
CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test gradient_gap_tier1
```

All six cart/sph/spinor Tier-1 1e parity tests passed at `atol=1e-12`.

## Prevention

- Anchor future signature edits on the function name, not repeated argument tails.
- Compile immediately after shared-writer signature changes.
