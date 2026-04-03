# Phase 8: Gaussian Primitive Infrastructure and Boys Function - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 08-gaussian-primitive-infrastructure-and-boys-function
**Areas discussed:** Boys function strategy, Rys quadrature approach, #[cube] function design patterns, Validation and oracle comparison

---

## Boys Function Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Replicate libcint table approach | Upload fmt.c-style gridded Taylor coefficients to device memory. Kernel does table lookup + polynomial evaluation. | ✓ |
| Pure recurrence (no tables) | Compute Fm(x,m) using upward/downward recurrence from a starting value. | |
| Hybrid (table + recurrence fallback) | Table lookup for common domain, switch to asymptotic for large x. | |
| You decide | Claude picks based on accuracy and GPU constraints. | |

**User's choice:** Replicate libcint table approach
**Notes:** User consistently chose faithful replication of libcint's mathematical approach.

---

### Boys: Domain Coverage

| Option | Description | Selected |
|--------|-------------|----------|
| Match libcint exactly | Same grid points, same turn-over thresholds, same polynomial degree. | ✓ |
| Cover practical range only | Focus on x values in 1e/2e integrals up to aug-cc-pVQZ. | |
| Full domain + edge cases | Practical range plus edge case testing. | |

**User's choice:** Match libcint exactly

---

### Boys: Table Upload Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Compile-time constants in #[cube] | Embed Taylor coefficients as const arrays. Zero upload overhead. | ✓ |
| Device buffer upload at init | Upload table data as device buffer during executor init. | |
| You decide | Claude picks based on CubeCL const-array support. | |

**User's choice:** Compile-time constants in #[cube]

---

### Boys: Large-x Branch

| Option | Description | Selected |
|--------|-------------|----------|
| Replicate fmt.c erfc path | Same erfc_like approach from libcint for large-x region. | ✓ |
| Standard asymptotic expansion | Well-known asymptotic formula, simpler in #[cube]. | |
| You decide | Claude picks based on what matches within 1e-12. | |

**User's choice:** Replicate fmt.c erfc path

---

### Boys: Turn-over Threshold

| Option | Description | Selected |
|--------|-------------|----------|
| Match libcint thresholds | Same m-dependent turn-over points from fmt.c. | |
| Optimize for GPU | Allow different thresholds if values within tolerance. | ✓ |
| You decide | Claude determines during research. | |

**User's choice:** Optimize for GPU (allow different thresholds if accuracy maintained)

---

### Boys: Maximum Order m

| Option | Description | Selected |
|--------|-------------|----------|
| Up to m=20 (standard) | Covers s through g functions. | |
| Up to m=30 (extended) | Covers through h/i functions. | |
| Match libcint's compile-time max | Whatever MMAX is defined as in fmt.c. | ✓ |
| You decide | Claude determines based on success criteria. | |

**User's choice:** Match libcint's compile-time max

---

### Boys: Grid Spacing

| Option | Description | Selected |
|--------|-------------|----------|
| Match libcint grid exactly | Same NGRID, same spacing, same polynomial degree. | |
| Allow GPU-tuned grid | Spacing can differ if accuracy maintained. | ✓ |
| You decide | Claude determines based on accuracy/memory trade-offs. | |

**User's choice:** Allow GPU-tuned grid

---

### Boys: Taylor Term Count

| Option | Description | Selected |
|--------|-------------|----------|
| Match libcint term count | Same number of Taylor coefficients per grid interval. | |
| Allow different term count | Can use more or fewer terms if accuracy within 1e-12. | ✓ |
| You decide | Claude balances accuracy vs register pressure. | |

**User's choice:** Allow different term count

---

### Boys: Function Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Single function with if/else | One boys_fm(x, m) with internal branching. | ✓ |
| Separate per-region functions | boys_fm_small and boys_fm_large called from kernel. | |
| You decide | Claude decides based on CubeCL control flow support. | |

**User's choice:** Single function with if/else

---

## Rys Quadrature Approach

### Rys: Coefficient Delivery

| Option | Description | Selected |
|--------|-------------|----------|
| Embed as compile-time constants | Same as Boys: const arrays in #[cube] code. | |
| Device buffer upload | Upload polyfits tables as device buffers. | |
| Match Boys function choice | Same strategy as Boys (compile-time constants). | ✓ |

**User's choice:** Match Boys function choice (compile-time constants)

---

### Rys: Quadrature Degree Coverage

| Option | Description | Selected |
|--------|-------------|----------|
| Match libcint's max degree | Whatever polyfits.c supports. Full coverage. | ✓ |
| Up to degree 6 (practical) | Covers s, p, d, f shell combinations. | |
| You decide | Claude determines based on kernel families in scope. | |

**User's choice:** Match libcint's max degree

---

### Rys: Algorithm Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Polynomial fits only | Replicate polyfits.c approach exclusively. | |
| Polynomial fits + Wheeler fallback | polyfits for common, Wheeler for edge cases. | ✓ |
| You decide | Claude determines based on degree coverage. | |

**User's choice:** Polynomial fits + Wheeler fallback

---

## #[cube] Function Design Patterns

### Math Module Location

| Option | Description | Selected |
|--------|-------------|----------|
| New src/math/ module | Dedicated math/ directory with boys.rs, rys.rs, etc. | ✓ |
| Inside src/kernels/ | Co-located with kernel stubs. | |
| Separate crate (cintx-math) | New workspace crate for math. | |
| You decide | Claude picks based on #[cube] import rules. | |

**User's choice:** New crates/cintx-cubecl/src/math/ module

---

### Compile-time Parameters

| Option | Description | Selected |
|--------|-------------|----------|
| Rust const generics | Use const generics where CubeCL supports them. | |
| Fixed constants (hardcoded) | const MMAX: usize = 20 etc. | ✓ |
| You decide | Claude determines based on CubeCL const-generic support. | |

**User's choice:** Fixed constants (hardcoded)

---

### Obara-Saika Recurrence Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Separate functions | hrr_step() and vrr_step() individually. | ✓ |
| Combined recurrence function | Single os_recurrence() for both. | |
| You decide | Claude designs based on g1e.c/g2e.c structure. | |

**User's choice:** Separate functions

---

### PairData Structure

| Option | Description | Selected |
|--------|-------------|----------|
| #[cube] struct | Typed PairData struct with named fields. | ✓ |
| Raw f64 slices | Flat arrays indexed by convention. | |
| You decide | Claude determines based on CubeCL struct support. | |

**User's choice:** #[cube] struct with named fields

---

## Validation and Oracle Comparison

### Reference Value Generation

| Option | Description | Selected |
|--------|-------------|----------|
| Oracle harness calls libcint C | Use cintx-oracle to call fmt.c, polyfits.c directly. | ✓ |
| Precomputed golden files | Generate once, store as golden files. | |
| Both (oracle generates, golden caches) | Oracle generates, golden caches for speed. | |
| You decide | Claude designs validation approach. | |

**User's choice:** Oracle harness calls libcint C

---

### Test Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Per-function test suites | boys_tests.rs, rys_tests.rs, etc. | |
| Combined math validation | One math_validation.rs for everything. | |
| Both (unit + integration) | Per-function unit tests plus combined integration. | ✓ |

**User's choice:** Both (unit + integration)

---

### Backend for Validation

| Option | Description | Selected |
|--------|-------------|----------|
| CPU-only for math validation | All tests use --features cpu. | |
| CPU primary, wgpu optional | Default CPU, also wgpu if SHADER_F64 available. | ✓ |
| You decide | Claude determines based on CI constraints. | |

**User's choice:** CPU primary, wgpu optional

---

### Tolerance Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Absolute only (1e-12) | Simple |computed - reference| < 1e-12. | ✓ |
| Absolute + relative | atol=1e-12 AND rtol=1e-10. | |
| Match libcint's own tolerance | Whatever libcint's test suite uses. | |

**User's choice:** Absolute only (1e-12)

---

## Claude's Discretion

- Exact const array layout for Taylor coefficients and polyfit tables
- Internal helper functions for erfc computation within #[cube]
- Wheeler algorithm implementation details
- Exact PairData #[cube] struct field names
- Test fixture design for reference value generation

## Deferred Ideas

None
