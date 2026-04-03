# Phase 9: 1e Real Kernel and Cart-to-Sph Transform - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 09-1e-real-kernel-and-cart-to-sph-transform
**Areas discussed:** 1e kernel structure, Cart-to-sph placement, Operator dispatch, Validation scope

---

## 1e Kernel Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Shared G-fill + operator switch | Single launch_one_electron dispatches to shared G-tensor fill (VRR/HRR), then per-operator post-processing. Matches libcint g1e.c. | ✓ |
| Separate kernel functions | Three independent functions: kernel_ovlp(), kernel_kin(), kernel_nuc(). Simpler per-function but duplicates VRR/HRR setup. | |
| Trait-based operator abstraction | Operator trait with apply() method. Generic kernel code calls operator.apply(). More extensible but adds abstraction overhead. | |

**User's choice:** Shared G-fill + operator switch (Recommended)
**Notes:** Matches libcint's architecture. Overlap/kinetic/nuclear share the G-tensor fill code.

---

## Cart-to-Sph Placement

| Option | Description | Selected |
|--------|-------------|----------|
| Host-side post-processing | Keep c2s as host-side Rust code. Kernel writes cartesian, host applies Condon-Shortley matrix after client.read(). Simpler to debug and validate. | ✓ |
| GPU-side #[cube] function | Implement c2s as a #[cube] kernel launched after integral kernel. Transform on-device before client.read(). | |
| Dual path | Host-side for CPU backend, #[cube] for GPU. Maximum performance but doubles implementation. | |

**User's choice:** Host-side post-processing (Recommended)
**Notes:** GPU-side c2s deferred as future optimization. Host-side matches Phase 8 pattern and is easier to validate.

---

## Operator Dispatch (Nuclear Attraction)

| Option | Description | Selected |
|--------|-------------|----------|
| Loop over atoms in kernel | Nuclear attraction sums over all atom centers C inside the kernel. Boys F_m(t), VRR with PC displacement, accumulate Z_c * result. Matches libcint. | ✓ |
| Separate kernel per atom | One kernel invocation per atom center, sum on host. Simpler per-invocation but N_atom launches. | |
| Pre-sum on host | Compute atom contributions on host CPU. Misses GPU compute point. | |

**User's choice:** Loop over atoms in kernel (Recommended)
**Notes:** Atom coordinates and charges passed as input arrays to kernel function. Uses boys_gamma_inc() from Phase 8.

---

## Validation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full l=0..4 coefficient validation | Validate Condon-Shortley coefficients for s,p,d,f,g via dedicated unit tests. H2O STO-3G for end-to-end. | ✓ |
| Only test what H2O exercises | Validate c2s only through H2O STO-3G (s+p). Higher l deferred to Phase 10. | |
| l=0..4 + higher-l stress test | Full coefficient validation plus stress tests with cc-pVTZ/cc-pVQZ. More thorough but heavier. | |

**User's choice:** Full l=0..4 coefficient validation (Recommended)
**Notes:** Unit tests prove coefficient tables correct for all angular momenta. End-to-end test exercises s+p via H2O STO-3G.

---

## Claude's Discretion

- G-tensor array sizing and indexing strategy
- GTO contraction loop structure
- Operator ID extraction from SpecializationKey/ExecutionPlan
- Host-side c2s buffer management approach
- Test fixture design for c2s coefficient validation

## Deferred Ideas

- GPU-side #[cube] cart-to-sph transform -- future optimization
- Higher angular momentum end-to-end tests (cc-pVTZ/cc-pVQZ) -- Phase 10
- Workgroup sizing for kernel launch -- post-v1.1
