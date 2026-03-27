# Phase 3: Safe Surface, C ABI Shim & Optional Families - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `03-CONTEXT.md`; this log preserves alternatives considered.

**Date:** 2026-03-28
**Phase:** 03-safe-surface-c-abi-shim-optional-families
**Areas discussed:** Safe API, C ABI, Optional Families, Unstable Source API

---

## Safe API

### Q1: Facade entrypoint style
| Option | Description | Selected |
|--------|-------------|----------|
| Typed session object | Expose `query_workspace`/`evaluate` on a typed request/session so validated state carries across calls. | ✓ |
| Free functions only | Expose standalone functions and pass all arguments each time. | |
| Builder-only facade | Require builder-constructed objects before both calls. | |

**User's choice:** Typed session object  
**Notes:** Keep the already-decided split while preventing evaluate-time contract drift.

### Q2: Safe output ownership
| Option | Description | Selected |
|--------|-------------|----------|
| Owned typed output | Return an owned result type; avoid caller-managed output buffers in the safe API. | ✓ |
| Caller buffer only | Require caller-provided mutable buffers for output writes. | |
| Both modes | Support owned output and explicit caller-buffer path. | |

**User's choice:** Owned typed output  
**Notes:** Raw buffer contracts stay in compat/C ABI layers.

### Q3: `query_workspace` result shape
| Option | Description | Selected |
|--------|-------------|----------|
| Structured query token | Return bytes/chunk metadata plus an execution token that `evaluate` must consume. | ✓ |
| Byte count only | Return only required workspace bytes, no structured planning metadata. | |
| Human-oriented summary | Return mostly descriptive planning info and leave execution validation implicit. | |

**User's choice:** Structured query token  
**Notes:** Supports explicit, verifiable query/evaluate pairing.

### Q4: Safe error surface
| Option | Description | Selected |
|--------|-------------|----------|
| Typed public enum | Expose a stable facade error enum preserving categories like `UnsupportedApi` and memory/layout failures. | ✓ |
| Re-export core runtime error directly | Expose `cintxRsError` unchanged as safe API error type. | |
| String-first errors | Flatten into string/status-style errors. | |

**User's choice:** Typed public enum  
**Notes:** Keep public facade contract stable while retaining error semantics.

---

## C ABI

### Q1: Status code semantics
| Option | Description | Selected |
|--------|-------------|----------|
| `0` success, nonzero typed failures | Stable integer codes per error category; failures include TLS last-error details. | ✓ |
| `0/1` only | Boolean-like success/fail only. | |
| Mixed warnings/status bands | Multi-band success/warn/fail scheme. | |

**User's choice:** `0` success, nonzero typed failures  
**Notes:** Chosen for deterministic interop and diagnosability.

### Q2: Last-error retrieval model
| Option | Description | Selected |
|--------|-------------|----------|
| TLS copy-out API | Thread-local error report with explicit copy/len functions; caller owns buffers. | ✓ |
| Internal string pointer | Return pointer to internal string storage. | |
| Global singleton error | One process-global error slot. | |

**User's choice:** TLS copy-out API  
**Notes:** Avoids shared mutable global state and lifetime ambiguity.

### Q3: C shim surface shape
| Option | Description | Selected |
|--------|-------------|----------|
| Thin compat-style wrappers | Mirror raw compat patterns and add status + last-error behavior. | ✓ |
| New opaque-handle API | Introduce new C object model in this phase. | |
| Both at once | Deliver wrappers and new handle model together. | |

**User's choice:** Thin compat-style wrappers  
**Notes:** Prioritizes migration compatibility and Phase 3 scope control.

### Q4: Failure write behavior
| Option | Description | Selected |
|--------|-------------|----------|
| Fail-closed, no partial writes | Do not expose partially written outputs on failure. | ✓ |
| Best-effort partial writes | Allow partially written outputs when errors occur. | |
| Caller-selectable mode | Let caller choose fail-closed vs partial-write mode. | |

**User's choice:** Fail-closed, no partial writes  
**Notes:** Aligns with existing runtime/compat safety contract.

---

## Optional Families

### Q1: Gate style
| Option | Description | Selected |
|--------|-------------|----------|
| Compile-time + runtime gates | Feature flags control visibility; runtime enforces envelope/representation validity. | ✓ |
| Runtime-only gating | Keep APIs visible and reject only at runtime. | |
| Compile-time only gating | Rely on compile-time exposure without runtime envelope checks. | |

**User's choice:** Compile-time + runtime gates  
**Notes:** Keeps contract explicit for both builders and runtime callers.

### Q2: `with-f12` out-of-envelope behavior
| Option | Description | Selected |
|--------|-------------|----------|
| Explicit `UnsupportedApi` with envelope reason | Allow validated sph envelope only; reject other combos with specific reason. | ✓ |
| Silent fallback | Auto-convert unsupported requests to nearest supported representation. | |
| Generic unsupported | Reject without envelope-specific detail. | |

**User's choice:** Explicit `UnsupportedApi` with envelope reason  
**Notes:** Needed for OPT-01 auditable behavior.

### Q3: `with-4c1e` bug-envelope behavior
| Option | Description | Selected |
|--------|-------------|----------|
| Strict rejection | Accept only validated envelope inputs; reject all others explicitly. | ✓ |
| Best-effort execution | Attempt execution outside validated envelope with warnings. | |
| Disable all 4c1e paths | Keep 4c1e blocked despite feature presence. | |

**User's choice:** Strict rejection  
**Notes:** Needed for OPT-02 conformance and reproducible diagnostics.

### Q4: Support matrix source of truth
| Option | Description | Selected |
|--------|-------------|----------|
| Manifest/resolver authority | Generated manifest profiles and resolver metadata are canonical. | ✓ |
| Manual allowlists in facade/capi | Hand-maintained symbol lists per layer. | |
| Hybrid manual + manifest | Manifest with handwritten overrides. | |

**User's choice:** Manifest/resolver authority  
**Notes:** Avoids divergence across layers.

---

## Unstable Source API

### Q1: Namespace boundary
| Option | Description | Selected |
|--------|-------------|----------|
| Explicit unstable namespace | Source-only APIs live under clearly unstable modules/types. | ✓ |
| Mix into stable modules | Place unstable APIs directly in normal surfaces when feature enabled. | |
| Compat-only exposure | Expose unstable APIs only through raw compat. | |

**User's choice:** Explicit unstable namespace  
**Notes:** Preserves stable GA surface.

### Q2: Unstable C ABI scope
| Option | Description | Selected |
|--------|-------------|----------|
| Keep C ABI stable-only | Do not export unstable source-only C symbols in Phase 3. | ✓ |
| Same-feature C exports | Export unstable C symbols under `unstable-source-api`. | |
| Separate C feature | Export unstable C symbols only via an additional C-specific feature. | |

**User's choice:** Keep C ABI stable-only  
**Notes:** Defers extra C ABI surface expansion beyond current phase.

### Q3: Promotion guard
| Option | Description | Selected |
|--------|-------------|----------|
| Manifest+oracle+release evidence | Promote only with repeatable gate evidence and explicit maintainer approval. | ✓ |
| Maintainer-only approval | Promote with manual judgment only. | |
| Time-based promotion | Promote automatically after a time window. | |

**User's choice:** Manifest+oracle+release evidence  
**Notes:** Keeps promotion auditable and compatibility-safe.

### Q4: Feature-disabled behavior
| Option | Description | Selected |
|--------|-------------|----------|
| Not compiled + explicit `UnsupportedApi` | Symbols absent in stable builds; indirect requests fail explicitly. | ✓ |
| Soft fallback to stable API | Auto-map unstable requests to stable alternatives. | |
| Generic internal error | Return non-specific failures. | |

**User's choice:** Not compiled + explicit `UnsupportedApi`  
**Notes:** Makes unsupported behavior deterministic and testable.

---

## the agent's Discretion

- Exact type/module naming in `cintx-rs` and `cintx-capi`.
- Concrete C status code integer mapping table.
- Last-error report payload formatting and helper names.

## Deferred Ideas

None.
