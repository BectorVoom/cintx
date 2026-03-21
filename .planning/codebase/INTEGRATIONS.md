# External Integrations

**Analysis Date:** 2026-03-21

## APIs & External Services

**Native Numerical Backend:**
- Vendored `libcint` C source is compiled and linked as the execution backend.
  - SDK/Client: build pipeline in `build.rs` and FFI symbol mapping in `src/runtime/backend/cpu/ffi.rs`.
  - Auth: Not applicable (local source compilation/linking only).

**Rust Wrapper Oracle (Test-Time Integration):**
- `libcint` Rust crate is used for wrapper parity/oracle comparisons.
  - SDK/Client: dependency declaration in `Cargo.toml`; usage in `tests/common/oracle_runner.rs`, `tests/two_e_wrapper_parity.rs`, and other `tests/*_wrapper_parity.rs` files.
  - Auth: Not applicable.

**External Network Services:**
- Not detected in `Cargo.toml`, `src/`, or `tests/` (no HTTP/DB client crates or outbound API client code observed).

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only.
- Governance locks are read from `compiled_manifest.lock.json` and `src/runtime/backend/cpu/route_coverage_manifest.lock.json`.
- `manifest_audit` writes lock updates to `compiled_manifest.lock.json` in `src/bin/manifest_audit.rs`.

**Caching:**
- Runtime distributed cache service: None detected.
- CI build cache uses `Swatinem/rust-cache@v2` in `.github/workflows/compat-governance-pr.yml` and `.github/workflows/compat-governance-release.yml`.

## Authentication & Identity

**Auth Provider:**
- Custom/External auth provider: Not detected.
  - Implementation: Not applicable for current crate/library execution path in `src/lib.rs` and `src/runtime/`.

## Monitoring & Observability

**Error Tracking:**
- External error tracking service: None detected.

**Logs:**
- In-process structured logging via `tracing` macros in `src/diagnostics/report.rs`.
- CLI failure output is emitted to stderr in `src/bin/manifest_audit.rs`.

## CI/CD & Deployment

**Hosting:**
- Hosting/deployment platform configuration: Not detected (repository contains no deployment manifests; `src/main.rs` is a minimal local binary entry).

**CI Pipeline:**
- GitHub Actions workflows in `.github/workflows/compat-governance-pr.yml` and `.github/workflows/compat-governance-release.yml`.
- External CI actions integrated:
  - `actions/checkout@v4`
  - `dtolnay/rust-toolchain@stable`
  - `Swatinem/rust-cache@v2`

## Environment Configuration

**Required env vars:**
- Cargo-provided build variables consumed in `build.rs`:
  - `CARGO_MANIFEST_DIR`
  - `OUT_DIR`
- Application-specific secret env vars: Not detected.

**Secrets location:**
- Repository-managed secret files are not referenced in code or workflows.
- `.env` files are not detected at repository root (`/home/chemtech/workspace/cintx`).

## Webhooks & Callbacks

**Incoming:**
- GitHub-triggered workflow events defined in:
  - `.github/workflows/compat-governance-pr.yml` (`pull_request`, `workflow_dispatch`)
  - `.github/workflows/compat-governance-release.yml` (`release`, `push` tags, `workflow_dispatch`)

**Outgoing:**
- Outbound webhooks/callback integrations: None detected.

---

*Integration audit: 2026-03-21*
