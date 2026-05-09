# Changelog

All notable changes to **cintx** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING (next release):** `Builder::default()` and any safe-API caller that
  uses `..ExecutionOptions::default()` now resolves to `BackendKind::Cpu`
  (previously `BackendKind::Wgpu`). Callers that need the wgpu backend must
  opt in explicitly via `BackendIntent { backend: BackendKind::Wgpu, .. }` and
  enable the `wgpu` feature on `cintx-cubecl`. This aligns the implicit
  default with Phase 16's `CINTX_BACKEND` unset-env-var contract (defaults to
  cpu) per ROADMAP success criterion 5. Migration: pass an explicit
  `BackendIntent` (any production wgpu code already does this), or set
  `CINTX_BACKEND=wgpu` and call `--features wgpu` at the consumer.

### Added
- `cintxRsError::BackendNotCompiled { requested: String, compiled_in: Vec<String> }`
  typed error variant in `cintx-core`. Surfaces through the existing public error
  enum; rendered Display matches `requested "<name>" is not compiled in;
  compiled-in backends: ["<a>", "<b>"]`. Used in Wave 1 by the fallible
  `resolve_backend_kind() -> Result<BackendKind, cintxRsError>` rewire. (Phase 16,
  D-01.)
- `CintxStatus::BackendNotCompiled = 10` and `CINTX_STATUS_BACKEND_NOT_COMPILED`
  C-ABI status code in `cintx-capi`, with mapping arm in `status_from_core_error`
  and exported-constant test coverage. Stable code; never to be re-used.
