# External Integrations

**Analysis Date:** 2026-03-14

## APIs & External Services

**Application Runtime Integrations (current root crate):**
- None implemented yet. The executable only prints a static message and does not call external APIs.

**Build/Dependency Services:**
- crates.io registry (build-time dependency source for Rust crates).
- Vendored upstream `libcint` references GitHub as its upstream source/documentation, but this repo currently vendors code locally rather than calling remote API endpoints at runtime.

**Evidence:**
- `src/main.rs` (only `println!("Hello, world!")`)
- `Cargo.lock` (`source = "registry+https://github.com/rust-lang/crates.io-index"`)
- `libcint-master/README.rst` (upstream GitHub references)

## Data Storage

**Databases:**
- None configured.

**File Storage:**
- Repository-local static data files are bundled in vendored `libcint` sources (numerical table files), not external object storage.

**Caching:**
- None configured.

**Evidence:**
- `src/main.rs`
- `Cargo.toml` (no DB/storage client dependencies)
- `libcint-master/src/roots_xw.dat`
- `libcint-master/src/sr_roots_part0_x.dat`

## Authentication & Identity

**Auth Provider:**
- None configured.

**OAuth Integrations:**
- None configured.

**Evidence:**
- `src/main.rs`
- `Cargo.toml` (no auth SDK dependencies)

## Monitoring & Observability

**Error Tracking:**
- No external error tracking integration configured.

**Analytics:**
- No analytics integration configured.

**Logs:**
- Local stdout logging only from the root binary.

**Evidence:**
- `src/main.rs`
- `Cargo.toml` (no monitoring/analytics SDK dependencies)

## CI/CD & Deployment

**Hosting:**
- No hosting/deployment target is defined for the root crate in repository-level deployment config.

**CI Pipeline:**
- No root-level CI workflow is present for this repository snapshot.
- A vendored upstream `.travis.yml` exists under `libcint-master/` for upstream C library CI behavior.

**Evidence:**
- `README.md`
- `libcint-master/.travis.yml`
- `libcint-master/CMakeLists.txt` (manual/local CMake build and optional tests)

## Environment Configuration

**Development:**
- No required runtime environment variables are currently consumed by the root crate.
- Vendored C library behavior is configured primarily via CMake options during build.

**Staging:**
- Not defined.

**Production:**
- Not defined.

**Evidence:**
- `src/main.rs`
- `libcint-master/CMakeLists.txt` (compile-time flags like `WITH_F12`, `WITH_4C1E`, `WITH_FORTRAN`)
- `.gitignore`

## Webhooks & Callbacks

**Incoming:**
- None configured.

**Outgoing:**
- None configured.

**Evidence:**
- `src/main.rs`
- `Cargo.toml`

---

*Integration audit: 2026-03-14*
*Update when adding/removing external services*
