### the Rust Redesign and Reimplementation of libcint


## 2.  Source Tree

```text
cintx-rs/
├── Cargo.toml                           # Workspace definition
├── rust-toolchain.toml                  # Toolchain pin
├── README.md                            # Usage overview / feature matrix
├── LICENSE
├── libcint-master                       # libcint project(origin)
├── test/home/chemtech/workspace/cintx/test/rust_crate_guideline.md
├── docs/
│   ├── design/
│   │   ├── cintx_rust_detailed_design_reviewed.md  # This design document
│   │   ├── api_manifest.csv                          # Generated manifest
│  
│ 
├── crates/
│   ├── cintx-rs/
│   │   ├── Cargo.toml                  # Facade crate
│   │   └── src/
│   │       ├── lib.rs                  # Facade exports
│   │       ├── api.rs                  # Safe Rust API
│   │       ├── builder.rs              # Builders
│   │       └── prelude.rs              # Convenience re-exports
│   ├── cintx-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── atom.rs                 # Atom / NuclearModel
│   │       ├── shell.rs                # Shell / ShellTuple2/3/4
│   │       ├── basis.rs                # BasisSet / BasisMeta / counts
│   │       ├── env.rs                  # EnvParams
│   │       ├── operator.rs             # Representation / OperatorId
│   │       ├── tensor.rs               # TensorShape / TensorLayout / views
│   │       └── error.rs                # thiserror v2 errors
│   ├── cintx-ops/
│   │   ├── Cargo.toml
│   │   ├── build.rs                    # Manifest codegen
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── generated/
│   │       │   ├── api_manifest.rs     # Generated enum/table
│   │       │   └── api_manifest.csv    # Generated snapshot
│   │       └── resolver.rs             # string→OperatorId resolution
│   ├── cintx-runtime/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── validator.rs            # Raw/typed validation
│   │       ├── planner.rs              # ExecutionPlan generation
│   │       ├── scheduler.rs            # Batch/chunking
│   │       ├── workspace.rs            # FallibleBuffer / pools
│   │       ├── dispatch.rs             # CubeCL capability / queue selection
│   │       ├── metrics.rs              # tracing / stats
│   │       └── options.rs              # ExecutionOptions
│   ├── cintx-cubecl/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── executor.rs             # CubeCL backend executor
│   │       ├── kernels/
│   │       │   ├── one_electron.rs     # 1e CubeCL kernels
│   │       │   ├── two_electron.rs     # 2e CubeCL kernels
│   │       │   ├── center_2c2e.rs      # 2c2e CubeCL kernels
│   │       │   ├── center_3c1e.rs      # 3c1e CubeCL kernels
│   │       │   ├── center_3c2e.rs      # 3c2e CubeCL kernels
│   │       │   └── center_4c1e.rs      # 4c1e CubeCL kernels
│   │       ├── transform/
│   │       │   ├── c2s.rs              # device-side cart→sph
│   │       │   └── c2spinor.rs         # device-side cart→spinor
│   │       ├── transfer.rs             # H2D/D2H planner
│   │       ├── resident_cache.rs       # Device metadata cache
│   │       ├── specialization.rs       # Kernel specialization cache
│   │       └── staging.rs              # Host-side packing / launch staging
│   ├── cintx-compat/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── raw.rs                  # Raw compatibility API
│   │       ├── legacy.rs               # Legacy wrappers
│   │       ├── helpers.rs              # Helper APIs
│   │       ├── optimizer.rs            # Optimizer compat handle
│   │       ├── transform.rs            # Helper transform APIs
│   │       └── layout.rs               # Compat buffer writer
│   ├── cintx-capi/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # extern C exports
│   │       ├── errors.rs               # `last_error` API
│   │       └── shim.rs                 # Symbol compatibility layer
│   └── cintx-oracle/
│   │       ├── Cargo.toml
│   │       ├── build.rs                # Vendored cintx build + bindgen
│   │       └── src/
│   │           ├── lib.rs              # Oracle adapter
│   │           ├── compare.rs          # Comparison harness
│   │           └── fixtures.rs         # Test datasets
├── xtask/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                     # Subcommand entry
│       ├── manifest_audit.rs           # Header/source/compiled symbol audit
│       ├── oracle_update.rs            # Oracle sync helper
│       ├── gen_docs.rs                 # Generate docs from manifest
│       └── bench_report.rs             # Benchmark aggregation
├── benches/
│   ├── micro_families.rs               # Family microbench
│   ├── macro_molecules.rs              # Molecule benchmark
│   └── cubecl_batch_threshold.rs       # CubeCL launch/batch-threshold benchmark
└── ci/
    ├── feature-matrix.yml              # Feature CI matrix
    ├── oracle-compare.yml              # Oracle comparison job
    └── gpu-bench.yml                   # GPU benchmark / consistency job
```
