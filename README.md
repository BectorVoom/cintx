### the Rust Redesign and Reimplementation of libcint


## 1.  Evaluating a work list

The integral-family API is per-shell-tuple, because libcint's is. That is a
compatibility requirement, not the shape a Fock build wants: a per-tuple call
pays a kernel launch and a blocking readback per tuple.

For whole work lists, the safe facade takes the list:

```rust
use cintx_rs::prelude::{EvaluationContext, QuartetBatchRequest};

let context = EvaluationContext::new();
let output = QuartetBatchRequest::new(
    int2e_sph,                 // OperatorId
    Representation::Spheric,
    &basis,                    // BasisSet
    quartets,                  // impl IntoIterator<Item = [u32; 4]>, indices into basis.shells()
    options,                   // ExecutionOptions
)
.evaluate_in(&context)?;

// output.values  — concatenated spherical AO blocks, in the request's order
// output.offsets — where each quartet's block starts
// output.stats   — launches, readbacks, transferred bytes
```

The list is grouped into launch classes — quartets sharing `(l_i, l_j, l_k, l_l)`
and therefore the G-tensor shape, the Rys order and the HRR branch — and costs
one dispatch per class rather than one per quartet. `output.stats` reports the
counts a speed claim would be made from, so it stays auditable.

`cintx-cubecl` exposes the same shape one index shorter for the other families:
`evaluate_1e_pair_batch`, `evaluate_2c2e_pair_batch`, `evaluate_3c2e_triple_batch`,
plus `ResidentTwoEBasis` for a basis kept on the device across calls.

Every batched result is **bit-identical** to the per-tuple route, which is
enforced rather than assumed: the per-tuple entry points are themselves one-tuple
batches, so both execute the same kernel.


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
│   ├── cintx-basis/
│   │   ├── Cargo.toml
│   │   ├── data/                       # Vendored BSE tables (see data/README.md)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── catalog.rs              # StandardBasis -> embedded table
│   │       ├── format.rs               # NWChem-dialect parser
│   │       ├── element.rs              # Symbol / Z / mass tables
│   │       ├── normalize.rs            # libcint/PySCF normalization
│   │       ├── build.rs                # Molecule -> typed BasisSet
│   │       └── raw.rs                  # Molecule -> atm/bas/env arrays
│   ├── cintx-driver/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── basis_view.rs           # Borrowed atm/bas/env view
│   │       ├── worklist.rs             # 8-fold canonical pair/quartet lists
│   │       ├── screening.rs            # Cauchy-Schwarz (Schwarz table)
│   │       ├── bucket.rs               # Launch classes + tiering
│   │       └── execute.rs              # Batch run + auditable statistics
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
