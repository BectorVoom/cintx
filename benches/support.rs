use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
use cintx_rs::{BatchRequest, EvaluationContext, SessionBuilder};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

pub struct RealCase {
    pub id: &'static str,
    pub operator: OperatorId,
    pub representation: Representation,
    pub arity: usize,
}

pub struct Fixture {
    basis: BasisSet,
    shells: Vec<Arc<Shell>>,
}

#[derive(Clone, Copy)]
pub struct Measurement {
    pub workspace_bytes: usize,
    pub transfer_bytes: usize,
    pub not0: i32,
    pub output_elements: usize,
    pub kernel_launch_count: usize,
    pub readback_count: usize,
    /// Host wall-clock control-plane timings. These are not device timestamps.
    pub pack_ns: u64,
    pub submit_ns: u64,
    pub readback_ns: u64,
}

pub const OVERLAP_CART: RealCase = RealCase {
    id: "int1e_ovlp_cart",
    operator: OperatorId::new(0),
    representation: Representation::Cart,
    arity: 2,
};

pub const KINETIC_CART: RealCase = RealCase {
    id: "int1e_kin_cart",
    operator: OperatorId::new(3),
    representation: Representation::Cart,
    arity: 2,
};

pub const TWO_ELECTRON_CART: RealCase = RealCase {
    id: "int2e_cart",
    operator: OperatorId::new(9),
    representation: Representation::Cart,
    arity: 4,
};

/// Calibration points from the CubeCL speed-optimization plan. They cover
/// latency-sensitive calls through batches large enough to amortize launch and
/// transfer overhead.
pub const BATCH_SIZES: [usize; 6] = [1, 8, 32, 128, 512, 2048];

/// Persist the warm-path execution metrics alongside Criterion's timing output.
///
/// Criterion owns the timing estimates; this row records the corresponding
/// allocation, transfer, launch, and host control-plane counters so
/// `xtask bench-report` can join the two sources without fabricating device
/// timing measurements.
pub fn record_measurement(suite_id: &str, case_id: &str, measurement: Measurement) {
    let path = env::var("CINTX_BENCH_ROWS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            env::var("CINTX_ARTIFACT_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/tmp/cintx_artifacts"))
                .join("cintx_phase_04_bench_rows.jsonl")
        });
    let profile = env::var("CINTX_BENCH_PROFILE").unwrap_or_else(|_| "criterion".to_owned());
    let row = serde_json::json!({
        "suite_id": suite_id,
        "case_id": case_id,
        "profile": profile,
        "workspace_bytes": measurement.workspace_bytes,
        "transfer_bytes": measurement.transfer_bytes,
        "not0": measurement.not0,
        "kernel_launch_count": measurement.kernel_launch_count,
        "readback_count": measurement.readback_count,
        "pack_ns": (measurement.pack_ns > 0).then_some(measurement.pack_ns),
        "submit_ns": (measurement.submit_ns > 0).then_some(measurement.submit_ns),
        "readback_ns": (measurement.readback_ns > 0).then_some(measurement.readback_ns),
        "timing_kind": "host_wall_clock_not_device_timestamp",
        "source_kind": "warm_path_metrics",
    });

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create benchmark artifact directory");
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("open benchmark metrics artifact");
    writeln!(file, "{row}").expect("append benchmark metrics artifact row");
}

pub fn fixture() -> Fixture {
    let atoms = Arc::from(
        vec![
            Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).expect("valid atom"),
            Atom::try_new(1, [0.0, 0.0, 1.4], NuclearModel::Point, None, None).expect("valid atom"),
        ]
        .into_boxed_slice(),
    );
    let shells = vec![
        shell(0, 0, &[1.24], &[1.0]),
        shell(1, 0, &[0.78], &[1.0]),
        shell(0, 1, &[0.55], &[0.7]),
        shell(1, 1, &[0.43], &[0.6]),
    ];
    let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice()))
        .expect("valid fixed benchmark basis");
    Fixture { basis, shells }
}

/// Measure the warm path with a caller-owned context. The context is intentionally
/// outside Criterion's iteration so backend resolution and host scratch allocation
/// are not charged to every sample.
pub fn evaluate_in(fixture: &Fixture, case: &RealCase, context: &EvaluationContext) -> Measurement {
    let tuple = ShellTuple::try_from_iter(fixture.shells.iter().take(case.arity).cloned())
        .expect("benchmark tuple arity");
    let output = SessionBuilder::new(case.operator, case.representation, &fixture.basis, tuple)
        .profile_label("cubecl-speed")
        .build()
        .query_workspace_in(context)
        .and_then(|query| query.evaluate())
        .expect("fixed real CubeCL benchmark evaluation must succeed");
    Measurement {
        workspace_bytes: output.stats.workspace_bytes,
        transfer_bytes: output.stats.transfer_bytes,
        not0: output.stats.not0,
        output_elements: output.tensor.owned_values.len(),
        kernel_launch_count: output.stats.chunk_count,
        readback_count: output.stats.chunk_count,
        pack_ns: 0,
        submit_ns: 0,
        readback_ns: 0,
    }
}

/// Execute the verified lane-per-tuple pilot using the two primitive Cartesian s shells.
#[allow(dead_code)] // shared by the three independent Criterion bench binaries
pub fn evaluate_overlap_ss_batch_in(
    fixture: &Fixture,
    item_count: usize,
    context: &EvaluationContext,
) -> Measurement {
    let tuple = ShellTuple::try_from_iter(fixture.shells.iter().take(2).cloned())
        .expect("benchmark s-s tuple");
    let request = SessionBuilder::new(
        OVERLAP_CART.operator,
        OVERLAP_CART.representation,
        &fixture.basis,
        tuple,
    )
    .profile_label("cubecl-speed-batch-pilot")
    .build();
    let output = BatchRequest::new((0..item_count).map(|_| request.clone()))
        .max_items_per_chunk(item_count)
        .evaluate_batch_in(context)
        .expect("fixed s-s batch pilot must succeed");
    let first = output.outputs.first().expect("non-empty benchmark batch");
    Measurement {
        workspace_bytes: first.workspace_bytes,
        transfer_bytes: output
            .outputs
            .iter()
            .map(|item| item.stats.transfer_bytes)
            .sum(),
        not0: output.outputs.iter().map(|item| item.stats.not0).sum(),
        output_elements: output.outputs.len(),
        kernel_launch_count: output.stats.kernel_launch_count,
        readback_count: output.stats.readback_count,
        pack_ns: output.stats.pack_ns,
        submit_ns: output.stats.submit_ns,
        readback_ns: output.stats.readback_ns,
    }
}

/// Compatibility baseline for the same s-s fixture and item count as the batch pilot.
#[allow(dead_code)] // shared by the three independent Criterion bench binaries
pub fn evaluate_overlap_ss_scalar_batch_in(
    fixture: &Fixture,
    item_count: usize,
    context: &EvaluationContext,
) -> Measurement {
    let tuple = ShellTuple::try_from_iter(fixture.shells.iter().take(2).cloned())
        .expect("benchmark s-s tuple");
    let request = SessionBuilder::new(
        OVERLAP_CART.operator,
        OVERLAP_CART.representation,
        &fixture.basis,
        tuple,
    )
    .profile_label("cubecl-speed-batch-pilot-baseline")
    .build();
    let mut measurement = Measurement {
        workspace_bytes: 0,
        transfer_bytes: 0,
        not0: 0,
        output_elements: 0,
        kernel_launch_count: 0,
        readback_count: 0,
        pack_ns: 0,
        submit_ns: 0,
        readback_ns: 0,
    };
    for _ in 0..item_count {
        let output = request
            .clone()
            .query_workspace_in(context)
            .and_then(|query| query.evaluate())
            .expect("fixed s-s scalar benchmark evaluation must succeed");
        measurement.workspace_bytes = measurement.workspace_bytes.max(output.workspace_bytes);
        measurement.transfer_bytes += output.stats.transfer_bytes;
        measurement.not0 += output.stats.not0;
        measurement.output_elements += output.tensor.owned_values.len();
        measurement.kernel_launch_count += output.stats.chunk_count;
        measurement.readback_count += output.stats.chunk_count;
    }
    measurement
}

fn shell(atom_index: u32, ang_momentum: u8, exponents: &[f64], coefficients: &[f64]) -> Arc<Shell> {
    Arc::new(
        Shell::try_new(
            atom_index,
            ang_momentum,
            exponents.len() as u16,
            1,
            0,
            Representation::Cart,
            Arc::from(exponents.to_vec().into_boxed_slice()),
            Arc::from(coefficients.to_vec().into_boxed_slice()),
        )
        .expect("valid fixed benchmark shell"),
    )
}
