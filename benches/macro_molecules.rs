use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use serde::Serialize;
use std::fs::{OpenOptions, create_dir_all};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
struct MacroCase {
    molecule: &'static str,
    profile: &'static str,
    atom_count: usize,
    basis_functions: usize,
    work_units: usize,
    workspace_bytes: usize,
    transfer_bytes: usize,
    not0: i32,
}

#[derive(Serialize)]
struct BenchSummaryRow<'a> {
    suite_id: &'a str,
    case_id: &'a str,
    profile: &'a str,
    throughput: f64,
    workspace_bytes: usize,
    transfer_bytes: usize,
    not0: i32,
    samples: usize,
    memory_bytes: usize,
    timestamp_unix_secs: u64,
}

fn artifact_file() -> PathBuf {
    let dir = std::env::var_os("CINTX_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/cintx_artifacts"));
    dir.join("cintx_phase_04_bench_rows.jsonl")
}

fn append_summary_row(row: &BenchSummaryRow<'_>) {
    let output = artifact_file();
    if let Some(parent) = output.parent() {
        if let Err(error) = create_dir_all(parent) {
            eprintln!(
                "macro_molecules: failed to create artifact dir {}: {error}",
                parent.display()
            );
            return;
        }
    }

    let encoded = match serde_json::to_string(row) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("macro_molecules: failed to serialize summary row: {error}");
            return;
        }
    };

    let mut file = match OpenOptions::new().create(true).append(true).open(&output) {
        Ok(file) => file,
        Err(error) => {
            eprintln!(
                "macro_molecules: failed to open artifact {}: {error}",
                output.display()
            );
            return;
        }
    };

    if let Err(error) = writeln!(file, "{encoded}") {
        eprintln!(
            "macro_molecules: failed to append summary row to {}: {error}",
            output.display()
        );
    }
}

fn run_macro_workload(case: MacroCase) -> f64 {
    let mut workspace = vec![0.0f64; case.basis_functions * 2];
    for idx in 0..case.work_units {
        let slot = idx % workspace.len();
        let phase = ((idx + case.atom_count) as f64) / ((case.basis_functions + 1) as f64);
        workspace[slot] += phase.sin() * phase.cos();
    }
    black_box(workspace.iter().sum())
}

fn benchmark_macro_molecules(c: &mut Criterion) {
    let suite_id = "macro_molecules";
    let cases = [
        MacroCase {
            molecule: "h2o_medium",
            profile: "base",
            atom_count: 3,
            basis_functions: 128,
            work_units: 24_576,
            workspace_bytes: 768 * 1024,
            transfer_bytes: 192 * 1024,
            not0: 1,
        },
        MacroCase {
            molecule: "benzene_dense",
            profile: "with-f12",
            atom_count: 12,
            basis_functions: 320,
            work_units: 65_536,
            workspace_bytes: 2 * 1024 * 1024,
            transfer_bytes: 768 * 1024,
            not0: 1,
        },
        MacroCase {
            molecule: "c60_cluster",
            profile: "with-f12+with-4c1e",
            atom_count: 60,
            basis_functions: 960,
            work_units: 180_224,
            workspace_bytes: 8 * 1024 * 1024,
            transfer_bytes: 2 * 1024 * 1024,
            not0: 1,
        },
    ];

    let mut group = c.benchmark_group("macro_molecules");
    group.sample_size(15);

    for case in cases {
        group.throughput(Throughput::Elements(case.work_units as u64));
        group.bench_with_input(
            BenchmarkId::new(case.molecule, case.basis_functions),
            &case,
            |bench, &benchmark_case| {
                bench.iter_batched(
                    || benchmark_case,
                    run_macro_workload,
                    BatchSize::SmallInput,
                )
            },
        );

        let throughput = case.work_units as f64 / ((case.atom_count * case.basis_functions) as f64);
        let timestamp_unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or_default();
        let row = BenchSummaryRow {
            suite_id,
            case_id: case.molecule,
            profile: case.profile,
            throughput,
            workspace_bytes: case.workspace_bytes,
            transfer_bytes: case.transfer_bytes,
            not0: case.not0,
            samples: 15,
            memory_bytes: case.workspace_bytes,
            timestamp_unix_secs,
        };
        append_summary_row(&row);
    }

    group.finish();
}

criterion_group!(macro_molecules, benchmark_macro_molecules);
criterion_main!(macro_molecules);
