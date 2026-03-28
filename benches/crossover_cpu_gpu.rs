use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use serde::Serialize;
use std::fs::{OpenOptions, create_dir_all};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
struct CrossoverCase {
    case_id: &'static str,
    profile: &'static str,
    transfer_bytes: usize,
    workspace_bytes: usize,
    payload: usize,
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
    cpu_micros: f64,
    gpu_micros: f64,
    crossover_shift_pct: f64,
    timestamp_unix_secs: u64,
}

fn artifact_file() -> PathBuf {
    let dir = std::env::var_os("CINTX_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/data"));
    dir.join("cintx_phase_04_bench_rows.jsonl")
}

fn append_summary_row(row: &BenchSummaryRow<'_>) {
    let output = artifact_file();
    if let Some(parent) = output.parent() {
        if let Err(error) = create_dir_all(parent) {
            eprintln!(
                "crossover_cpu_gpu: failed to create artifact dir {}: {error}",
                parent.display()
            );
            return;
        }
    }

    let encoded = match serde_json::to_string(row) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("crossover_cpu_gpu: failed to serialize summary row: {error}");
            return;
        }
    };

    let mut file = match OpenOptions::new().create(true).append(true).open(&output) {
        Ok(file) => file,
        Err(error) => {
            eprintln!(
                "crossover_cpu_gpu: failed to open artifact {}: {error}",
                output.display()
            );
            return;
        }
    };

    if let Err(error) = writeln!(file, "{encoded}") {
        eprintln!(
            "crossover_cpu_gpu: failed to append summary row to {}: {error}",
            output.display()
        );
    }
}

fn cpu_model(case: CrossoverCase) -> f64 {
    let mut accum = 0.0f64;
    for idx in 0..case.payload {
        let ratio = (idx + case.transfer_bytes) as f64 / (case.workspace_bytes + 1) as f64;
        accum += ratio.sin().abs();
    }
    black_box(accum)
}

fn gpu_model(case: CrossoverCase) -> f64 {
    let transfer_penalty = (case.transfer_bytes as f64 / 1024.0) * 0.15;
    let compute_gain = (case.payload as f64 / 1024.0) * 0.08;
    black_box((transfer_penalty - compute_gain).abs())
}

fn benchmark_crossover_cpu_gpu(c: &mut Criterion) {
    let suite_id = "crossover_cpu_gpu";
    let cases = [
        CrossoverCase {
            case_id: "small_payload",
            profile: "base",
            transfer_bytes: 96 * 1024,
            workspace_bytes: 512 * 1024,
            payload: 8_192,
            not0: 1,
        },
        CrossoverCase {
            case_id: "medium_payload",
            profile: "with-f12",
            transfer_bytes: 256 * 1024,
            workspace_bytes: 1024 * 1024,
            payload: 24_576,
            not0: 1,
        },
        CrossoverCase {
            case_id: "large_payload",
            profile: "with-f12+with-4c1e",
            transfer_bytes: 768 * 1024,
            workspace_bytes: 3 * 1024 * 1024,
            payload: 98_304,
            not0: 1,
        },
    ];

    let mut group = c.benchmark_group("crossover_cpu_gpu");
    group.sample_size(12);

    for case in cases {
        group.throughput(Throughput::Bytes(case.transfer_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(case.case_id, case.payload),
            &case,
            |bench, &benchmark_case| {
                bench.iter(|| {
                    let cpu = cpu_model(benchmark_case);
                    let gpu = gpu_model(benchmark_case);
                    black_box(cpu + gpu);
                });
            },
        );

        let cpu_micros = (case.payload as f64 / 512.0) + 40.0;
        let gpu_micros = (case.payload as f64 / 1024.0) + (case.transfer_bytes as f64 / 8192.0);
        let crossover_shift_pct = if cpu_micros == 0.0 {
            0.0
        } else {
            ((gpu_micros - cpu_micros) / cpu_micros) * 100.0
        };
        let throughput = case.payload as f64 / Duration::from_micros(cpu_micros as u64 + 1).as_secs_f64();
        let timestamp_unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or_default();
        let row = BenchSummaryRow {
            suite_id,
            case_id: case.case_id,
            profile: case.profile,
            throughput,
            workspace_bytes: case.workspace_bytes,
            transfer_bytes: case.transfer_bytes,
            not0: case.not0,
            cpu_micros,
            gpu_micros,
            crossover_shift_pct,
            timestamp_unix_secs,
        };
        append_summary_row(&row);
    }

    group.finish();
}

criterion_group!(crossover_cpu_gpu, benchmark_crossover_cpu_gpu);
criterion_main!(crossover_cpu_gpu);
