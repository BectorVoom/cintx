mod support;

use cintx_rs::EvaluationContext;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn benchmark_macro_molecules(c: &mut Criterion) {
    let fixture = support::fixture();
    let cases = [
        ("h2_like_overlap", &support::OVERLAP_CART),
        ("h2_like_kinetic", &support::KINETIC_CART),
        ("h2_like_eri", &support::TWO_ELECTRON_CART),
    ];
    // This name is part of the benchmark artifact contract consumed by xtask.
    let mut group = c.benchmark_group("macro_molecules");
    group.sample_size(15);
    for (name, case) in cases {
        let context = EvaluationContext::new();
        let warm = support::evaluate_in(&fixture, case, &context);
        let case_id = format!("{name}/{}", case.id);
        support::record_measurement("macro_molecules", &case_id, warm);
        black_box((
            warm.workspace_bytes,
            warm.transfer_bytes,
            warm.not0,
            warm.kernel_launch_count,
            warm.readback_count,
        ));
        group.throughput(Throughput::Elements(warm.output_elements as u64));
        group.bench_with_input(BenchmarkId::new(name, case.id), case, |bench, case| {
            bench.iter(|| black_box(support::evaluate_in(&fixture, case, &context)));
        });
    }
    group.finish();
}

criterion_group!(macro_molecules, benchmark_macro_molecules);
criterion_main!(macro_molecules);
