mod support;

use cintx_rs::EvaluationContext;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn benchmark_crossover_cpu_gpu(c: &mut Criterion) {
    // CI records a CPU curve; each configured CubeCL backend produces its own
    // real curve. No modeled GPU duration is emitted.
    let fixture = support::fixture();
    let cases = [
        &support::OVERLAP_CART,
        &support::KINETIC_CART,
        &support::TWO_ELECTRON_CART,
    ];
    // This name is part of the benchmark artifact contract consumed by xtask.
    let mut group = c.benchmark_group("crossover_cpu_gpu");
    group.sample_size(12);
    for case in cases {
        let context = EvaluationContext::new();
        let warm = support::evaluate_in(&fixture, case, &context);
        support::record_measurement("crossover_cpu_gpu", case.id, warm);
        black_box((
            warm.workspace_bytes,
            warm.transfer_bytes,
            warm.not0,
            warm.kernel_launch_count,
            warm.readback_count,
        ));
        group.throughput(Throughput::Elements(warm.output_elements as u64));
        group.bench_with_input(BenchmarkId::from_parameter(case.id), case, |bench, case| {
            bench.iter(|| black_box(support::evaluate_in(&fixture, case, &context)));
        });
    }
    for item_count in support::BATCH_SIZES {
        let scalar_context = EvaluationContext::new();
        let scalar_warm =
            support::evaluate_overlap_ss_scalar_batch_in(&fixture, item_count, &scalar_context);
        let scalar_case_id = format!("int1e_ovlp_cart_ss_scalar/{item_count}");
        support::record_measurement("crossover_cpu_gpu", &scalar_case_id, scalar_warm);
        black_box((
            scalar_warm.workspace_bytes,
            scalar_warm.transfer_bytes,
            scalar_warm.not0,
            scalar_warm.kernel_launch_count,
            scalar_warm.readback_count,
        ));
        group.throughput(Throughput::Elements(scalar_warm.output_elements as u64));
        group.bench_with_input(
            BenchmarkId::new("int1e_ovlp_cart_ss_scalar", item_count),
            &item_count,
            |bench, &item_count| {
                bench.iter(|| {
                    black_box(support::evaluate_overlap_ss_scalar_batch_in(
                        &fixture,
                        item_count,
                        &scalar_context,
                    ))
                });
            },
        );

        let batch_context = EvaluationContext::new();
        let batch_warm =
            support::evaluate_overlap_ss_batch_in(&fixture, item_count, &batch_context);
        let batch_case_id = format!("int1e_ovlp_cart_ss_batch/{item_count}");
        support::record_measurement("crossover_cpu_gpu", &batch_case_id, batch_warm);
        black_box((
            batch_warm.workspace_bytes,
            batch_warm.transfer_bytes,
            batch_warm.not0,
            batch_warm.kernel_launch_count,
            batch_warm.readback_count,
        ));
        group.throughput(Throughput::Elements(batch_warm.output_elements as u64));
        group.bench_with_input(
            BenchmarkId::new("int1e_ovlp_cart_ss_batch", item_count),
            &item_count,
            |bench, &item_count| {
                bench.iter(|| {
                    black_box(support::evaluate_overlap_ss_batch_in(
                        &fixture,
                        item_count,
                        &batch_context,
                    ))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(crossover_cpu_gpu, benchmark_crossover_cpu_gpu);
criterion_main!(crossover_cpu_gpu);
