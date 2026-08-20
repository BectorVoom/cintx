mod support;

use cintx_rs::EvaluationContext;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn benchmark_micro_families(c: &mut Criterion) {
    let fixture = support::fixture();
    let cases = [
        &support::OVERLAP_CART,
        &support::KINETIC_CART,
        &support::TWO_ELECTRON_CART,
    ];
    // Keep the Criterion group stable with xtask's suite identifier so timing
    // estimates and the warm-path metric rows can be joined in the artifact.
    let mut group = c.benchmark_group("micro_families");
    group.sample_size(20);
    for case in cases {
        let context = EvaluationContext::new();
        let warm = support::evaluate_in(&fixture, case, &context);
        support::record_measurement("micro_families", case.id, warm);
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
        support::record_measurement("micro_families", &scalar_case_id, scalar_warm);
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

        let context = EvaluationContext::new();
        let warm = support::evaluate_overlap_ss_batch_in(&fixture, item_count, &context);
        let batch_case_id = format!("int1e_ovlp_cart_ss_batch/{item_count}");
        support::record_measurement("micro_families", &batch_case_id, warm);
        black_box((
            warm.workspace_bytes,
            warm.transfer_bytes,
            warm.not0,
            warm.kernel_launch_count,
            warm.readback_count,
        ));
        group.throughput(Throughput::Elements(warm.output_elements as u64));
        group.bench_with_input(
            BenchmarkId::new("int1e_ovlp_cart_ss_batch", item_count),
            &item_count,
            |bench, &item_count| {
                bench.iter(|| {
                    black_box(support::evaluate_overlap_ss_batch_in(
                        &fixture, item_count, &context,
                    ))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(micro_families, benchmark_micro_families);
criterion_main!(micro_families);
