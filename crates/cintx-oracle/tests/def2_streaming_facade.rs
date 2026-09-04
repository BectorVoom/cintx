//! `def2_speed_memory_optimization_plan.md` M2 — the streaming consumer surface.
//!
//! # Why a work list needs a streaming shape at all
//!
//! `QuartetBatchRequest::evaluate` returns every block of the work list in one
//! buffer. For SO2/def2-TZVP that is 99.8 MiB and merely large; for a 30-atom
//! def2-TZVP system the dense ERI tensor is measured in terabytes and the
//! request is not one a machine can answer in that form.
//!
//! A direct-SCF Fock build never wanted it in that form. It wants each block
//! once, contracted into a matrix, and discarded — so the peak it needs is one
//! chunk, not one molecule. `for_each_chunk` is that shape, and
//! `evaluate_into` is the middle case: the caller owns the buffer and reuses it
//! across SCF iterations instead of reallocating.
//!
//! What this file pins:
//!
//! 1. **Streaming reproduces the materialized run bit for bit**, block for
//!    block, in the same order. Chunking is a memory decision and must not be a
//!    numerical one.
//! 2. **`evaluate_into` matches `evaluate`**, and refuses a short buffer rather
//!    than writing past it.
//! 3. **A consumer's error stops the stream** and comes back as the consumer's
//!    own error, not a backend paraphrase of it.

#![cfg(feature = "cpu")]

#[path = "def2_fixtures.rs"]
mod def2_fixtures;

use cintx_basis::{StandardBasis, to_raw_arrays};
use cintx_core::{BasisSet, OperatorId, Representation};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_ops::resolver::Resolver;
use cintx_rs::FacadeError;
use cintx_rs::prelude::{EvaluationContext, QuartetBatchRequest};
use cintx_runtime::{BackendIntent, BackendKind, ExecutionOptions};
use def2_fixtures::sulfur_dioxide;

/// The one operator the batch surface serves.
fn int2e_sph_operator() -> OperatorId {
    Resolver::descriptor_by_symbol("int2e_sph")
        .expect("int2e_sph must be in the manifest")
        .id
}

fn options(limit: Option<usize>) -> ExecutionOptions {
    ExecutionOptions {
        backend_intent: BackendIntent {
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        memory_limit_bytes: limit,
        ..Default::default()
    }
}

/// SO2/def2-SVP: 20 shells, 22 155 canonical quartets, a 4.9 MiB output.
///
/// def2-SVP rather than def2-TZVP because the streaming surface is a memory
/// property and should be gated in every build; TZVP's `nroots = 6` classes
/// need the `extended-device-rys` feature.
fn fixture() -> (BasisSet, Vec<[u32; 4]>) {
    let molecule = sulfur_dioxide(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis_view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let list = enumerate_quartets(&enumerate_pairs(&basis_view))
        .into_iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();
    (molecule.to_basis_set().expect("basis set"), list)
}

fn request<'a>(
    basis: &'a BasisSet,
    list: &[[u32; 4]],
    limit: Option<usize>,
) -> QuartetBatchRequest<'a> {
    QuartetBatchRequest::new(
        int2e_sph_operator(),
        Representation::Spheric,
        basis,
        list.iter().copied(),
        options(limit),
    )
}

/// The stream must reproduce the materialized run exactly, in order.
#[test]
fn streaming_reproduces_the_materialized_run_bit_for_bit() {
    let (basis, list) = fixture();
    let context = EvaluationContext::new();

    let whole = request(&basis, &list, None)
        .evaluate_in(&context)
        .expect("materialized batch");

    // A 6 MiB working set above the output forces several chunks.
    let limit = whole.stats.host_output_bytes + 6 * 1024 * 1024;
    let mut seen = 0_usize;
    let mut chunks = 0_usize;
    let mut peak_chunk_bytes = 0_usize;
    let stats = request(&basis, &list, Some(limit))
        .for_each_chunk(&context, &mut |chunk| {
            assert_eq!(
                chunk.first_quartet, seen,
                "chunks must arrive in request order without gaps"
            );
            assert_eq!(chunk.offsets.len(), chunk.quartets + 1);
            peak_chunk_bytes =
                peak_chunk_bytes.max(chunk.values.len() * std::mem::size_of::<f64>());
            for n in 0..chunk.quartets {
                let index = chunk.first_quartet + n;
                let block = &chunk.values[chunk.offsets[n]..chunk.offsets[n + 1]];
                let reference_start = whole.offsets[index];
                for (element, value) in block.iter().enumerate() {
                    let reference = whole.values[reference_start + element];
                    assert_eq!(
                        value.to_bits(),
                        reference.to_bits(),
                        "quartet {index} element {element}: streamed {value:.17e} vs {reference:.17e}"
                    );
                }
            }
            seen += chunk.quartets;
            chunks += 1;
            Ok(())
        })
        .expect("streamed batch");

    assert_eq!(seen, list.len(), "every quartet must be delivered once");
    assert!(
        chunks > 1,
        "the budget must actually produce several chunks"
    );
    assert!(
        peak_chunk_bytes < whole.stats.host_output_bytes,
        "a chunk must be smaller than the whole output: {peak_chunk_bytes} vs {}",
        whole.stats.host_output_bytes
    );
    println!(
        "SO2/def2-SVP streamed in {chunks} chunks, peak chunk {:.2} MiB against a {:.2} MiB output",
        peak_chunk_bytes as f64 / (1024.0 * 1024.0),
        whole.stats.host_output_bytes as f64 / (1024.0 * 1024.0),
    );
    assert_eq!(stats.items_executed, list.len());
}

/// The streaming budget is charged against one chunk, not the whole list.
///
/// A limit sized only for the whole materialized output defeats the point of
/// streaming — the caller asked to bound memory to less than the full list.
/// A budget well below the whole output, but big enough for one small chunk
/// plus its tables and scratch, must still be admitted.
#[test]
fn streaming_admits_a_budget_smaller_than_the_whole_output() {
    let (basis, list) = fixture();
    let context = EvaluationContext::new();

    let whole = request(&basis, &list, None)
        .evaluate_in(&context)
        .expect("materialized batch");

    // Well under the 4.93 MiB whole output, but ample for one chunk of this
    // fixture's launch classes.
    let limit = 2 * 1024 * 1024;
    assert!(
        limit < whole.stats.host_output_bytes,
        "the point of this test is a budget smaller than the whole output"
    );

    let mut seen = 0_usize;
    let mut chunks = 0_usize;
    let stats = request(&basis, &list, Some(limit))
        .for_each_chunk(&context, &mut |chunk| {
            for n in 0..chunk.quartets {
                let index = chunk.first_quartet + n;
                let block = &chunk.values[chunk.offsets[n]..chunk.offsets[n + 1]];
                let reference_start = whole.offsets[index];
                for (element, value) in block.iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        whole.values[reference_start + element].to_bits(),
                        "quartet {index} element {element} differs from the materialized run"
                    );
                }
            }
            seen += chunk.quartets;
            chunks += 1;
            Ok(())
        })
        .expect("a per-chunk budget below the whole output must still be admitted");

    assert_eq!(seen, list.len(), "every quartet must be delivered once");
    assert!(chunks > 1, "a 2 MiB budget must force several chunks");
    assert_eq!(stats.items_executed, list.len());
}

/// `evaluate_into` is the same run against the caller's buffer.
#[test]
fn evaluate_into_matches_the_allocating_form() {
    let (basis, list) = fixture();
    let context = EvaluationContext::new();

    let whole = request(&basis, &list, None)
        .evaluate_in(&context)
        .expect("materialized batch");

    let mut values = vec![0.0_f64; whole.values.len()];
    let (offsets, stats) = request(&basis, &list, None)
        .evaluate_into(&context, &mut values)
        .expect("evaluate_into");

    assert_eq!(offsets, whole.offsets);
    assert_eq!(stats.items_executed, whole.stats.items_executed);
    for (index, (a, b)) in values.iter().zip(whole.values.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {index} differs");
    }
}

/// A buffer too short is refused, not overrun.
#[test]
fn evaluate_into_refuses_a_short_buffer() {
    let (basis, list) = fixture();
    let context = EvaluationContext::new();
    let mut values = vec![0.0_f64; 16];
    let error = request(&basis, &list, None)
        .evaluate_into(&context, &mut values)
        .expect_err("a 16-element buffer must be refused");
    // The typed backend failure travels out; what matters is that it is a
    // refusal rather than a write past the end.
    assert!(
        format!("{error}").contains("BufferTooSmall"),
        "expected a buffer-size refusal, got {error}"
    );
    assert!(
        values.iter().all(|value| *value == 0.0),
        "a refused call must not have written anything"
    );
}

/// A consumer that fails stops the stream, and its own error comes back.
#[test]
fn a_consumer_error_stops_the_stream() {
    let (basis, list) = fixture();
    let context = EvaluationContext::new();
    let limit = 5 * 1024 * 1024 + 6 * 1024 * 1024;

    let mut delivered = 0_usize;
    let error = request(&basis, &list, Some(limit))
        .for_each_chunk(&context, &mut |chunk| {
            delivered += chunk.quartets;
            Err(FacadeError::Validation {
                detail: "consumer stopped early".to_owned(),
            })
        })
        .expect_err("the consumer's refusal must surface");
    match error {
        FacadeError::Validation { detail } => assert_eq!(detail, "consumer stopped early"),
        other => panic!("expected the consumer's own error, got {other:?}"),
    }
    assert!(delivered > 0, "the consumer should have seen one chunk");
    assert!(
        delivered < list.len(),
        "the stream must stop rather than run to completion: {delivered} of {}",
        list.len()
    );
}
