#!/usr/bin/env bash
# GTH-MOLOPT 2e profile on an NVIDIA T4 (Google Colab), with hardware counters.
#
# `ci/colab_t4_verification.sh` establishes that the CUDA backend produces
# libcint's numbers. This script is the *measurement* half for the batched
# `int2e_sph` path over the GTH-MOLOPT bases, following the four steps of the
# CubeCL profiling manual (`16_profiling_and_bottleneck_identification.md`):
#
#   1. verify correctness first (a profile of a wrong kernel is a wasted run),
#   2. time portably, as in-process A/B ratios (`gth_profile`),
#   3. attribute with vendor counters (`nsys` timeline, `ncu` memory metrics),
#   4. leave every number in one directory to paste back.
#
# The switches the profile alternates are runtime scalars — one compiled
# program per launch signature — so an A/B here measures the kernel and not
# the JIT. The one it exists to answer on a discrete GPU is G1, the ket-pair
# split (`CINTX_2E_KL_SPLIT`): `klsplit=off` is one quartet per cube, the
# shape `gth_molopt_speed_memory_plan.md` §10.5 found latency-bound on
# gfx1151; the default spreads each quartet over enough cubes to give every
# SM several workgroups.
#
# A T4's f64 rate is 1/32 of its f32 rate (~254 GFLOP/s). Absolute times here
# describe the T4; the *ratios* — split vs unsplit, contraction probe vs
# default, lane-0 vs split build — describe the kernel and transfer.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#
#   Runtime > Change runtime type > T4 GPU, then:
#
#     !git clone <your cintx remote> /content/cintx     # or upload and untar
#     !bash /content/cintx/ci/colab_t4_profile.sh
#
#   Results land in $OUT (default /content/cintx_t4_profile). Budget 30-50
#   minutes on Colab's 2-core VM; the release build dominates, `ncu` is the
#   next largest cost (it replays every profiled launch).
#
#   CINTX_T4_FIXTURES=H2O      which GTH fixtures the profile runs (label filter)
#   CINTX_T4_SKIP_NCU=1        timeline and A/B only, no counter replay
#   CINTX_T4_NCU_LAUNCHES=3    launches ncu profiles per kernel after the warm-up

set -uo pipefail

REPO="${CINTX_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FEATURES="cpu,cuda,extended-device-rys,gth"
OUT="${CINTX_T4_OUT:-/content/cintx_t4_profile}"
FIXTURES="${CINTX_T4_FIXTURES:-H2O}"
NCU_LAUNCHES="${CINTX_T4_NCU_LAUNCHES:-3}"
mkdir -p "$OUT"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }

# ── 1. The device and the tools ──────────────────────────────────────────────
say "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "no nvidia-smi: this session has no GPU. Runtime > Change runtime type > T4 GPU."
    exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv \
    | tee "$OUT/gpu.csv"

say "Profilers"
for tool in nsys ncu; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "$tool: $(command -v "$tool")"
    else
        # Colab images ship the CUDA toolkit under /usr/local/cuda; the
        # profilers live beside it when present.
        for dir in /usr/local/cuda/bin /usr/local/cuda/nsight-compute* /opt/nvidia/nsight-systems/*/bin; do
            [ -x "$dir/$tool" ] && { export PATH="$dir:$PATH"; echo "$tool: $dir/$tool"; break; }
        done
        command -v "$tool" >/dev/null 2>&1 || warn "$tool not found; that step will be skipped"
    fi
done

# ── 2. Toolchain ─────────────────────────────────────────────────────────────
say "Rust"
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
cargo --version && rustc --version
command -v cc >/dev/null 2>&1 || { warn "no C compiler: the vendored libcint oracle cannot be built"; exit 1; }

# ── 3. Build ─────────────────────────────────────────────────────────────────
say "Build (release, features: $FEATURES)"
cd "$REPO" || exit 1
export CINTX_ORACLE_BUILD_VENDOR=1
if ! cargo test --release -p cintx-oracle --features "$FEATURES" \
        --test def2_cuda_verification --test gth_profile --test two_e_cooperative_arm --no-run \
        2>&1 | tail -20; then
    warn "build failed"
    exit 1
fi
# The test binaries, by name, so the profilers can be pointed at them directly
# rather than through cargo (which would profile the build as well).
BIN_PROFILE=$(ls -t target/release/deps/gth_profile-* | grep -v '\.d$' | head -1)
echo "gth_profile binary: $BIN_PROFILE"

# ── 4. Correctness first ─────────────────────────────────────────────────────
export CINTX_CUDA_ORACLE=1
STATUS=0
say "Step 1 — correctness: def2 on CUDA against vendored libcint"
cargo test --release -p cintx-oracle --features "$FEATURES" \
    --test def2_cuda_verification -- --ignored --nocapture --test-threads=1 \
    2>&1 | tee "$OUT/step1_def2_cuda_verification.log" | tail -25 || STATUS=1

say "Step 1 — correctness: the ket-pair split (G1) and the cooperative G build, CPU-pinned"
cargo test --release -p cintx-oracle --features "$FEATURES" \
    --test two_e_cooperative_arm 2>&1 | tee "$OUT/step1_cooperative_arm.log" | tail -12 || STATUS=1

# ── 5. Portable timing: the in-process A/B on CUDA ───────────────────────────
#
# One table per fixture: default, naive contraction, no-contraction probe,
# klsplit=off, coop=lane0, xform=device — every row a ratio against the
# default in the same process, with the vendor gap and memory columns.
say "Step 2 — gth_profile on CUDA ($FIXTURES)"
CINTX_GTH_BACKEND=cuda CINTX_GTH_FILTER="$FIXTURES" CINTX_BENCH_REPEATS=3 \
    "$BIN_PROFILE" --ignored --nocapture 2>&1 | tee "$OUT/step2_gth_profile_cuda.log" \
    | grep -E '^==|variant|default |naive|probe|klsplit|coop|xform|chunk=|launches|panicked|test result' \
    || STATUS=1

say "Step 2 — gth_profile on the CPU backend of this VM, for the cross-backend gap"
CINTX_GTH_BACKEND=cpu CINTX_GTH_FILTER="$FIXTURES" CINTX_BENCH_REPEATS=2 CINTX_GTH_DUMP="$OUT" \
    "$BIN_PROFILE" --ignored --nocapture 2>&1 | tee "$OUT/step2_gth_profile_cpu.log" \
    | grep -E '^==|default |launches|dumped|test result' || true

# ── 6. Attribution: the timeline and the counters ────────────────────────────
#
# The kernel's entry-point symbol is the function name plus the `Float`
# generic's type: `two_electron_scalar_kernel_f_f64`, and the ket-split reduce
# `reduce_kl_partials_f_f64` (manual §3.2). `--launch-skip` steps over the
# prewarm; the profile harness alternates variants, so the first launches
# after it are the default configuration.
say "Step 3 — nsys timeline"
if command -v nsys >/dev/null 2>&1; then
    CINTX_GTH_BACKEND=cuda CINTX_GTH_FILTER="$FIXTURES" CINTX_BENCH_REPEATS=1 \
        nsys profile --stats=true --force-overwrite=true -o "$OUT/timeline" \
        "$BIN_PROFILE" --ignored --nocapture > "$OUT/step3_nsys.log" 2>&1 || warn "nsys failed"
    nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT/kern_sum" "$OUT/timeline.nsys-rep" \
        > /dev/null 2>&1 || true
    nsys stats --report cuda_gpu_kern_sum "$OUT/timeline.nsys-rep" 2>/dev/null | head -30
else
    warn "nsys not available; skipping the timeline"
fi

say "Step 3 — ncu memory-pipeline metrics on the 2e kernel"
if [ "${CINTX_T4_SKIP_NCU:-0}" != "1" ] && command -v ncu >/dev/null 2>&1; then
    # Confirm the metric names on this part before relying on them (manual §3.3).
    ncu --query-metrics 2>/dev/null | grep -E 'hit_rate|bank_conflicts|bytes_per_sector|sm__throughput|dram__throughput' \
        | head -20 > "$OUT/ncu_available_metrics.txt" || true
    METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_registers,launch__registers_per_thread,\
launch__grid_size,launch__block_size,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,\
sm__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
sm__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__inst_executed.sum,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct"
    CINTX_GTH_BACKEND=cuda CINTX_GTH_FILTER="$FIXTURES" CINTX_BENCH_REPEATS=1 \
        ncu --kernel-name regex:'two_electron_scalar_kernel.*|reduce_kl_partials.*' \
            --launch-skip 4 --launch-count "$NCU_LAUNCHES" \
            --metrics "$METRICS" --csv --log-file "$OUT/ncu_metrics.csv" \
            "$BIN_PROFILE" --ignored --nocapture > "$OUT/step3_ncu.log" 2>&1 \
        || warn "ncu failed (see $OUT/step3_ncu.log; Colab may need --target-processes all or a newer driver)"
    # One full-section report of a single launch, for the guided analysis.
    CINTX_GTH_BACKEND=cuda CINTX_GTH_FILTER="$FIXTURES" CINTX_BENCH_REPEATS=1 \
        ncu --kernel-name regex:'two_electron_scalar_kernel.*' --launch-skip 4 --launch-count 1 \
            --set full -o "$OUT/two_electron_full" -f \
            "$BIN_PROFILE" --ignored --nocapture > "$OUT/step3_ncu_full.log" 2>&1 \
        || warn "ncu --set full failed (non-fatal)"
    [ -f "$OUT/ncu_metrics.csv" ] && head -5 "$OUT/ncu_metrics.csv"
else
    warn "ncu skipped"
fi

# ── 7. Summary ───────────────────────────────────────────────────────────────
say "Summary"
python3 - "$OUT" <<'PY'
import csv, json, re, sys, pathlib
out = pathlib.Path(sys.argv[1])
summary = {"gpu": (out / "gpu.csv").read_text().strip() if (out / "gpu.csv").exists() else None,
           "profile_cuda": {}, "ncu": {}}
log = out / "step2_gth_profile_cuda.log"
if log.exists():
    case = None
    for line in log.read_text().splitlines():
        m = re.match(r"== (.*) ==", line)
        if m:
            case = m.group(1); summary["profile_cuda"][case] = {}
            continue
        m = re.match(r"\s+(\S+)\s+([\d.]+)\s+([\d.]+)x", line)
        if m and case:
            summary["profile_cuda"][case][m.group(1)] = {"ms": float(m.group(2)), "ratio": float(m.group(3))}
        m = re.match(r"\s+launches=(\d+) classes=(\d+) chunks=(\d+) kl_split=(\d+)", line)
        if m and case:
            summary["profile_cuda"][case]["launches"] = int(m.group(1))
            summary["profile_cuda"][case]["kl_split"] = int(m.group(4))
csvp = out / "ncu_metrics.csv"
if csvp.exists():
    rows = [r for r in csv.reader(csvp.open()) if r and r[0] != "==PROF=="]
    header = next((r for r in rows if "Metric Name" in r), None)
    if header:
        i_kernel, i_name, i_val = header.index("Kernel Name"), header.index("Metric Name"), header.index("Metric Value")
        for r in rows:
            if len(r) > i_val and r is not header and r[i_name]:
                summary["ncu"].setdefault(r[i_kernel][:60], {})[r[i_name]] = r[i_val]
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["profile_cuda"], indent=2))
print(f"wrote {out/'summary.json'}")
PY

say "Result"
echo "Artifacts in $OUT:"
ls -1 "$OUT"
if [ "$STATUS" -eq 0 ]; then
    echo "PASS — correctness gates held; paste summary.json and the step2/step3 logs back."
else
    echo "FAIL — a correctness gate failed; the profile numbers are not meaningful until it passes."
fi
exit "$STATUS"
