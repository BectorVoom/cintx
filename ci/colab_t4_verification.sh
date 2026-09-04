#!/usr/bin/env bash
# CUDA runtime verification for cintx, sized for a Google Colab T4 session.
#
# `.planning/notes/cuda-metal-verification-gap.md` records the CUDA backend as
# compile-only: it is built in the feature matrix and has never executed a kernel
# on real hardware. This script is what a CUDA device runs to change that.
#
# ── What it establishes ──────────────────────────────────────────────────────
#
#   1. The backend resolves and runs.
#   2. The FMA-fusion probe's answer on NVIDIA, which decides whether the
#      extended Rys ceiling (nroots 6-12, and so def2-TZVP) is available. This
#      is a property of the compiler's contraction behaviour and cannot be
#      inferred from AMD's answer.
#   3. `int2e_sph` against vendored libcint 6.1.3 at the project's 1e-12.
#   4. CUDA against the CPU backend, sizing the cooperative-versus-per-unit
#      divergence on this device.
#   5. The M3 device-side cart-to-sph transform, bit-identical and reading back
#      the spherical output rather than the larger Cartesian one.
#
# ── What it does NOT establish ───────────────────────────────────────────────
#
# **Nothing about speed.** A T4's f64 rate is 1/32 of its f32 rate (about
# 254 GFLOP/s against 8.1 TFLOP/s), and cintx's public path is f64 by contract.
# A throughput number from this device would say something about the T4, not
# about cintx. It is a correctness target, exactly as the gfx1151 in the
# development host is.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#
#   Runtime > Change runtime type > T4 GPU, then:
#
#     !git clone <your cintx remote> /content/cintx     # or upload and untar
#     !bash /content/cintx/ci/colab_t4_verification.sh
#
# Budget 25-45 minutes on Colab's 2-core VM; the release build of the dependency
# tree dominates. Everything after "cargo build" is minutes.

set -uo pipefail

REPO="${CINTX_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FEATURES="cpu,cuda,extended-device-rys"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }

# ── 1. The device ────────────────────────────────────────────────────────────
say "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "no nvidia-smi: this session has no GPU. Runtime > Change runtime type > T4 GPU."
    exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in
    *T4*) echo "T4 confirmed. Note: f64 runs at 1/32 rate here — correctness only." ;;
    *)    warn "expected a T4, found '$GPU_NAME'. The verification is still valid; \
the f64-rate caveat may differ." ;;
esac

say "CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -2
else
    warn "nvcc not on PATH; cubecl-cuda loads the driver API dynamically, so this \
is usually fine. If the build fails on a CUDA symbol, install the toolkit."
fi

# ── 2. Toolchain ─────────────────────────────────────────────────────────────
say "Rust"
if ! command -v cargo >/dev/null 2>&1; then
    echo "installing rustup (a few minutes)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
cargo --version
rustc --version

# The vendored libcint oracle is built with `cc`; Colab has one, but say so
# clearly if it does not.
command -v cc >/dev/null 2>&1 || { warn "no C compiler: the vendored libcint \
oracle cannot be built, and every parity check needs it."; exit 1; }

# ── 3. Build ─────────────────────────────────────────────────────────────────
say "Build (release, features: $FEATURES)"
cd "$REPO" || exit 1
export CINTX_ORACLE_BUILD_VENDOR=1
if ! cargo build --release -p cintx-oracle --features "$FEATURES" --tests 2>&1 | tail -20; then
    warn "build failed"
    exit 1
fi

# ── 4. Verify ────────────────────────────────────────────────────────────────
export CINTX_CUDA_ORACLE=1
STATUS=0

say "CUDA verification — host transform"
cargo test --release -p cintx-oracle --features "$FEATURES" \
    --test def2_cuda_verification -- --ignored --nocapture --test-threads=1 || STATUS=1

say "CUDA verification — M3 device transform"
CINTX_2E_TRANSFORM=device cargo test --release -p cintx-oracle --features "$FEATURES" \
    --test def2_cuda_verification -- --ignored --nocapture --test-threads=1 \
    cuda_device_transform || STATUS=1

# The launch-class coverage gate is backend-agnostic and cheap; running it here
# says whether every def2 class this device is asked for is actually accepted,
# rather than refused and silently counted as covered.
say "Device coverage"
cargo test --release -p cintx-oracle --features "$FEATURES" \
    --test def2_device_coverage -- --nocapture || STATUS=1

# ── 5. Optional: the S3 defect reproduction ──────────────────────────────────
#
# On gfx1151, *using* one shared-memory element inside the batched 2e kernel —
# unit 0 writes it, one barrier, every unit reads it — corrupts that kernel's
# output, while the identical traffic round-trips exactly in a small kernel on
# the same device. Allocating the slab without reading it is harmless. Whether
# NVIDIA shows the same behaviour is the single most useful extra datum a second
# vendor can provide, so it is offered here behind an explicit opt-in.
if [ "${CINTX_TRY_SHARED_G:-0}" = "1" ]; then
    # Clear the code-object cache first. Compiled kernels are cached by
    # signature and not by body, so a stale object can silently answer this
    # question wrongly — that is exactly how the ROCm bisect went wrong twice
    # before it was noticed.
    rm -rf "$HOME/.cache/comgr" 2>/dev/null || true
    say "S3 shared-memory G (fails on ROCm; unknown on CUDA)"
    CINTX_2E_SHARED_G=1 cargo test --release -p cintx-oracle --features "$FEATURES" \
        --test def2_cuda_verification -- --ignored --nocapture --test-threads=1 \
        cuda_int2e_matches || warn "shared-memory G failed on CUDA too — same defect class"
    say "S3 primitive isolation probe"
    cargo test --release -p cintx-cubecl --features "$FEATURES" --lib \
        shared_memory_through -- --ignored --nocapture || true
fi

say "Result"
if [ "$STATUS" -eq 0 ]; then
    echo "PASS — the CUDA backend produced libcint's numbers on this device."
    echo "Record the FMA-fusion answer and the nroots ceiling above: they decide"
    echo "whether def2-TZVP runs on NVIDIA at full angular momentum."
else
    echo "FAIL — see the output above. Paste it back; the numbers are the finding."
fi
exit "$STATUS"
