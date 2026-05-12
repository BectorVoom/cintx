/*
 * cintx-authored minimal `dgemm_` shim. NOT from upstream PySCF
 * (see ../NOTICE).
 *
 * Provides a reference Fortran BLAS `dgemm_` implementation sufficient to
 * satisfy the linker symbol referenced by `nr_ecp.c`. The implementation
 * is correctness-first: row-major-of-column-major triple loop, no
 * blocking or threading. Performance-critical use should relink against a
 * system BLAS (libopenblas / libblas), which provides a more efficient
 * `dgemm_` with identical observable behavior; cintx parity tests
 * compare against the byte-identity output of the algorithm in nr_ecp.c,
 * not the choice of BLAS backend.
 *
 * BLAS Fortran convention (column-major):
 *   C := alpha * op(A) * op(B) + beta * C
 *   where op(X) = X if transX = 'N', X^T if transX = 'T' or 'C'.
 *   A is (lda x k_or_m), B is (ldb x n_or_k), C is (ldc x n).
 *
 * Per the calls in nr_ecp.c only 'N' and 'T' transposes are used.
 */

#include <ctype.h>
#include <stddef.h>
#include <string.h>

/*
 * cintx-authored `NPdset0` implementation. Called by PySCF nr_ecp.c and
 * nr_ecp_deriv.c to zero-fill a double array (line 6253 of nr_ecp.c,
 * multiple sites in nr_ecp_deriv.c). The PySCF upstream implementation
 * in pyscf/lib/np_helper/np_helper.c is a thin memset wrapper; we
 * inline it here to keep the link surface self-contained.
 */
void NPdset0(double *p, const size_t n) {
    memset(p, 0, n * sizeof(double));
}

static int dgemm_shim_trans(const char *t) {
    /* Returns 0 for 'N'/'n', 1 for 'T'/'t'/'C'/'c'. */
    if (t == 0) return 0;
    char c = *t;
    return (c == 'N' || c == 'n') ? 0 : 1;
}

void dgemm_(const char *transa, const char *transb,
            const int  *m,      const int  *n,      const int *k,
            const double *alpha, const double *a,   const int *lda,
                                 const double *b,   const int *ldb,
            const double *beta,        double *c,   const int *ldc) {
    const int M    = *m;
    const int N    = *n;
    const int K    = *k;
    const int LDA  = *lda;
    const int LDB  = *ldb;
    const int LDC  = *ldc;
    const double alpha_v = *alpha;
    const double beta_v  = *beta;
    const int ta = dgemm_shim_trans(transa);
    const int tb = dgemm_shim_trans(transb);

    /* C := beta * C  (handle beta == 0 cleanly to mirror BLAS contract). */
    if (beta_v == 0.0) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                c[i + j * LDC] = 0.0;
            }
        }
    } else if (beta_v != 1.0) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                c[i + j * LDC] *= beta_v;
            }
        }
    }

    if (alpha_v == 0.0 || K == 0) {
        return;
    }

    /* C := C + alpha * op(A) * op(B). Column-major. */
    if (ta == 0 && tb == 0) {
        /* C[i,j] += alpha * sum_l A[i,l] * B[l,j] */
        for (int j = 0; j < N; j++) {
            for (int l = 0; l < K; l++) {
                double b_lj = b[l + j * LDB];
                if (b_lj == 0.0) continue;
                double a_scaled = alpha_v * b_lj;
                for (int i = 0; i < M; i++) {
                    c[i + j * LDC] += a_scaled * a[i + l * LDA];
                }
            }
        }
    } else if (ta == 1 && tb == 0) {
        /* C[i,j] += alpha * sum_l A[l,i] * B[l,j] */
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                double sum = 0.0;
                for (int l = 0; l < K; l++) {
                    sum += a[l + i * LDA] * b[l + j * LDB];
                }
                c[i + j * LDC] += alpha_v * sum;
            }
        }
    } else if (ta == 0 && tb == 1) {
        /* C[i,j] += alpha * sum_l A[i,l] * B[j,l] */
        for (int j = 0; j < N; j++) {
            for (int l = 0; l < K; l++) {
                double b_jl = b[j + l * LDB];
                if (b_jl == 0.0) continue;
                double a_scaled = alpha_v * b_jl;
                for (int i = 0; i < M; i++) {
                    c[i + j * LDC] += a_scaled * a[i + l * LDA];
                }
            }
        }
    } else {
        /* ta == 1 && tb == 1: C[i,j] += alpha * sum_l A[l,i] * B[j,l] */
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                double sum = 0.0;
                for (int l = 0; l < K; l++) {
                    sum += a[l + i * LDA] * b[j + l * LDB];
                }
                c[i + j * LDC] += alpha_v * sum;
            }
        }
    }
}
