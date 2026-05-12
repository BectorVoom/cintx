/*
 * cintx-authored minimal shim for PySCF's `vhf/fblas.h`. NOT from upstream
 * PySCF (see ../../NOTICE).
 *
 * PySCF `pyscf/lib/gto/nr_ecp.c` and `nr_ecp_deriv.c` `#include "vhf/fblas.h"`
 * to obtain Fortran BLAS extern "C" declarations. The only BLAS symbol
 * actually called from those two files is `dgemm_`; the rest of the upstream
 * `vhf/fblas.h` declarations are forwarded here for compilation parity but
 * do not need to be linked.
 *
 * The real `dgemm_` implementation is provided by `src/dgemm_shim.c` (a
 * minimal cintx-authored reference implementation) when no system BLAS is
 * available. Tests that require high-performance BLAS may relink against
 * a system libblas at the cintx-oracle linker boundary; this header stays
 * the same.
 */
#ifndef CINTX_VENDOR_PYSCF_VHF_FBLAS_H
#define CINTX_VENDOR_PYSCF_VHF_FBLAS_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <complex.h>

double dasum_(const int *n, const double *dx, const int *incx);
void   dscal_(const int *n, const double *da, double *dx, const int *incx);
void   daxpy_(const int *n, const double *da, const double *dx,
              const int *incx, double *dy, const int *incy);
double ddot_(const int *n, const double *dx, const int *incx,
             const double *dy, const int *incy);
void   dcopy_(const int *n, const double *dx, const int *incx,
              double *dy, const int *incy);

/* Primary symbol used by nr_ecp.c — provided by src/dgemm_shim.c when no
 * system BLAS is linked. */
void   dgemm_(const char *transa, const char *transb,
              const int  *m,      const int  *n,      const int *k,
              const double *alpha, const double *a,   const int *lda,
                                   const double *b,   const int *ldb,
              const double *beta,        double *c,   const int *ldc);

void   dgemv_(const char *trans,
              const int  *m,      const int  *n,
              const double *alpha, const double *a,   const int *lda,
                                   const double *x,   const int *incx,
              const double *beta,        double *y,   const int *incy);

void   dger_(const int *m, const int *n,
             const double *alpha,
             const double *x, const int *incx,
             const double *y, const int *incy,
             double *a, const int *lda);

void   dsymm_(const char *side, const char *uplo,
              const int *m, const int *n,
              const double *alpha, const double *a, const int *lda,
                                   const double *b, const int *ldb,
              const double *beta,        double *c, const int *ldc);

#if defined(__cplusplus)
}
#endif

#endif /* CINTX_VENDOR_PYSCF_VHF_FBLAS_H */
