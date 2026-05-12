/*
 * cintx-authored minimal shim for PySCF's `np_helper/np_helper.h`. NOT from
 * upstream PySCF (see ../../NOTICE).
 *
 * PySCF `pyscf/lib/gto/nr_ecp.c` and `nr_ecp_deriv.c` `#include
 * "np_helper/np_helper.h"`, but neither file actually calls any of the
 * `NPdsymm_triu` / `NPdunpack_tril` / `NPdpack_tril` / `NPdtranspose` /
 * `NPomp_*` symbols declared in the upstream header. Forward declarations
 * are provided here for compilation parity; the matching `np_helper`
 * implementation in upstream PySCF is intentionally NOT vendored — the
 * symbols are not link-dependencies of cintx-oracle's PySCF nr_ecp build.
 *
 * If a future cintx integration calls any of these symbols, link against a
 * real PySCF `np_helper` build or replace this shim with the upstream file
 * plus its companion C sources.
 */
#ifndef CINTX_VENDOR_PYSCF_NP_HELPER_H
#define CINTX_VENDOR_PYSCF_NP_HELPER_H

#include <stdint.h>
#include <complex.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* The NumPy-style int aliases that some PySCF C code expects. */
typedef int32_t NPY_INT32;
typedef int64_t NPY_INT64;

/* Forward declarations only — not called from nr_ecp.{c,_deriv.c}, so the
 * matching implementations are deliberately not vendored. */
void NPdsymm_triu(int n, double *mat, int hermi);
void NPzhermi_triu(int n, double complex *mat, int hermi);
void NPdunpack_tril(int n, double *tril, double *mat, int hermi);
void NPzunpack_tril(int n, double complex *tril, double complex *mat,
                    int hermi);
void NPdpack_tril(int n, double *tril, double *mat);
void NPzpack_tril(int n, double complex *tril, double complex *mat);

#if defined(__cplusplus)
}
#endif

#endif /* CINTX_VENDOR_PYSCF_NP_HELPER_H */
