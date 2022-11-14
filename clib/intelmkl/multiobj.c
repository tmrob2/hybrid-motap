#include <stdio.h>
#include <stdlib.h>

#include <mkl_cblas.h>
#include <mkl_spblas.h>
#include <mkl.h>

#define CHECK_STATUS(func)                              \
{                                                       \
    sparse_status_t status = (func);                    \
    if (status != SPARSE_STATUS_SUCCESS) {              \
        printf("MKL API failed at %d with error: %d\n", \
               __LINE__, status);                       \
        exit_status = 1;                                \
        goto exit;                                      \
    }                                                   \
}                                                       \

float compute_max_diff(float *x, float *y, int N) {
    float eps = 0.;
    for (int k = 0; k < N; k++) {
        if (y[k] - x[k] > eps) {
            eps = y[k] - x[k];
        }
    }
    return eps;
}

int intial_policy(
    int *p_row_ptr,
    int *p_col_ptr,
    float *p_vals,
    int pm,
    int pn,
    int *r_row_ptr,
    int *r_col_ptr,
    float *r_vals,
    int rm,
    int rn,
    float *r_v,
    float *w,
    float *x, 
    float *y,
    float epsilon
) {
    // Need to set the problem to a sequential threading problem as we will use
    // multithreading to constrol the CPU/GPU resource sharing
    mkl_set_threading_layer(1);
    struct matrix_descr descrA;
    MKL_INT PM = pm, PN = pn, RM = rm, RN = rn;
    float alpha = 1.0, beta = 0.0, beta1 = 0.0, eps = 0.;
    int exit_status = 0;


    // Structure with sparse matrix stored in CSR format
    sparse_matrix_t       csrP, csrR;
    sparse_status_t           status;

    // ------------------------------------------------------------------
    // Construct the rewards sparse matrix in csr format pre hints
    // ------------------------------------------------------------------

    CHECK_STATUS(mkl_sparse_s_create_csr(
        &csrR,
        SPARSE_INDEX_BASE_ZERO,
        RM, 
        RN,
        r_row_ptr,
        r_row_ptr + 1,
        r_col_ptr,
        r_vals
    ));

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    CHECK_STATUS(mkl_sparse_set_mv_hint(csrR, SPARSE_OPERATION_NON_TRANSPOSE, descrA, 1));

    CHECK_STATUS(mkl_sparse_optimize(csrR));

    // ------------------------------------------------------------------
    // Compute R.w as this will only occur once
    // ------------------------------------------------------------------
    CHECK_STATUS(mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrR, 
                                 descrA, w, beta, r_v));

    cblas_scopy(RM, r_v, 1, y, 1);

    // ------------------------------------------------------------------
    // Construct the transition sparse matrix in csr format pre hints
    // ------------------------------------------------------------------

    CHECK_STATUS(mkl_sparse_s_create_csr(
        &csrP,
        SPARSE_INDEX_BASE_ZERO,
        PM,
        PN,
        p_row_ptr,
        p_row_ptr + 1,
        p_col_ptr,
        p_vals
    ));

    // ------------------------------------------------------------------
    // Set hints for the different operations before calling the mkl sparse
    // optimise function which actually does the analyse step
    // ------------------------------------------------------------------

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    CHECK_STATUS(mkl_sparse_set_mv_hint(csrP, SPARSE_OPERATION_NON_TRANSPOSE, descrA, 1));

    CHECK_STATUS(mkl_sparse_optimize(csrP));

    // This step is the R.w + P.v, because the policy has already been selected
    // before the start of this algorithm there is not need to do action comparison
    CHECK_STATUS(mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrP, descrA, 
                                 x, beta1, y));

    // Once the above has been performed the next step we need to check the convergence
    // of the epsilon value, i.e. the difference between x, and y where y has just been
    // modified by the matrix vector multiplication. 
    eps = compute_max_diff(x, y, pm);

    printf("EPS after 1 iteration: %.3f\n", eps);

    exit:
        mkl_sparse_destroy(csrP);
        mkl_sparse_destroy(csrR);
    return exit_status;
}