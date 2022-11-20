#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include <mkl_cblas.h>
#include <mkl_spblas.h>
#include <mkl.h>

int MAX_ITERATIONS = 5;

/*
-------------------------------------------------------------------
|                         error checking                          |
|                                                                 |
-------------------------------------------------------------------
*/

#define CHECK_STATUS(func)                                  \
    {                                                       \
        sparse_status_t status = (func);                    \
        if (status != SPARSE_STATUS_SUCCESS)                \
        {                                                   \
            printf("MKL API failed at %d with error: %d\n", \
                   __LINE__, status);                       \
            exit_status = 1;                                \
            goto exit;                                      \
        }                                                   \
    }

/*
-------------------------------------------------------------------
|                         helper functions                        |
|                                                                 |
-------------------------------------------------------------------
*/

float abs_max_diff(float *x, float *y, int N)
{
    float eps = 0.0;
    for (int k = 0; k < N; k++)
    {
        if (fabs(x[k] - y[k]) > eps)
        {
            eps = fabs(y[k] - x[k]);
        }
    }
    return eps;
}

bool action_comparison(
    float *y,
    int *enbabled_actions,
    int *adj_sidx,
    float *xnew,
    float *xold,
    int *pi,
    int N,
    float epsilon,
    float *max_eps
    ) {
    bool policy_stable = true;
    for (int r = 0; r < N; r++)
    {
        float max_value = -INFINITY;
        int argmax_a = -1;
        for (int a = 0; a < enbabled_actions[r]; a++)
        {
            if (y[adj_sidx[r] + a] > max_value)
            {
                max_value = y[adj_sidx[r] + a];
                argmax_a = a;
            }
        }
        xnew[r] = max_value;
        if (max_value - xold[r] > epsilon)
        {
            // update the policy, other wise the policy remains the same
            policy_stable = false;
            pi[r] = argmax_a;
            *max_eps = max_value - xold[r];
        }
    }
    return policy_stable;
}

/*
-------------------------------------------------------------------
|                         (sp)blas functions                      |
|                                                                 |
-------------------------------------------------------------------
(1) compute the value of the initial random policy
(2) compute the optimal policy for the product MDP
(3) Given an allocation function compute the multi-objective value
    for the allocated task to an agent
*/

int initial_policy(
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
    float epsilon)
{
    // Need to set the problem to a sequential threading problem as we will use
    // multithreading to constrol the CPU/GPU resource sharing
    //mkl_set_threading_layer(1);
    struct matrix_descr descrA;
    MKL_INT PM = pm, PN = pn, RM = rm, RN = rn;
    float alpha = 1.0, beta = 0.0, beta1 = 1.0, eps = 1.;
    int exit_status = 0;

    // Structure with sparse matrix stored in CSR format
    sparse_matrix_t csrP, csrR;
    sparse_status_t status;

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
        r_vals));

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    CHECK_STATUS(mkl_sparse_set_mv_hint(csrR, SPARSE_OPERATION_NON_TRANSPOSE,
                                        descrA, 1));

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
        p_vals));

    // ------------------------------------------------------------------
    // Set hints for the different operations before calling the mkl sparse
    // optimise function which actually does the analyse step
    // ------------------------------------------------------------------

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    CHECK_STATUS(mkl_sparse_set_mv_hint(csrP, SPARSE_OPERATION_NON_TRANSPOSE,
                                        descrA, 1));

    CHECK_STATUS(mkl_sparse_optimize(csrP));

    // This step is the R.w + P.v, because the policy has already been selected
    // before the start of this algorithm there is not need to do action comparison
    // for (int algo_k = 0; algo_k < MAX_ITERATIONS; algo_k ++) {
    int algo_k = 0;
    while (algo_k < MAX_ITERATIONS && eps > epsilon)
    {

        CHECK_STATUS(mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha,
                                     csrP, descrA, x, beta1, y));

        // Once the above has been performed the next step we need to check the convergence
        // of the epsilon value, i.e. the difference between x, and y where y has just been
        // modified by the matrix vector multiplication.
        eps = abs_max_diff(x, y, pm);

        /*
        printf("X + Y: \n");
        for (int k = 0; k < rm; k ++ ){
            printf("%4.2f, ", y[k]);
        }
        printf("\n");
        */

        printf("EPS after %i iteration: %.3f\n", algo_k, eps);
        // x <- y  update the value vector with the calculated R + Pv vector
        // y <- r  reset y to the rewards vector
        cblas_scopy(PM, y, 1, x, 1);
        cblas_scopy(RM, r_v, 1, y, 1);
        algo_k++;
    }

exit:
    mkl_sparse_destroy(csrP);
    mkl_sparse_destroy(csrR);
    return exit_status;
}

int policy_optimisation(
    const int *p_row_ptr,
    const int *p_col_ptr,
    const float *p_vals,
    const int pm,
    const int pn,
    const int *r_row_ptr,
    const int *r_col_ptr,
    const float *r_vals,
    const int rm,
    const int rn,
    float *r_v,
    const float *w,
    float *x,
    float *y,
    int *pi,
    const int *enabled_actions,
    const int *adj_sidx,
    const float epsilon,
    const int N)
{
    // Set the threading layer to be sequential
    mkl_set_threading_layer(1);
    struct matrix_descr descrA;
    MKL_INT PM = pm, PN = pn, RM = rm, RN = rn;
    float alpha = 1.0, beta = 0.0, beta1 = 1.0, eps = 1.;
    int exit_status = 0;
    bool policy_stable = false;

    float *xtmp = (float *)malloc(sizeof(float) * N);
    xtmp = memcpy(xtmp, x, sizeof(float) * N);

    // Structure with sparse matrix stored in CSR format
    sparse_matrix_t csrP, csrR;
    sparse_status_t status;

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
        r_vals));

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    CHECK_STATUS(mkl_sparse_set_mv_hint(csrR, SPARSE_OPERATION_NON_TRANSPOSE,
                                        descrA, 1));

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
        p_vals));

    // ------------------------------------------------------------------
    // Set hints for the different operations before calling the mkl sparse
    // optimise function which actually does the analyse step
    // ------------------------------------------------------------------

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    CHECK_STATUS(mkl_sparse_set_mv_hint(csrP, SPARSE_OPERATION_NON_TRANSPOSE,
                                        descrA, 1));

    CHECK_STATUS(mkl_sparse_optimize(csrP));

    // The next step is to do the computation of Rw + Pv, take note that the
    // size of x is smaller than the size of y

    // for (int k = 0; k < 4; k ++) {
    while (!policy_stable)
    {
        // Note that y = r at this point in the computation
        CHECK_STATUS(mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrP,
                                     descrA, x, 1., y));

        // Now the size of y incorporates all of the enabled actions which we are
        // required to compare
        //
        // Note that x is updated in this function

        policy_stable = action_comparison(y, enabled_actions, adj_sidx,
                                          xtmp, x, pi, N, epsilon, &eps);

        // copy the data from temp to x
        cblas_scopy(N, xtmp, 1, x, 1);
        // Reset the value of y for the computation
        cblas_scopy(rm, r_v, 1, y, 1);

        //printf("EPS: %4.3f\n", eps);
    }

exit:
    mkl_sparse_destroy(csrP);
    mkl_sparse_destroy(csrR);
    return exit_status;
}