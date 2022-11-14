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
}


float test_blas_routine(void) {
    mkl_set_threading_layer(1);
    MKL_INT  n, incx, incy, i;
    float   *x, *y;
    float    res;
    MKL_INT  len_x, len_y;

    n = 5;
    incx = 2;
    incy = 1;

    len_x = 1+(n-1)*abs(incx);
    len_y = 1+(n-1)*abs(incy);
    x    = (float *)calloc( len_x, sizeof( float ) );
    y    = (float *)calloc( len_y, sizeof( float ) );
    if( x == NULL || y == NULL ) {
        printf( "\n Can't allocate memory for arrays\n");
        return 1;
    }

    for (i = 0; i < n; i++) {
        x[i*abs(incx)] = 2.0;
        y[i*abs(incy)] = 1.0;
    }

    res = cblas_sdot(n, x, incx, y, incy);

    printf("\n       SDOT = %7.3f\n", res);

    free(x);
    free(y);

    return res;
}


int test_mv(int *row_ptr, int *cols, float *vals, int m, int n, float *x, float *y) {
    mkl_set_threading_layer(1);
    // Descriptor of main sparse matrix properties
    struct matrix_descr descrA;
    MKL_INT M = m, N = n;
    double alpha = 1.0, beta = 0.0;
    int exit_status = 0;

    // Structure with sparse matrix stored in CSR format
    sparse_matrix_t       csrA;
    sparse_status_t     status;

    CHECK_STATUS(mkl_sparse_s_create_csr(&csrA,
        SPARSE_INDEX_BASE_ZERO,
        M, 
        N,
        row_ptr,
        row_ptr + 1,
        cols,
        vals
    ));

    // Set hints for the different operations before calling the mkl_sparse_optimise()
    // api which actually does the analyse step 

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    CHECK_STATUS(mkl_sparse_set_mv_hint(csrA, SPARSE_OPERATION_NON_TRANSPOSE, descrA, 1));

    /*if (status != SPARSE_STATUS_SUCCESS && status != SPARSE_STATUS_NOT_SUPPORTED) {
        printf("Error in set hints: mkl_sparse_set_mv_hint: %d \n", status);
    }*/

    CHECK_STATUS(mkl_sparse_optimize(csrA));

    CHECK_STATUS(mkl_sparse_s_mv( SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, 
                                  descrA, x, beta, y));

    for (int i = 0; i < M; i++) {
        printf("%7.3f, ", y[i]);
    }
    printf("\n");

    exit:
        mkl_sparse_destroy(csrA);

    return exit_status;
}
