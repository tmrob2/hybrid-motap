#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cublas_v2.h"
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

//int MAX_ITERATIONS = 1000;
//const int MAX_UNSTABLE = 30;
/*
#######################################################################
#                           KERNELS                                   #
#######################################################################
*/

__global__ void max_value(
    float *y,
    int *enabled_actions,
    int *adj_sidx,
    float *xnew,
    float *xold,
    int *pi,
    float *stable,
    float epsilon,
    int N
    ) {
    // The purpose of this kernel is to do effective row-wise comparison of values
    // to determine the new policy and the new value vector without copy of 
    // data from the GPU to CPU
    //
    // It is recognised that this code will be slow due to memory segmentation
    // and cache access, but this should in theory be faster then sending data
    // back and forth between the GPU and CPU
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float max_value = -INFINITY;
        int argmax_a = -1;
        for (int k = 0; k < enabled_actions[tid]; k++) {
            if (y[adj_sidx[tid] + k] > max_value) {
                max_value = y[adj_sidx[tid] + k];
                argmax_a = k;
            }
        }
        xnew[tid] = max_value;
        stable[tid] = 0.;
        if (max_value - xold[tid] > epsilon) {
            stable[tid] = 1.0;
            pi[tid] = argmax_a;
        }
    }
}

__global__ void abs_diff(float *a, float *b, float *c, int *unstable, int m, int max_unstable) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // HANDLE THE DATA AT THIS INDEX
    if (tid < m) {
        // compute the absolute diff between two elems
        if (fabsf(b[tid] - a[tid]) < c[tid] || a[tid] == 0.) {
            unstable[tid] = 0;
        } else {
            unstable[tid]++;
        }
        c[tid] = fabsf(b[tid] - a[tid]);
        if (unstable[tid] > max_unstable && a[tid] < 0) {
            a[tid] = -INFINITY;
        }
    } 
}

__global__ void mobj_abs_diff(float *x, float *y, float *eps_capture, int *unstable, 
    int obj, int N, int max_unstable) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        if (fabsf(x[tid] - y[tid]) < eps_capture[obj * N + tid] || y[tid] == 0.) {
            unstable[obj * N + tid] = 0;
        } else {
            unstable[obj * N + tid]++;
        }
        eps_capture[obj * N + tid] = fabs(x[tid] - y[tid]);
        if (unstable[obj * N + tid] > max_unstable && y[tid] < 0) {
            y[tid] = -INFINITY;
        }
    }
}

__global__ void change_elem(float *arr, int idx, int val) {
    arr[idx] = val;
}

__global__ void copy_elems(float *dest, int begin_idx, float *src, int begin_cp, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        dest[tid + begin_idx] = src[tid + begin_cp];
    }
}

void copy_elems_launcher(float *dest, int begin_idx, float *src, int begin_cp, int N) {
    int blockSize = 0;    // The launch configurator returned block size
    int minGridSize;  // The maximum grid size needed to achieve max
                      // maximum occupancy
    int gridSize;     // The grid size needed, based on the input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max_value, 0, 0);

    // Round up according to array size
    gridSize = (N + blockSize - 1) / blockSize;

    copy_elems<<<gridSize, blockSize>>>(dest, begin_idx, src, begin_cp, N);
}

void max_value_launcher(float *y, int*enabled_actions, int *adj_sidx, float *xnew,
    float *xold, int *pi, float *stable, float epsilon, int N
) {
    int blockSize = 0;    // The launch configurator returned block size
    int minGridSize;  // The maximum grid size needed to achieve max
                      // maximum occupancy
    int gridSize;     // The grid size needed, based on the input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max_value, 0, 0);

    // Round up according to array size
    gridSize = (N + blockSize - 1) / blockSize;

    max_value<<<gridSize, blockSize>>>(y, enabled_actions, adj_sidx, xnew, xold, pi,
        stable, epsilon, N
    );
}

void mobj_abs_diff_launcher(float *x, float*y, float *eps_capture, int *unstable, 
    int obj, int N, int max_unstable) {
    int blockSize = 0;    // The launch configurator returned block size
    int minGridSize = 0;  // The maximum grid size needed to achieve max
                      // maximum occupancy
    int gridSize;     // The grid size needed, based on the input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, max_value, 0, 0);

    // Round up according to array size
    gridSize = (N + blockSize - 1) / blockSize;

    mobj_abs_diff<<<gridSize, blockSize>>>(x, y, eps_capture, unstable, obj, N, max_unstable);
}

void abs_diff_launcher(float *a, float *b, float* c, int *unstable, int m, int max_unstable) {
    int blockSize = 0;    // The launch configurator returned block size
    int minGridSize;  // The maximum grid size needed to achieve max
                      // maximum occupancy
    int gridSize;     // The grid size needed, based on the input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, abs_diff, 0, 0);

    // Round up according to array size
    gridSize = (m + blockSize - 1) / blockSize;

    abs_diff<<<gridSize, blockSize>>>(a, b, c, unstable, m, max_unstable);
}

/*
#######################################################################
#                              CUDA                                   #
#######################################################################
*/

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

extern "C" {

int warm_up_gpu() {
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    cublasHandle_t blashandle;
    CHECK_CUBLAS(cublasCreate(&blashandle));

    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUBLAS(cublasDestroy(blashandle));
    return 0.;
}

int initial_policy_value(
    int pm,
    int pn,
    int pnz,
    int * pi,
    int * pj,
    float * px,
    int pi_size,
    int rm,
    int rn,
    int rnz,
    int *ri,
    int *rj,
    float *rx,
    int ri_size,
    float *x,
    float *y,
    float *w,
    float *rmv,
    int *unstable,
    float eps,
    int max_iter,
    int max_unstable
    ) {
    /* 
    Get the COO matrix into sparsescoo fmt

    Then multiply the COO by the initial value vector

    The rewards matrix is also sparse so it will need a sparse matrix descr
    as well. Multiply R by a repeated weight vector in the number 
    of prods and actions

    Finally sum the result

    This should happen in a loop until convergence

    I also want to do some wall timing to see some statistics on 
    the GPU 
    */
    // build the sparse transition matrix first

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    cublasHandle_t blashandle;
    CHECK_CUBLAS(cublasCreate(&blashandle));


    cusparseSpMatDescr_t descrP = NULL;
    cusparseSpMatDescr_t descrR = NULL;

    // allocated the device memory for the COO matrix

    // ----------------------------------------------------------------
    //                       Transition Matrix
    // ----------------------------------------------------------------

    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dPCsrRowPtr, *dPCsrColPtr;
    float *dPCsrValPtr;

    // allocate device memory to store the sparse CSR 
    cudaMalloc((void **)&dPCsrValPtr, sizeof(float) * pnz);
    cudaMalloc((void **)&dPCsrRowPtr, sizeof(int) * pi_size);
    cudaMalloc((void **)&dPCsrColPtr, sizeof(int) * pnz);

    cudaMemcpy(dPCsrValPtr, px, sizeof(float) * pnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dPCsrColPtr, pj, sizeof(int) * pnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dPCsrRowPtr, pi, sizeof(int) * pi_size, cudaMemcpyHostToDevice);
    
    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrP, // MATRIX DESCRIPTION
        pm, // NUMBER OF ROWS
        pn, // NUMBER OF COLS
        pnz, // NUMBER OF NON ZERO VALUES
        dPCsrRowPtr, // ROWS OFFSETS
        dPCsrColPtr, // COL INDICES
        dPCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));
    
    // ----------------------------------------------------------------
    //                       Rewards Matrix
    // ----------------------------------------------------------------
    
    int *dRCsrRowPtr, *dRCsrColPtr;
    float *dRCsrValPtr;

    // allocate device memory to store the sparse CSR 
    cudaMalloc((void **)&dRCsrValPtr, sizeof(float) * rnz);
    cudaMalloc((void **)&dRCsrRowPtr, sizeof(int) * ri_size);
    cudaMalloc((void **)&dRCsrColPtr, sizeof(int) * rnz);
    cudaMemcpy(dRCsrValPtr, rx, sizeof(float) * rnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dRCsrColPtr, rj, sizeof(int) * rnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dRCsrRowPtr, ri, sizeof(int) * ri_size, cudaMemcpyHostToDevice);

    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrR, // MATRIX DESCRIPTION
        rm, // NUMBER OF ROWS
        rn, // NUMBER OF COLS
        rnz, // NUMBER OF NON ZERO VALUES
        dRCsrRowPtr, // ROWS OFFSETS
        dRCsrColPtr, // COL INDICES
        dRCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));

    // ----------------------------------------------------------------
    //                      Start of VI
    // ----------------------------------------------------------------

    // --------------TRANSITION MATRIX MULTIPLICATION SETUP------------
    
    float alpha = 1.0;
    float beta = 1.0;
    float *epsilon = (float*) malloc(pm * sizeof(float));
    //int iepsilon;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecX, vecY;
    float *dX, *dY, *dZ, *dStaticY, *dOutput;
    int *dUnstable;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Allocate the device memory
    cudaMalloc((void**)&dX, pm * sizeof(float));
    cudaMalloc((void**)&dOutput, pm * sizeof(float));
    cudaMalloc((void**)&dY, pm * sizeof(float));
    cudaMalloc((void**)&dZ, pm * sizeof(float));
    cudaMalloc((void**)&dStaticY, pm * sizeof(float));
    cudaMalloc((void**)&dUnstable, pm * sizeof(float));
    //cudaMalloc((void**)&d_eps, sizeof(float));

    // create a initial Y vector
    float *static_y = (float*) calloc(pm, sizeof(float));
    
    // copy the vector from host memory to device memory
    cudaMemcpy(dX, x, pn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dStaticY, static_y, pm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dUnstable, unstable, pm * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    cusparseCreateDnVec(&vecX, pn, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, pm, dY, CUDA_R_32F);

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    
    // --------------REWARDS MATRIX MULTIPLICATION SETUP---------------

    float alphaR = 1.0;
    float betaR = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecW, vecRMv;
    float *dRw, *dRMv;
    void* dBufferR = NULL;
    size_t bufferSizeR = 0;

    //float *rmv = (float*) calloc(rm, sizeof(float));

    cudaMalloc((void**)&dRw, rn * sizeof(float));
    cudaMalloc((void**)&dRMv, rm * sizeof(float));

    // copy the vector from host memory to device memory
    cudaMemcpy(dRw, w, rn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dRMv, rmv, rm * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(dRstaticMx, rmv, rm * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    cusparseCreateDnVec(&vecW, rn, dRw, CUDA_R_32F);
    cusparseCreateDnVec(&vecRMv, rm, dRMv, CUDA_R_32F);

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSizeR);
    cudaMalloc(&dBufferR, bufferSizeR);

    // ALGORITHM LOOP

    // Copy the zero vector to initialise Y -> captures A.x result 
    // for transition matrix
    //csparseDnVecSetValues(vecY, dY);
    //cublasScopy(blashandle, pm, dYStatic, 1, dY, 1);
    // copy the static Y vector to initialise Y
    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBufferR));

    float maxeps;
    maxeps = 0.0f;

    for (int algo_i = 0; algo_i < max_iter; algo_i ++) {

        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F, 
            CUSPARSE_MV_ALG_DEFAULT, dBuffer));

        // push this into the algorithm loop


        // ---------------------SUM DENSE VECTORS-------------------------

        /* 
        The gpu memory should already be allocated, i.e. we are summing
        dY + dRMv
        */
        CHECK_CUBLAS(cublasSaxpy(blashandle, pm, &alpha, dRMv, 1, dY, 1));
        
        // ---------------------COMPUTE EPSILON---------------------------

        // what is the difference between dY and dX

        // EPSILON COMPUTATION
        abs_diff_launcher(dY, dX, dZ, dUnstable, pm, max_unstable);
        //CHECK_CUBLAS(cublasIsamax(blashandle, pm, dZ, 1, &iepsilon));

        thrust::device_ptr<float> dev_ptr(dZ);
        maxeps = *thrust::max_element(thrust::device, dev_ptr, dev_ptr + pm);

        CHECK_CUBLAS(cublasScopy(blashandle, pm, dY, 1, dX, 1));
        // RESET Y
        CHECK_CUBLAS(cublasScopy(blashandle, pm, dStaticY, 1, dY, 1));
        //std::cout << "EPS_TEST " << "THRUST "<< maxeps << std::endl;
        if (maxeps < eps || isnan(maxeps)) {
            //printf("INITIAL POLICY GENERATED; EPS TOL REACHED in %i ITERATIONS\n", algo_i);
            break;
        }
    }
    
    CHECK_CUDA(cudaMemcpy(y, dX, pm *sizeof(float), cudaMemcpyDeviceToHost));
    
    //cudaMemcpy(rmv, dRMv, rm *sizeof(float), cudaMemcpyDeviceToHost);
    //destroy the vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(descrP));
    CHECK_CUSPARSE(cusparseDestroySpMat(descrR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecRMv));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecW));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUBLAS(cublasDestroy(blashandle));

    // Free the device memory
    CHECK_CUDA(cudaFree(dPCsrColPtr));
    CHECK_CUDA(cudaFree(dPCsrRowPtr));
    CHECK_CUDA(cudaFree(dPCsrValPtr));
    CHECK_CUDA(cudaFree(dRCsrColPtr));
    CHECK_CUDA(cudaFree(dRCsrRowPtr));
    CHECK_CUDA(cudaFree(dRCsrValPtr));
    //cudaFree(d_eps);
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dStaticY));
    CHECK_CUDA(cudaFree(dUnstable));
    CHECK_CUDA(cudaFree(dZ));
    CHECK_CUDA(cudaFree(dRw));
    CHECK_CUDA(cudaFree(dRMv));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dBufferR));
    free(epsilon);
    free(static_y);
    return 0;
}

int policy_optimisation(
    int *Pi, // SIZE OF THE INIT POLICY WILL BE P.M
    const int pm,    // TRANSITION COL NUMBER
    const int pn,    // TRANSITION ROW NUMBER 
    const int pnz,   // TRANSITION NON ZERO VALUES
    const int *pi,   // TRANSITION ROW PTR CSR
    const int *pj,   // TRANSITION COL VECTOR CSR
    const float *px, // TRANSITION VALUE VECTOR
    const int rm,    // REWARDS VALUE ROW NUMBER
    const int rn,    // REWARDS VALUE COLS NUMBER
    const int rnz,   // REWARDS NON ZERO VALUES
    const int *ri,   // REWARDS MATRIX ROW PTR CSR
    const int *rj,   // REWARDS MATRIX COL VECTOR CSR
    const float *rx, // REWARDS MATRIX VALUE VECTOR
    float *x,  // Assumes that x is set to the initial value
    float *y,  // TMP ACC VALUE VECTOR
    float *rmv, // initial R vec
    const float *w,  // REPEATED WEIGHT VECTOR
    const float eps,  // THRESHOLD
    int block_size,
    const int *enabled_actions,
    const int *adj_sidx,
    const float *stable,
    int max_iter
){
    /*
    This function is the second part of the value iteration implementation
    */
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    cublasHandle_t blashandle;
    cublasCreate(&blashandle);
    cusparseSpMatDescr_t descrP = NULL;
    cusparseSpMatDescr_t descrR = NULL;

    // ----------------------------------------------------------------
    //                             POLICY
    // ----------------------------------------------------------------

    int *PI, *EnabledActions, *AdjSIDX;
    float *dStable;
    CHECK_CUDA(cudaMalloc((void**)&PI, block_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(PI, Pi, block_size * sizeof(int), cudaMemcpyHostToDevice));
    //
    CHECK_CUDA(cudaMalloc((void**)&EnabledActions, block_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(EnabledActions, enabled_actions, block_size * sizeof(int), cudaMemcpyHostToDevice));
    //
    CHECK_CUDA(cudaMalloc((void**)&AdjSIDX, block_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(AdjSIDX, adj_sidx, block_size * sizeof(int), cudaMemcpyHostToDevice));
    //
    CHECK_CUDA(cudaMalloc((void**)&dStable, block_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dStable, stable, block_size * sizeof(float), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------------
    //                       Transition Matrix
    // ----------------------------------------------------------------

    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dPCsrRowPtr, *dPCsrColPtr;
    float *dPCsrValPtr;

    // allocate device memory to store the sparse CSR 
    CHECK_CUDA(cudaMalloc((void **)&dPCsrValPtr, sizeof(float) * pnz));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrRowPtr, sizeof(int) * (pm + 1)));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrColPtr, sizeof(int) * pnz));

    CHECK_CUDA(cudaMemcpy(dPCsrValPtr, px, sizeof(float) * pnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dPCsrColPtr, pj, sizeof(int) * pnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dPCsrRowPtr, pi, sizeof(int) * (pm + 1), cudaMemcpyHostToDevice));
    
    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrP, // MATRIX DESCRIPTION
        pm, // NUMBER OF ROWS
        pn, // NUMBER OF COLS
        pnz, // NUMBER OF NON ZERO VALUES
        dPCsrRowPtr, // ROWS OFFSETS
        dPCsrColPtr, // COL INDICES
        dPCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));

    // ----------------------------------------------------------------
    //                       Rewards Matrix
    // ----------------------------------------------------------------
    
    int *dRCsrRowPtr, *dRCsrColPtr;
    float *dRCsrValPtr;

    // allocate device memory to store the sparse CSR 
    CHECK_CUDA(cudaMalloc((void **)&dRCsrValPtr, sizeof(float) * rnz));
    CHECK_CUDA(cudaMalloc((void **)&dRCsrRowPtr, sizeof(int) * (rm + 1)));
    CHECK_CUDA(cudaMalloc((void **)&dRCsrColPtr, sizeof(int) * rnz));
    CHECK_CUDA(cudaMemcpy(dRCsrValPtr, rx, sizeof(float) * rnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRCsrColPtr, rj, sizeof(int) * rnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRCsrRowPtr, ri, sizeof(int) * (rm + 1), cudaMemcpyHostToDevice)); 

    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrR, // MATRIX DESCRIPTION
        rm, // NUMBER OF ROWS
        rn, // NUMBER OF COLS
        rnz, // NUMBER OF NON ZERO VALUES
        dRCsrRowPtr, // ROWS OFFSETS
        dRCsrColPtr, // COL INDICES
        dRCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));

    // ----------------------------------------------------------------
    //                      Start of VI
    // ----------------------------------------------------------------

    // --------------TRANSITION MATRIX MULTIPLICATION SETUP------------
    float alpha = 1.0;
    float beta = 1.0;
    float policy_stable = 0.;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecX, vecY;
    //
    float *dX, *dXtmp, *dY, *dStaticY; 
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    CHECK_CUDA(cudaMalloc((void**)&dX, block_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dY, pm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dXtmp, block_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dStaticY, pm * sizeof(float)));
    //cudaMalloc((void**)&d_eps, sizeof(float));

    // create a initial Y vector
    //float *static_y = (float*) calloc(pm, sizeof(float));
    
    // copy the vector from host memory to device memory
    CHECK_CUDA(cudaMemcpy(dY, y, pm * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dStaticY, y, pm * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dX, x, block_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dXtmp, x, block_size * sizeof(float), cudaMemcpyHostToDevice));

    // create a dense vector on device memory
    // printf("block size: %i\n", block_size);
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, block_size, dX, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, pm, dY, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    
    // --------------REWARDS MATRIX MULTIPLICATION SETUP---------------

    float alphaR = 1.0;
    float betaR = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecW, vecRMv;
    float *dRw, *dRMv, *dRstaticMx;
    void* dBufferR = NULL;
    size_t bufferSizeR = 0;

    //float *rmv = (float*) calloc(rm, sizeof(float));

    CHECK_CUDA(cudaMalloc((void**)&dRw, rn * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dRMv, rm * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dRstaticMx, rm * sizeof(float)));

    // copy the vector from host memory to device memory
    CHECK_CUDA(cudaMemcpy(dRw, w, rn * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRMv, rmv, rm  * sizeof(float), cudaMemcpyHostToDevice));
    //cudaMemcpy(dRstaticMx, rmv, rm * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecW, rn, dRw, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecRMv, rm, dRMv, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSizeR));
    CHECK_CUDA(cudaMalloc(&dBufferR, bufferSizeR));

    // ONE OFF REWARDS COMPUTATION

    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBufferR));
    
    // ALGORITHM LOOP - POLICY GENERATION
    for (int algo_i = 0; algo_i < max_iter; algo_i ++) {

        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F, 
            CUSPARSE_MV_ALG_DEFAULT, dBuffer));    

        // ---------------------SUM DENSE VECTORS-------------------------

        /* 
        i.e. we are summing dY + dRMv
        */
        
        CHECK_CUBLAS(cublasSaxpy(blashandle, pm, &alpha, dRMv, 1, dY, 1));
        // ------------------COMPUTE POLICY STABLE------------------------
        
        max_value_launcher(dY, EnabledActions, AdjSIDX, dXtmp, dX, PI, dStable, 
                           eps, block_size);
        
        // we can compute if the policy is stable with cublas 
        CHECK_CUBLAS(cublasScopy(blashandle, block_size, dXtmp, 1, dX, 1));
        
        cublasSasum(blashandle, block_size, dStable, 1, &policy_stable);
        if (policy_stable == 0) {
            break;
        }
        CHECK_CUBLAS(cublasScopy(blashandle, pm, dStaticY, 1, dY, 1));
    }

    CHECK_CUDA(cudaMemcpy(x, dX, block_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(Pi, PI, block_size * sizeof(int), cudaMemcpyDeviceToHost));
    

    // MEMORY MANAGEMENT
    //destroy the vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(descrP));
    CHECK_CUSPARSE(cusparseDestroySpMat(descrR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecRMv));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecW));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUBLAS(cublasDestroy(blashandle));

    // Free the device memory
    CHECK_CUDA(cudaFree(dPCsrColPtr));
    CHECK_CUDA(cudaFree(dPCsrRowPtr));
    CHECK_CUDA(cudaFree(dPCsrValPtr));
    CHECK_CUDA(cudaFree(dRCsrColPtr));
    CHECK_CUDA(cudaFree(dRCsrRowPtr));
    CHECK_CUDA(cudaFree(dRCsrValPtr));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dXtmp));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dStaticY));
    CHECK_CUDA(cudaFree(dStable));
    CHECK_CUDA(cudaFree(AdjSIDX));
    CHECK_CUDA(cudaFree(EnabledActions));
    CHECK_CUDA(cudaFree(dRw));
    CHECK_CUDA(cudaFree(dRMv));
    CHECK_CUDA(cudaFree(dRstaticMx));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dBufferR));
    CHECK_CUDA(cudaFree(PI));
    //free(epsilon);
    return 0;
}

int policy_value_stream(
    int p_init_m,
    int p_init_n,
    int p_init_nz,
    int *p_init_i,
    int *p_init_j,
    float *p_init_x,
    int p_init_i_size,
    int p_m,
    int p_n,
    int p_nz,
    int *p_i,
    int *p_j,
    float *p_x,
    int p_i_size,
    int r_init_m,
    int r_init_n,
    int r_init_nz,
    int *r_init_i,
    int *r_init_j,
    float *r_init_x,
    int r_init_i_size,
    int r_m,
    int r_n,
    int r_nz,
    int *r_i,
    int *r_j,
    float *r_x,
    int r_i_size,
    float *x_init,
    float *y_init,
    float *w_init,
    float *rmv_init,
    float *y, 
    float *rmv,
    float *w,
    int *unstable,
    int *Pi,
    int *enabled_actions,
    int *adj_sidx,
    float *stable,
    float eps,
    int max_iter,
    int max_unstable
    ) {
    /* 
    Get the COO matrix into sparsescoo fmt

    Then multiply the COO by the initial value vector

    The rewards matrix is also sparse so it will need a sparse matrix descr
    as well. Multiply R by a repeated weight vector in the number 
    of prods and actions

    Finally sum the result

    This should happen in a loop until convergence

    I also want to do some wall timing to see some statistics on 
    the GPU 
    */
    // build the sparse transition matrix first

    cusparseHandle_t     handle;
    cublasHandle_t       blashandle;

    //cudaEvent_t          start, stop;
    cudaStream_t         stream0, stream1;


    cusparseSpMatDescr_t descrPinit = NULL, descrP = NULL, 
                         descrRinit = NULL, descrR = NULL;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUBLAS(cublasCreate(    &blashandle));
    CHECK_CUDA( cudaStreamCreate( &stream0 ) );
    CHECK_CUDA( cudaStreamCreate( &stream1 ) );
    // allocated the device memory for the COO matrix
    CHECK_CUDA( cudaHostRegister(w_init, r_init_n * sizeof(float), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(w, r_init_n * sizeof(float), cudaHostRegisterDefault));

    // declare some pinned memory for the transition matrix
    CHECK_CUDA( cudaHostRegister(p_init_i, p_init_i_size * sizeof(int), cudaHostRegisterDefault) );
    CHECK_CUDA( cudaHostRegister(p_init_j, p_init_nz * sizeof(int), cudaHostRegisterDefault) );
    CHECK_CUDA( cudaHostRegister(p_init_x, p_init_nz * sizeof(float), cudaHostRegisterDefault) );

    CHECK_CUDA( cudaHostRegister(p_i, p_i_size * sizeof(int), cudaHostRegisterDefault) );
    CHECK_CUDA( cudaHostRegister(p_j, p_nz * sizeof(int), cudaHostRegisterDefault) );
    CHECK_CUDA( cudaHostRegister(p_x, p_nz * sizeof(float), cudaHostRegisterDefault) );

    CHECK_CUDA( cudaHostRegister(r_init_i, r_init_i_size * sizeof(int), cudaHostRegisterDefault) );
    CHECK_CUDA( cudaHostRegister(r_init_j, r_init_nz * sizeof(int), cudaHostRegisterDefault) );
    CHECK_CUDA( cudaHostRegister(r_init_x, r_init_nz * sizeof(float), cudaHostRegisterDefault) );
 
    CHECK_CUDA( cudaHostRegister(r_i, r_i_size * sizeof(int), cudaHostRegisterDefault) );
    CHECK_CUDA( cudaHostRegister(r_j, r_nz * sizeof(int), cudaHostRegisterDefault) );
    CHECK_CUDA( cudaHostRegister(r_x, r_nz * sizeof(float), cudaHostRegisterDefault) );

    CHECK_CUDA( cudaHostRegister(x_init, p_init_n * sizeof(float), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(y_init, p_init_m * sizeof(float), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(rmv_init, p_init_m * sizeof(float), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(unstable, p_init_m * sizeof(int), cudaHostRegisterDefault));

    CHECK_CUDA( cudaHostRegister(y, p_m * sizeof(float), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(rmv, p_m * sizeof(float), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(Pi, p_n * sizeof(int), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(enabled_actions, p_n * sizeof(int), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(adj_sidx, p_n * sizeof(int), cudaHostRegisterDefault));
    CHECK_CUDA( cudaHostRegister(stable, p_n * sizeof(float), cudaHostRegisterDefault));
    
    
    // ----------------------------------------------------------------
    //       STREAM 0: DATA TRANSFER
    // ----------------------------------------------------------------

    // ----------------------------------------------------------------
    //                       Initial Policy Transition Matrix
    // ----------------------------------------------------------------
    
    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    
    int *dPInitCsrRowPtr, *dPInitCsrColPtr;
    float *dPInitCsrValPtr;
    
    // allocate device memory to store the sparse CSR 
    CHECK_CUDA(cudaMalloc((void **)&dPInitCsrValPtr, sizeof(float) * p_init_nz));
    CHECK_CUDA(cudaMalloc((void **)&dPInitCsrColPtr, sizeof(int) * p_init_nz));
    CHECK_CUDA(cudaMalloc((void **)&dPInitCsrRowPtr, sizeof(int) * p_init_i_size));

    // |
    // --------------------> Put all of the init on stream 0
    CHECK_CUDA(cudaMemcpyAsync(dPInitCsrValPtr, p_init_x, sizeof(float) * p_init_nz, cudaMemcpyHostToDevice, stream0) );
    CHECK_CUDA(cudaMemcpyAsync(dPInitCsrColPtr, p_init_j, sizeof(int) * p_init_nz, cudaMemcpyHostToDevice, stream0) );
    CHECK_CUDA(cudaMemcpyAsync(dPInitCsrRowPtr, p_init_i, sizeof(int) * p_init_i_size, cudaMemcpyHostToDevice, stream0) );

    // create the sparse CSR matrix in device memory
    
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrPinit, // MATRIX DESCRIPTION
        p_init_m, // NUMBER OF ROWS
        p_init_n, // NUMBER OF COLS
        p_init_nz, // NUMBER OF NON ZERO VALUES
        dPInitCsrRowPtr, // ROWS OFFSETS
        dPInitCsrColPtr, // COL INDICES
        dPInitCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));

    // ----------------------------------------------------------------
    //                       Initial Rewards Matrix
    // ----------------------------------------------------------------

    int *dRInitCsrRowPtr, *dRinitCsrColPtr;
    float *dRInitCsrValPtr;

    // allocate device memory to store the sparse CSR 
    CHECK_CUDA( cudaMalloc((void **)&dRInitCsrValPtr, sizeof(float) * r_init_nz) );
    CHECK_CUDA( cudaMalloc((void **)&dRInitCsrRowPtr, sizeof(int) * r_init_i_size) );
    CHECK_CUDA( cudaMalloc((void **)&dRinitCsrColPtr, sizeof(int) * r_init_nz) );
    CHECK_CUDA( cudaMemcpyAsync(dRInitCsrValPtr, r_init_x, sizeof(float) * r_init_nz, cudaMemcpyHostToDevice, stream0) );
    CHECK_CUDA( cudaMemcpyAsync(dRinitCsrColPtr, r_init_j, sizeof(int) * r_init_nz, cudaMemcpyHostToDevice, stream0) );
    CHECK_CUDA( cudaMemcpyAsync(dRInitCsrRowPtr, r_init_i, sizeof(int) * r_init_i_size, cudaMemcpyHostToDevice, stream0) );

    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrRinit, // MATRIX DESCRIPTION
        r_init_m, // NUMBER OF ROWS
        r_init_n, // NUMBER OF COLS
        r_init_nz, // NUMBER OF NON ZERO VALUES
        dRInitCsrRowPtr, // ROWS OFFSETS
        dRinitCsrColPtr, // COL INDICES
        dRInitCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));

    // --------------INTIAL TRANSITION MATRIX MULTIPLICATION SETUP------------
    float alpha = 1.0;
    float beta = 1.0;
    float policy_stable = 0.;
    //float *epsilon = (float*) malloc(p_init_m * sizeof(float));
    //int iepsilon;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecXinit, vecYinit;
    float *dXinit, *dYinit, *dZinit, *dStaticYinit;
    int *dUnstableInit;
    void* dBufferInit = NULL;
    size_t bufferSizeInit = 0;

    // Allocate the device memory
    CHECK_CUDA( cudaMalloc((void**)&dXinit, p_init_m * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**)&dYinit, p_init_m * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**)&dZinit, p_init_m * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**)&dStaticYinit, p_init_m * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**)&dUnstableInit, p_init_m * sizeof(int)) );

    // Allocate registered memory to the device
    CHECK_CUDA( cudaMemcpyAsync(dXinit, x_init, sizeof(float) * p_init_n, cudaMemcpyHostToDevice, stream0) );
    CHECK_CUDA( cudaMemcpyAsync(dYinit, y_init, sizeof(float) * p_init_m, cudaMemcpyHostToDevice, stream0) );
    CHECK_CUDA( cudaMemcpyAsync(dUnstableInit, unstable, sizeof(int) * p_init_m, cudaMemcpyHostToDevice, stream0) );
    CHECK_CUDA( cudaMemcpyAsync(dStaticYinit, y_init, sizeof(float) * p_init_m, cudaMemcpyHostToDevice, stream0) );

    // --------------REWARDS MATRIX MULTIPLICATION SETUP---------------
    float alphaR = 1.0;
    float betaR = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecWinit, vecRMvinit;
    float *dRwInit, *dRMvInit;
    void* dBufferRinit = NULL;
    size_t bufferSizeRinit = 0;

    //float *rmv = (float*) calloc(rm, sizeof(float));

    CHECK_CUDA( cudaMalloc((void**)&dRwInit, r_init_n * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**)&dRMvInit, r_init_m * sizeof(float)) );

    // copy the vector from host memory to device memory
    CHECK_CUDA( cudaMemcpyAsync(dRwInit, w_init, r_init_n * sizeof(float), cudaMemcpyHostToDevice, stream0) );
    CHECK_CUDA( cudaMemcpyAsync(dRMvInit, rmv_init, r_init_m * sizeof(float), cudaMemcpyHostToDevice, stream0) );
    
    // ----------------------------------------------------------------
    //       STREAM 0: END OF DATA TRANSFER
    // ----------------------------------------------------------------

    // ----------------------------------------------------------------
    //       STREAM 1: DATA TRANSFER
    // ----------------------------------------------------------------

    // ----------------------------------------------------------------
    //                       Complete Transition Matrix
    // ----------------------------------------------------------------
    
    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dPCsrRowPtr, *dPCsrColPtr;
    float *dPCsrValPtr;

    // allocate device memory to store the sparse CSR 
    CHECK_CUDA(cudaMalloc((void **)&dPCsrValPtr, sizeof(float) * p_nz));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrColPtr, sizeof(int) * p_nz));
    CHECK_CUDA(cudaMalloc((void **)&dPCsrRowPtr, sizeof(int) * p_i_size));

    // |
    // --------------------> The complete transition matrix => stream1
    CHECK_CUDA(cudaMemcpyAsync(dPCsrValPtr, p_x, sizeof(float) * p_nz, cudaMemcpyHostToDevice, stream1) );
    CHECK_CUDA(cudaMemcpyAsync(dPCsrColPtr, p_j, sizeof(int) * p_nz, cudaMemcpyHostToDevice, stream1) );
    CHECK_CUDA(cudaMemcpyAsync(dPCsrRowPtr, p_i, sizeof(int) * p_i_size, cudaMemcpyHostToDevice, stream1) );

    
    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrP, // MATRIX DESCRIPTION
        p_m, // NUMBER OF ROWS
        p_n, // NUMBER OF COLS
        p_nz, // NUMBER OF NON ZERO VALUES
        dPCsrRowPtr, // ROWS OFFSETS
        dPCsrColPtr, // COL INDICES
        dPCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));
    

    // ----------------------------------------------------------------
    //                       Complete Rewards Matrix
    // ----------------------------------------------------------------

    int *dRCsrRowPtr, *dRCsrColPtr;
    float *dRCsrValPtr;

    // allocate device memory to store the sparse CSR 
    CHECK_CUDA( cudaMalloc((void **)&dRCsrValPtr, sizeof(float) * r_nz) );
    CHECK_CUDA( cudaMalloc((void **)&dRCsrRowPtr, sizeof(int) * r_i_size) );
    CHECK_CUDA( cudaMalloc((void **)&dRCsrColPtr, sizeof(int) * r_nz) );
    CHECK_CUDA( cudaMemcpyAsync(dRCsrValPtr, r_x, sizeof(float) * r_nz, cudaMemcpyHostToDevice, stream1) );
    CHECK_CUDA( cudaMemcpyAsync(dRCsrColPtr, r_j, sizeof(int) * r_nz, cudaMemcpyHostToDevice, stream1) );
    CHECK_CUDA( cudaMemcpyAsync(dRCsrRowPtr, r_i, sizeof(int) * r_i_size, cudaMemcpyHostToDevice, stream1) );

    
    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrR, // MATRIX DESCRIPTION
        r_m, // NUMBER OF ROWS
        r_n, // NUMBER OF COLS
        r_nz, // NUMBER OF NON ZERO VALUES
        dRCsrRowPtr, // ROWS OFFSETS
        dRCsrColPtr, // COL INDICES
        dRCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));
    

    // ----------------------------------------------------------------
    //                             POLICY
    // ----------------------------------------------------------------

    int *PI, *EnabledActions, *AdjSIDX;
    float *dStable;
    CHECK_CUDA(cudaMalloc((void**)&PI, p_n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&EnabledActions, p_n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&AdjSIDX, p_n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dStable, p_n * sizeof(float)));
    //
    CHECK_CUDA(cudaMemcpyAsync(PI, Pi, p_n * sizeof(int), cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA(cudaMemcpyAsync(EnabledActions, enabled_actions, p_n * sizeof(int), cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA(cudaMemcpyAsync(AdjSIDX, adj_sidx, p_n * sizeof(int), cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA(cudaMemcpyAsync(dStable, stable, p_n * sizeof(float), cudaMemcpyHostToDevice, stream1));
    //
    // --------------TRANSITION MATRIX MULTIPLICATION SETUP------------

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecX, vecY;
    //
    float *dXtmp, *dY, *dStaticY; 
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    CHECK_CUDA(cudaMalloc((void**)&dXtmp, p_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dY, p_m * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dStaticY, p_m * sizeof(float)));
    // copy the vector from host memory to device memory
    CHECK_CUDA(cudaMemcpyAsync(dY, y, p_m * sizeof(float), cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA(cudaMemcpyAsync(dStaticY, y, p_m * sizeof(float), cudaMemcpyHostToDevice, stream1));

    // --------------REWARDS MATRIX MULTIPLICATION SETUP---------------

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecW, vecRMv;
    float *dRw, *dRMv;
    void* dBufferR = NULL;
    size_t bufferSizeR = 0;

    //float *rmv = (float*) calloc(rm, sizeof(float));

    CHECK_CUDA(cudaMalloc((void**)&dRw, r_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dRMv, r_m * sizeof(float)));

    // copy the vector from host memory to device memory
    CHECK_CUDA(cudaMemcpyAsync(dRw, w, r_n * sizeof(float), cudaMemcpyHostToDevice, stream1) );
    CHECK_CUDA(cudaMemcpyAsync(dRMv, rmv, r_m  * sizeof(float), cudaMemcpyHostToDevice, stream1) );

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, p_m, dY, CUDA_R_32F));

    CHECK_CUSPARSE( cusparseSetStream(handle, stream1) );

    // create a dense vector on device memory
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecW, r_n, dRw, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecRMv, r_m, dRMv, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSizeR));
    CHECK_CUDA(cudaMalloc(&dBufferR, bufferSizeR));

    // ----------------------------------------------------------------
    //       STREAM 1: END OF DATA TRANSFER
    // ----------------------------------------------------------------

    // ----------------------------------------------------------------
    //                      Start of VI
    // ----------------------------------------------------------------
    
    // TRANSITION MATRIX CUSPARSE SETUP

    // create a dense vector on device memory
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecYinit, p_init_m, dYinit, CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecXinit, p_init_n, dXinit, CUDA_R_32F) );

    // REWARDS MATRIX CUSPARSE SETUP
    //
    //cudaMemcpy(dRstaticMx, rmv, rm * sizeof(float), cudaMemcpyHostToDevice);

    // create a dense vector on device memory
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecWinit, r_init_n, dRwInit, CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecRMvinit, r_init_m, dRMvInit, CUDA_R_32F) );
    
    // Allocate the buffers for the matrix-vector multiplication workspace
    //
    // Set the cusparse handle to the stream just before the operation 
    // according to the documentation 
    
    CHECK_CUSPARSE( cusparseSetStream(handle, stream0) );
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrPinit, vecXinit, &beta, vecYinit, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSizeInit) );
    CHECK_CUDA( cudaMalloc(&dBufferInit, bufferSizeInit) );
    
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alphaR, descrRinit, vecWinit, &betaR, vecRMvinit, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSizeRinit) );
    CHECK_CUDA( cudaMalloc(&dBufferRinit, bufferSizeRinit) );
    CHECK_CUBLAS( cublasSetStream(blashandle, stream0) );
    
    
    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alphaR, descrRinit, vecWinit, &betaR, vecRMvinit, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBufferRinit));
    
    float maxeps;
    maxeps = 0.0f;

    for (int algo_i = 0; algo_i < max_iter; algo_i ++) {

        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alpha, descrPinit, vecXinit, &beta, vecYinit, CUDA_R_32F, 
            CUSPARSE_MV_ALG_DEFAULT, dBufferInit));

        // push this into the algorithm loop


        // ---------------------SUM DENSE VECTORS-------------------------

         
        //The gpu memory should already be allocated, i.e. we are summing
        //dY + dRMv
        
        CHECK_CUBLAS(cublasSaxpy(blashandle, p_init_m, &alpha, dRMvInit, 1, dYinit, 1));
        
        // ---------------------COMPUTE EPSILON---------------------------
        // what is the difference between dY and dX

        // EPSILON COMPUTATION
        abs_diff_launcher(dYinit, dXinit, dZinit, dUnstableInit, p_init_m, max_unstable);
        //CHECK_CUBLAS(cublasIsamax(blashandle, pm, dZ, 1, &iepsilon));

        thrust::device_ptr<float> dev_ptr(dZinit);
        maxeps = *thrust::max_element(thrust::device, dev_ptr, dev_ptr + p_init_m);

        CHECK_CUBLAS(cublasScopy(blashandle, p_init_m, dYinit, 1, dXinit, 1));
        // RESET Y
        CHECK_CUBLAS(cublasScopy(blashandle, p_init_m, dStaticYinit, 1, dYinit, 1));
        //std::cout << "EPS_TEST " << "THRUST "<< maxeps << std::endl;
        if (maxeps < eps || isnan(maxeps)) {
            //printf("INITIAL POLICY GENERATED; EPS TOL REACHED in %i ITERATIONS\n", algo_i);
            break;
        }
    }

    // --------------STREAM SYNCHRONISATION------------

    CHECK_CUDA( cudaDeviceSynchronize() );

    // ----------------POLICY OPTIMISATION-------------

    CHECK_CUDA(cudaMemcpy(dXtmp, dXinit, p_n * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUSPARSE( cusparseSetStream(handle, 0) );

    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F, 
        CUSPARSE_MV_ALG_DEFAULT, dBufferR));

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrP, vecXinit, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // ALGORITHM LOOP - POLICY GENERATION
    for (int algo_i = 0; algo_i < max_iter; algo_i ++) {

        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alpha, descrP, vecXinit, &beta, vecY, CUDA_R_32F, 
            CUSPARSE_MV_ALG_DEFAULT, dBuffer));    

        // ---------------------SUM DENSE VECTORS-------------------------

        /* 
        i.e. we are summing dY + dRMv
        */
        
        CHECK_CUBLAS(cublasSaxpy(blashandle, p_m, &alpha, dRMv, 1, dY, 1));
        // ------------------COMPUTE POLICY STABLE------------------------
        
        max_value_launcher(dY, EnabledActions, AdjSIDX, dXtmp, dXinit, PI, dStable, 
                           eps, p_n);
        
        // we can compute if the policy is stable with cublas 
        CHECK_CUBLAS(cublasScopy(blashandle, p_n, dXtmp, 1, dXinit, 1));
        
        cublasSasum(blashandle, p_n, dStable, 1, &policy_stable);
        if (policy_stable == 0) {
            break;
        }
        CHECK_CUBLAS(cublasScopy(blashandle, p_m, dStaticY, 1, dY, 1));
    }

    // COPY THE SOLUTION BACK TO THE HOST
    CHECK_CUDA(cudaMemcpy(x_init, dXinit, p_n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(Pi, PI, p_n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Load the the other data on the 
    // 
    // ----------------------------------------------------------------
    //                       Memory Management
    // ----------------------------------------------------------------
    //CHECK_CUDA( cudaStreamSynchronize( stream1 ) );
    CHECK_CUDA( cudaStreamDestroy( stream0 ));
    CHECK_CUDA( cudaStreamDestroy( stream1 ));
    //destroy the vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(descrPinit));
    CHECK_CUSPARSE(cusparseDestroySpMat(descrP));
    CHECK_CUSPARSE(cusparseDestroySpMat(descrRinit));
    CHECK_CUSPARSE(cusparseDestroySpMat(descrR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecRMv));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecW));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecYinit));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecRMvinit));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecWinit));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUBLAS(cublasDestroy(blashandle));

    // Free the device memory
    CHECK_CUDA(cudaFree(dPInitCsrColPtr));
    CHECK_CUDA(cudaFree(dPInitCsrRowPtr));
    CHECK_CUDA(cudaFree(dPInitCsrValPtr));
    CHECK_CUDA(cudaFree(dRinitCsrColPtr));
    CHECK_CUDA(cudaFree(dRInitCsrRowPtr));
    CHECK_CUDA(cudaFree(dRInitCsrValPtr));
    CHECK_CUDA(cudaFree(dPCsrColPtr));
    CHECK_CUDA(cudaFree(dPCsrRowPtr));
    CHECK_CUDA(cudaFree(dPCsrValPtr));
    CHECK_CUDA(cudaFree(dRCsrColPtr));
    CHECK_CUDA(cudaFree(dRCsrRowPtr));
    CHECK_CUDA(cudaFree(dRCsrValPtr));

    CHECK_CUDA(cudaFree(dXinit));
    CHECK_CUDA(cudaFree(dYinit));
    CHECK_CUDA(cudaFree(dStaticYinit));
    CHECK_CUDA(cudaFree(dZinit));
    CHECK_CUDA(cudaFree(dUnstableInit));
    CHECK_CUDA(cudaFree(PI));
    CHECK_CUDA(cudaFree(EnabledActions));
    CHECK_CUDA(cudaFree(AdjSIDX));
    CHECK_CUDA(cudaFree(dStable));
    CHECK_CUDA(cudaFree(dXtmp));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dStaticY));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dRw));
    CHECK_CUDA(cudaFree(dRwInit));
    CHECK_CUDA(cudaFree(dRMv));
    CHECK_CUDA(cudaFree(dBufferR));

    CHECK_CUDA( cudaHostUnregister(p_init_i) );
    CHECK_CUDA( cudaHostUnregister(p_init_j) );
    CHECK_CUDA( cudaHostUnregister(p_init_x) );
    CHECK_CUDA( cudaHostUnregister(p_i) );
    CHECK_CUDA( cudaHostUnregister(p_j) );
    CHECK_CUDA( cudaHostUnregister(p_x) );

    CHECK_CUDA( cudaHostUnregister(r_init_i) );
    CHECK_CUDA( cudaHostUnregister(r_init_j) );
    CHECK_CUDA( cudaHostUnregister(r_init_x) );
    CHECK_CUDA( cudaHostUnregister(r_i) );
    CHECK_CUDA( cudaHostUnregister(r_j) );
    CHECK_CUDA( cudaHostUnregister(r_x) );

    CHECK_CUDA( cudaHostUnregister(x_init) );
    CHECK_CUDA( cudaHostUnregister(w_init) );
    CHECK_CUDA( cudaHostUnregister(w) );
    CHECK_CUDA( cudaHostUnregister(y_init) );
    CHECK_CUDA( cudaHostUnregister(rmv_init) );
    CHECK_CUDA( cudaHostUnregister(unstable) );
    CHECK_CUDA( cudaHostUnregister(y) );
    CHECK_CUDA( cudaHostUnregister(rmv) );
    CHECK_CUDA( cudaHostUnregister(Pi) );
    CHECK_CUDA( cudaHostUnregister(enabled_actions ) );
    CHECK_CUDA( cudaHostUnregister(adj_sidx) );
    CHECK_CUDA( cudaHostUnregister(stable) );
    //cudaFree(d_eps);
    return 0;
}


int multi_obj_solution(
    int pm,
    int pn,
    int pnz,
    int * pi,
    int * pj,
    float * px,
    int pi_size,
    int rm,
    int rn,
    int rnz,
    int *ri,
    int *rj,
    float *rx,
    int ri_size,
    float eps,
    int nobjs,
    float *x,
    float *w,
    float *z,
    int *unstable,
    int max_iter, 
    int max_unstable
) {
    // Setup the framework infrastructure
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    cublasHandle_t blashandle;
    CHECK_CUBLAS(cublasCreate(&blashandle));


    cusparseSpMatDescr_t descrP = NULL;
    cusparseSpMatDescr_t descrR = NULL;

    //float rStorage[rm * nobjs] = { 0. };

    // allocated the device memory for the COO matrix

    // ----------------------------------------------------------------
    //                       Transition Matrix
    // ----------------------------------------------------------------

    //allocate dCsrRowPtr, dCsrColPtr, dCsrValPtr
    int *dPCsrRowPtr, *dPCsrColPtr;
    float *dPCsrValPtr;

    // allocate device memory to store the sparse CSR 
    cudaMalloc((void **)&dPCsrValPtr, sizeof(float) * pnz);
    cudaMalloc((void **)&dPCsrRowPtr, sizeof(int) * pi_size);
    cudaMalloc((void **)&dPCsrColPtr, sizeof(int) * pnz);

    cudaMemcpy(dPCsrValPtr, px, sizeof(float) * pnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dPCsrColPtr, pj, sizeof(int) * pnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dPCsrRowPtr, pi, sizeof(int) * pi_size, cudaMemcpyHostToDevice);
    
    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrP, // MATRIX DESCRIPTION
        pm, // NUMBER OF ROWS
        pn, // NUMBER OF COLS
        pnz, // NUMBER OF NON ZERO VALUES
        dPCsrRowPtr, // ROWS OFFSETS
        dPCsrColPtr, // COL INDICES
        dPCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));
    
    // ----------------------------------------------------------------
    //                       Rewards Matrix
    // ----------------------------------------------------------------
    
    int *dRCsrRowPtr, *dRCsrColPtr;
    float *dRCsrValPtr;

    // allocate device memory to store the sparse CSR 
    cudaMalloc((void **)&dRCsrValPtr, sizeof(float) * rnz);
    cudaMalloc((void **)&dRCsrRowPtr, sizeof(int) * ri_size);
    cudaMalloc((void **)&dRCsrColPtr, sizeof(int) * rnz);
    cudaMemcpy(dRCsrValPtr, rx, sizeof(float) * rnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dRCsrColPtr, rj, sizeof(int) * rnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dRCsrRowPtr, ri, sizeof(int) * ri_size, cudaMemcpyHostToDevice);

    // create the sparse CSR matrix in device memory
    CHECK_CUSPARSE(cusparseCreateCsr(
        &descrR, // MATRIX DESCRIPTION
        rm, // NUMBER OF ROWS
        rn, // NUMBER OF COLS
        rnz, // NUMBER OF NON ZERO VALUES
        dRCsrRowPtr, // ROWS OFFSETS
        dRCsrColPtr, // COL INDICES
        dRCsrValPtr, // VALUES
        CUSPARSE_INDEX_32I, // INDEX TYPE ROWS
        CUSPARSE_INDEX_32I, // INDEX TYPE COLS
        CUSPARSE_INDEX_BASE_ZERO, // BASE INDEX TYPE
        CUDA_R_32F // DATA TYPE
    ));

    // ----------------------------------------------------------------
    //                      Start of VI
    // ----------------------------------------------------------------

    // --------------TRANSITION MATRIX MULTIPLICATION SETUP------------
    
    float alpha = 1.0;
    float beta = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecX, vecY;
    float *dX, *dY, *dZ, *dStorage;
    int *dUnstable;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    // Allocate the device memory
    CHECK_CUDA( cudaMalloc((void**)&dX, pm * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**)&dY, pm * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**)&dZ, pm * nobjs * sizeof(float)) ); // use this to store the epsilon values
    CHECK_CUDA( cudaMalloc((void**)&dStorage, pm * nobjs * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**)&dUnstable, pm * nobjs * sizeof(int)) );
    
    // copy the vector from host memory to device memory
    CHECK_CUDA(cudaMemcpy(dX, x, pn * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dY, dX, pm * sizeof(float), cudaMemcpyDeviceToDevice) );
    CHECK_CUDA(cudaMemcpy(dZ, z, pm * nobjs * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dStorage, dZ, nobjs * pm * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dUnstable, unstable, nobjs * pm * sizeof(float), cudaMemcpyHostToDevice));

    // create a dense vector on device memory
    cusparseCreateDnVec(&vecX, pn, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, pm, dY, CUDA_R_32F);

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descrP, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    
    // --------------REWARDS MATRIX MULTIPLICATION SETUP---------------

    float alphaR = 1.0;
    float betaR = 1.0;

    // assign the cuda memory for the vectors
    cusparseDnVecDescr_t vecW, vecRMv;
    float *dRw, *dRMv, *dRStorage;
    void* dBufferR = NULL;
    size_t bufferSizeR = 0;

    //float *rmv = (float*) calloc(rm, sizeof(float));

    CHECK_CUDA( cudaMalloc((void**)&dRw, rn * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**)&dRMv, rm * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**)&dRStorage, rm * nobjs * sizeof(float)) )

    // copy the vector from host memory to device memory
    CHECK_CUDA( cudaMemcpy(dRw, w, rn * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dRMv, dX, rm * sizeof(float), cudaMemcpyDeviceToDevice) );
    CHECK_CUDA( cudaMemcpy(dRStorage, dZ, rm * nobjs * sizeof(float), cudaMemcpyDeviceToDevice) );
    // create a dense vector on device memory
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecW, rn, dRw, CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecRMv, rm, dRMv, CUDA_R_32F) );

    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSizeR) );
    CHECK_CUDA( cudaMalloc(&dBufferR, bufferSizeR) );

    for (int k = 0; k < nobjs; k++) {
        // reset the W vector to all zeros
        CHECK_CUDA( cudaMemset(dRw, 0., nobjs * sizeof(float)) )
        CHECK_CUDA( cudaMemset(dRMv, 0., rm * sizeof(float)) )
        // Change the value of the w array according to the objective we are
        // considering
        change_elem<<<1, 1>>>(dRw, k, 1.0);
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
            &alphaR, descrR, vecW, &betaR, vecRMv, CUDA_R_32F, 
            CUSPARSE_MV_ALG_DEFAULT, dBufferR));
        // copy the relevant values to the Rstorage array in the range of |S|
        copy_elems_launcher(dRStorage, k * rm, dRMv, 0,  rm);
    }

    float maxeps;
    maxeps = 0.0f;

    for (int i = 0; i < max_iter; i++) {

        for (int k = 0; k < nobjs; k++) {
            copy_elems_launcher(dY, 0, dRStorage, k * rm, rm);
            copy_elems_launcher(dX, 0, dStorage, k * pm, pm);
            // The next line compute R(k) + P.x
            /*
            CHECK_CUDA(cudaMemcpy(x, dX, pm * sizeof(float), cudaMemcpyDeviceToHost));
            printf("k=%i\n", k);
            for (int i=0; i<pm; i++) {
                printf("%f, ", x[i]);
            }
            printf("\n");
            */
            CHECK_CUSPARSE(cusparseSpMV(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                &alpha, descrP, vecX, &betaR, vecY, CUDA_R_32F, 
                CUSPARSE_MV_ALG_DEFAULT, dBuffer));

            // update the epsilons for the computed values
            CHECK_CUDA(cudaMemcpy(x, dY, pm * sizeof(float), cudaMemcpyDeviceToHost));
            /*
            printf("Y; k=%i\n", k);
            for (int i=0; i<pm; i++) {
                printf("%f, ", x[i]);
            }
            printf("\n");
            */
            mobj_abs_diff_launcher(dX, dY, dZ, dUnstable, k, pm, max_unstable);

            // copy x <- y
            copy_elems_launcher(dStorage, k * pm, dY, 0, pm);
        }
        
        //CHECK_CUDA(cudaMemcpy(unstable, dUnstable, pm * nobjs * sizeof(int), cudaMemcpyDeviceToHost));
        //printf("\n");
        //for (int i=0; i<pm *nobjs; i++) {
        //    printf("%i, ", unstable[i]);
        //}
        //printf("\n");
        
        // lets try and see if we can access our Z values
        thrust::device_ptr<float> dev_ptr(dZ);
        maxeps = *thrust::max_element(thrust::device, dev_ptr, dev_ptr + pm * nobjs);
        //std::cout << "EPS_TEST " << "THRUST "<< maxeps << std::endl;
        if (maxeps < eps || isnan(maxeps) || isinf(maxeps)) {
            //printf("\nFinished M_obj VPI in %i steps\n", i);
            break;
        }
    }

    CHECK_CUDA(cudaMemcpy(z, dStorage, pm * nobjs * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    CHECK_CUSPARSE( cusparseDestroySpMat(descrP) )
    CHECK_CUSPARSE( cusparseDestroySpMat(descrR) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecRMv) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecW) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUBLAS( cublasDestroy(blashandle) )

    // Free the device memory
    CHECK_CUDA( cudaFree(dPCsrColPtr) )
    CHECK_CUDA( cudaFree(dPCsrRowPtr) )
    CHECK_CUDA( cudaFree(dPCsrValPtr) )
    CHECK_CUDA( cudaFree(dRCsrColPtr) )
    CHECK_CUDA( cudaFree(dRCsrRowPtr) )
    CHECK_CUDA( cudaFree(dRCsrValPtr) )
    //cudaFree(d_eps);
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    CHECK_CUDA( cudaFree(dRStorage) )
    CHECK_CUDA( cudaFree(dUnstable) )
    CHECK_CUDA( cudaFree(dZ) )
    CHECK_CUDA( cudaFree(dStorage) )
    CHECK_CUDA( cudaFree(dRw) )
    CHECK_CUDA( cudaFree(dRMv) )
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dBufferR) )
    //free(x); free(y); free(rmv); free(w); free(unstable);
    return 0;
}

}