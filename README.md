KEY CONCEPTS AND CONTRIBUTIONS:
1. The key thing is parallelising the problem, in a memory safe, and threadsafe manner. How do we achieve memory safety. We achieve memory safety through ownership of each aspect of the model 

Tasks:
- [x] Build the compressed enabled action data structure
- [x] Construct the build function for the Product MDP using the compressed data structure
- [x] Work out which functions are absolutely necessary for the ProductMDP
- [ ] Build the CPU thread service product model implementation
    - [x] (MKL BLAS/BLIS/C DOES NOT WORK - Use SPRS instead of thread safety) code the value iteration for MKL blas, spblas; dig out CXSparse
    - [x] (BLAS DOES NOT WORK - USE SPRS instead iwth mutable for each iteration) incorporate f32 CSR BLAS routines into the Value iteration routine
    - [x] stream the product models to the CPUs computing there r values
    - [x] collect r values in the MPSC and compute the allocation function
- [x] Build the simplest working example of the single block GPU multi-product model matrix
    - [x] construct the policy optimisation kernel for the GPU based on the CSR pointers
    - [x] build the multi-objective value iteration implementation based on the compressed data structure

Scratch Notes:

I have constructed the model for building the matrices P, and R which includes the compressed action format
only for enabled actions. 

A couple of things to do at this point. 
1. We need to work out how to efficiently use threading to process the CPU model so that we don't get context splitting. This is the major source of inefficiency in the CPU distributed model. 

2. Re implement CPU Value iteration model to best take advantage of the new compressed action sparse matrix format. However there are a couple of problems
    - The new CSR format uses f32 floating point numbers for better memory efficiency. 
    - We can no longer use the CXSparse matrix because it only uses f64 IEEE format. 
    - The compression and argmax routines will need to be copied over from the GPU algorithm so that we can handle the f32 compressions without CXSparse

3. Current sources of error for the new data structure could be accidental misindexing in the state-action adjusted reference - This was an error but is now fixed. Need to remember to pass the memory reference to the adjusted state-action -> state mapping for any functions which use the action compressed CSR matrix. 

4. Using Intel MKL breaks thread safety for an unknown reason, and I am not able to fix it without rewriting the threading context in C which has some enormous implications. Therefore, the next best option is to write the particular sparse matrix functions we require.
- [X] (THIS IS NOT THREAD SAFE AND CAUSES THREAD BLOCKING) Matrix Vector multiplication cs_si_gaxpy which will do the R.w + P.v computation.

5. Actually it seems that calling any BLAS routine from a thread blocks other threads from accessing BLAS. This is a big problem  for multi-threaded blas. I wonder though if this is a BLAS thing, and therefore we are better off implementing Pure rust implementations.
- So it turns out, as can be observed in the benchmarks `cargo bench`, that the above sparse matrix machinery using BLAS in the multithreaded setting is much slower than `sprs`. If I were only looking at the forums this would never be apparent as there are commentors who clearly have not done these experiments. Setup 5x5 and 4x6 sparse matrices with nz: 13, 8 resp. Outcomes: Rayon C blis over 100_000 sparse matrix dense vector computations = 68.72ms `+-` 0.35%(mean) over 100 samples; Rayon SPRS over 100_000 sparse matrix dense vector computations = 1.55ms `+-` 1.93%(mean) over 100 samples. 
- Further: blis sparse GAXPY 427.7 ns per matrix vector computation compared with SPRS sparse_dense_mul 225.85 ns per matrix vector computation.
- Conclusion: Not only is there a threading problem with GAXPY being a thread blocking operation, there is also an overhead in launching GAXPY from Rust which results in approx 100% extra computation time. This does not mean that SPRS is faster than BLIS for sequential sparse matrix computations, it just means that when Rust is used to call BLIS there is an overhead somewhere. Therefore, best to use SPRS is all sparse matrix vector computations in value iteration. 

6. IDEA: 
- [x] Compute a block of product models equal to the number of cores in the POOL. Maybe `a (times)` where `a` is a coefficient of the number of double ups to send to each of the processors, e.g. if `a=2` then send 2 product models to each thread.
- [x] The Cxx data structure will be consumed by the CSR SPRS type, so the algorithm will only need CSR from that point onwards. We will still need the Cxx matrix on exit though because it will be used for generating the new w vector.

7. GPU Related issues:
    1. What is the best way of constructing the CSR data structure for the GPU?
    2. Do we use a sparse matrix API or do we construct our own CSR format kernel for doing one step value-policy iteration and comparing the action values?
        - A problem related to the enabled actions per state. If there is a large difference between the number of actions enabled in particular states (e.g. some with 1 action, some with 100 actions), then some threads will be computing a lot more than other threads, meaning that some threads will be idle while waiting for other threads to finish. This creates a load imbalance problem. Either way we are presented with some issues. On the one hand if we use dynamic resourcing, then we will require thread synchronisation which is slow. On the other hand, we are presented with thread workload imbalance issues. 

8. GPU/CPU Streaming:
We have a total problem which (1) may or may not fit into memory, (2) we will need to do load balancing between the CPU and GPU to make sure that the we are loading the correct device with a sufficient amount of data.
    1. I expect that the device load is depending on the number of cores not on the device memory allocation (which would be much smaller than maximum load). For example, on a device which has 6100 CUDA cores and 48 CPUs I expect the load will be a matrix of maximum shape (6100, n) and 48 M_ij structs for GPU and CPU allocation respectively for maximum parallelisation. 
    ---THIS IS NOT IMPORTANT: THE MOST IMPORTANT ASPECT IS DATA UP/DOWNLOAD TO THE GPU, the most time consuming step
