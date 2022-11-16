Tasks:
- [x] Build the compressed enabled action data structure
- [x] Construct the build function for the Product MDP using the compressed data structure
- [x] Work out which functions are absolutely necessary for the ProductMDP
- [ ] Build the CPU thread service product model implementation
    - [x] code the value iteration for MKL blas, spblas; dig out CXSparse
    - [x] incorporate f32 CSR BLAS routines into the Value iteration routine
    - [ ] stream the product models to the CPUs computing there r values
    - [ ] collect r values in the MPSC and compute the allocation function
- [ ] Build the simplest working example of the single block GPU multi-product model matrix
    - [ ] construct the policy optimisation kernel for the GPU based on the CSR pointers
    - [ ] build the multi-objective value iteration implementation based on the compressed data structure

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
