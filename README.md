# Introduction

This repo is in support of the paper 'Multi-Objective Task Assignment and Multiagent Planning with Hybrid GPU-CPU Acceleration'. A long version of this paper can be found in this repo at [paper](GPU_MOTAP_NFM23_LONG.pdf).

# Installation

<b>Requirements</b>:
1. GPU device with compute >=5.0. For best results compute 8.6 and above is preferred. 
2. To run larger MAS 16Gb RAM is required. 
3. Linux (Debian Tested, Windows and MACOS not tested). Windows WSL should work correcly if the shared libraries are linked correctly.

Clone this repository and `cd` into it. <b>Installation Steps</b>:

1. Install a python environment `python -m venv path/to/env` and activate.
2. Install `maturin` with `pip install maturin`
3. If Rust is not installed, install it with `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
4. Install CUDA 11.8 runtime. Steps 5-7 assume a local terminal instance or editing .`bashrc`.
5. `export CUDA_ARCH=your_GPU_arch`. E.g. RTX3070Ti `export CUDA_ARCH=compute_86`.
6. `export CUDA_CM=your_compute`. E.g. RTX3070Ti `export CUDA_CM=sm_86`
7. `export CUDA_LIB=/path/to/lib64/` E.g. default CUDA installation `export CUDA_LIB=/usr/local/cuda/lib64/`
8. A `C` compiler is required e.g. `gcc`
9. Once the above setup has been successfully completed, running `maturin develop --release` will install the MORAP framework with optimisations, and link all the required shared libraries. 
10. make the experiment shell script executable `chmod u+x exp.sh`

Note: Steps 4-8 are optional. If the GPU is not configured to use Nvidia, then Hybrid, GPU framework architecture cannot be used. However, CPU parallelism will still work. 

# Reproducing Experiments

For convenience, run the shell script `exp.sh` which manages the execution of all of the experiments. WARNING some of the larger multiagent system experiments will take up considerable resources and are long running. For this reason there are convenience settings for running the shell script. The default is 
```bash
./exp.sh -c
```
Which runs only the experiments executing in reasonable time (about 30s total) and uses only the CPU. If a GPU is available and the intallation is successful, the experiments script can be run with the `-q` option which runs CPU, GPU, Hybrid algorithms for quick options. Further configurations accessed with `-h`. 

# Development Notes

    [ ] Implement CUDA streams to optimise model loading in the policy optimisation step.
    [ ] omega-regular properties for long running tasks i.e. limit determininstic Buchi   automata