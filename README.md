# ARC

0. About

Adaptive Ray-tracing with CUDA (ARC) is an Astronomy module designed to efficiently track radiation in a cosmological volume and its effects on the ionization state of the gas using the power of CUDA to take advantage of the power of multi-level parallelization (multiple parallel GPUs, each with thousands of parallel cores). The version of the code presented here is designed to track a cube of fixed sized over any given period of cosmic time on a fixed grid with size 2^n. (Current version tracks (10 Mpc)^3 starting at z=30 with PLANCK parameters, but will soon be updated to allow freedom of choice for these parameters).

In depth code description and accuracy tests are presented in full detail here:

ARC: adaptive ray-tracing with CUDA, a new ray tracing code for parallel GPUs
https://arxiv.org/pdf/1807.07094.pdf

1. How to set up this code

asdf

2. How to run this code

asdf

3. Contact the author

I may be contacted at bth@astro.umd.edu for any questions regarding this code.

I wrote ARC from the ground up for my own use in my thesis work, and as such the code is self-contained and designed to work with the files I had at the outset of the project. However, I tried to keep the code as flexible and modular as possible, so it shouldn't be too difficult to apply it to different file types, or to utilize pieces of the code (the CUDA ray-tracing, CUDA ionization integrator, halo-matching algorithm, etc.) for different codes with comparable structure. CUDA is a uniquely powerful framework for these types of computations, and I believe that any large scale project with these types of calculations would benefit from the utilization of CUDA.
