# Parallel MCMC for CPU/GPU

This project implements parallel Markov Chain Monte Carlo (MCMC) on both CPU and GPU according to the algorithm presented in [Neiswanger, Wang, and Xing (2013)](https://arxiv.org/abs/1311.4780). A simple synthetic dataset is generated in Python, and then parameter estimation is performed using MCMC. The CPU implentation is written in C++ using MPI. The GPU implementation is written in CUDA and allows for both shared and global memory.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The code was developed and tested in Linux, so users on other operating systems may require slightly different commands.

### Prerequisites

The CPU code requires a C++ MPI library, such as [Open MPI](https://www.open-mpi.org/), as well as [BLAS](http://www.netlib.org/blas/) for efficient random number generation. The GPU code requires the [Nvidia CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) for access to GPU-accelerated libraries and the C++ compiler, and [cuBLAS](https://developer.nvidia.com/cublas).

### Installing
The executable MCMC programs can be compiled and linked very simply using the provided Makefile. To build both the CPU and GPU programs, run
```
make
```
If you only want to build the CPU program, run
```
make cpu
```
and if you only want the GPU program run
```
make gpu
```

### Running the Code
The MCMC program performs parameter estimation using synthetic data. For convenience, a basic Python script is provided that generates random samples from a multivariate normal distribution of specified dimension. The calculation of the determinant requires a diagonal covariance matrix in the current implementation, although this could be trivially generalized to allow for covariance between the variables. To generate the synthetic data using the provided script, run
```
python n_dimensions n_samples
```
This will save n\_samples drawn from a multivariate normal distribution with n\_dimensions to a text file that can be parsed by the MCMC programs. 

The CPU code is run on multiple processors using mpirun. An example to run the MCMC analysis for 10000 steps on a 10-dimensional Gaussian on 4 processors  would look like
```
mpirun -np 4 ./mcmc_cpu --n_steps=10000 --n_dim=10
```
The GPU code for the same parameters would be
```
./mcmc_gpu --n_steps=10000 --n_dim=10
```
A full list of command line arguments for either program can be viewed using the help flag -h or --help.

### Analyzing the Parameter Estimation 
We can now analyze the results of the parameter estimation. For the CPU results, the individual estimates must first be aggregated. This can be accomplished using the provided script aggregate_cpu_estimates.py. Then we can look at the estimates for the means and compute the l2 norm of the error. The figure below shows the convergence for the means of a 15-dimensional Gaussian.

<img align="center" src="/images/evolution_15.png" width="50%" alt="Convergence" />
![Convergence][/images/evolution_15.png]

### Scaling
The runtime scales well with number of steps, dimensions, and datapoints.

<img align="center" src="/images/scaling_20.png" width="50%" alt="Step Scaling" />
<img align="center" src="/images/scaling_data.png" width="50%" alt="Data Scaling" />
<img align="center" src="/images/scaling_dim.png" width="50%" alt="Dimension Scaling" />
