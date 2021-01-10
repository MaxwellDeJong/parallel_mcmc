default: cpu gpu

gpu: mcmc.cu
	nvcc mcmc.cu -lcurand -o mcmc_gpu
	
cpu: mcmc_cpu.cpp
	mpicxx mcmc_cpu.cpp -O3 -lm -lgsl -lgslcblas -o mcmc_cpu
