#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line ) {
    #ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err));
         exit(-1);
                                        }
    #endif
}


inline void __cudaCheckError( const char *file, const int line ) {
    #ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit( -1 );
                                            }

        // More careful checking. However, this will affect performance.
        // Comment away if needed.
        err = cudaDeviceSynchronize();
        if( cudaSuccess != err ) {
            fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
            exit( -1 );
                                                    }
    #endif
}


__device__ void calc_cov_est(float* cov_arr, float* cov_inv_arr, float* cov_est, int n, int n_threads) {

    int i, j;
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    int offset = threadID * n;

    for (i = offset; i < offset + n; i++) {
        cov_inv_arr[i] = 1.f / cov_arr[i];
    }

    if (threadID == 0) {
        for (i = 0; i < n; i++) {
            cov_est[i] = 0.f;
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n_threads; j++) {
                cov_est[i] += cov_inv_arr[i + n * j];
            }
            cov_est[i] = 1.f / cov_est[i];
            // printf("Covariance estimate for element %d: %f\n", i, cov_est[i]);
        }
    }

    __syncthreads();

}


__device__ void calc_mu_est(float* mu_arr, float* cov_inv_arr, float* mu_inv_arr, float* cov_est, float* mu_est, int n, int n_threads) {

    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    
    int offset = threadID * n;
    int i, j;

    for (i = offset; i < offset + n; i++) {
        mu_inv_arr[i] = mu_arr[i] * cov_inv_arr[i];
    }

    if (threadID == 0) {
        for (i = 0; i < n; i++) {
            mu_est[i] = 0.f;
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n_threads; j++) {
                mu_est[i] += mu_inv_arr[i + n * j];
            }
        }

        for (i = 0; i < n; i++) {
            mu_est[i] = cov_est[i] * mu_est[i];
        }
    }

    __syncthreads();
}


__device__ void rand_init(curandState *state) {

    // unsigned int seed = (unsigned int) clock64();
    int seed = 0;
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(seed+threadID, 0, 0, &state[threadID]);
}


__device__ void get_rand_nums(curandState *state, float* rand_num, int* rand_ints, int max_int) {

    float rand_int_num;
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    rand_num[threadID] = (float) (curand_uniform(&state[threadID]));
    rand_int_num = (float) (curand_uniform(&state[threadID]));

    rand_ints[threadID] = int(max_int * rand_int_num);
}


__device__ void get_rand_gaussian(curandState *state, float sigma1, float* gaussian_arr1, float sigma2, float* gaussian_arr2) {

    int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    gaussian_arr1[threadID] = (float) (curand_normal(&state[threadID]) * sigma1);
    gaussian_arr2[threadID] = (float) (curand_normal(&state[threadID]) * sigma2);
}


__device__ float alt_calc_det(float* cov_arr, int n) {

    int i;
    float det = 1.f;

    int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    int initial_idx = n * threadID;

    for (i = initial_idx; i < initial_idx + n; i++) {
        det *= cov_arr[i];
    }

    return det;
}


__device__ float calc_vec_mat_vec_prod(float* cov_arr, float* data, float* mu_arr, int data_idx, int n) {

    int i, j;
    
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    int initial_idx = n * threadID;
    float diff;
    float cum_sum = 0.f;

    for (i = initial_idx; i < initial_idx + n; i++) {

        j = i - initial_idx + data_idx;

        diff = data[j] - mu_arr[i];
        cum_sum += diff * diff * 1. / cov_arr[i];
    }

    return cum_sum;
}


__device__ float get_log_det(float* A, int n) {

    float det = alt_calc_det(A, n);

    return log(det);
}


__device__ float get_log_likelihood(float* data, float* mu_arr, float* cov_arr, float cov_det, int data_idx, int n) {
    // Log likelihood, assuming Gaussian errors.
    
    float t1, t2, t3;
    float L;
    float fl_inf = 10000000000000000000;

    t1 = -0.5 * cov_det;
    t2 = -0.5 * calc_vec_mat_vec_prod(cov_arr, data, mu_arr, data_idx, n);
    t3 = -0.5 * n * log(2 * M_PI);

    // printf("Terms: %f, %f, %f\n", t1, t2, t3);

    L = t1 + t2 + t3;

    if (isnan(L)) {
        return -1 * fl_inf;
    } else {
        return L;
    }
}


__device__ float get_total_log_likelihood(float* cov, float* mu, float* data, int n_data_per_thread, int n_data, int n) {
    // Gets the log likelihood for all of the data files, summing the result.

    int data_idx;
    float cov_det = get_log_det(cov, n);

    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int data_offset = threadID * n * n_data_per_thread;
    float cum_L = 0.f;

    for (data_idx = data_offset; data_idx < data_offset + n * n_data_per_thread; data_idx+=n) {
       cum_L += get_log_likelihood(data, mu, cov, cov_det, data_idx, n);
    }

    return cum_L;
}


__device__ void generate_random_nums(curandState *state, float* rand_mu, float* rand_cov, float mu_step, float cov_step, float* rand_num, int* rand_ints, int n) {

    int n_params = 2 * n;

    get_rand_nums(state, rand_num, rand_ints, n_params);

    get_rand_gaussian(state, mu_step, rand_mu, cov_step, rand_cov);
}


__device__ void perturb_cov(float* old_cov, float* new_cov, int param_idx, float rand_cov_num, int n) {

    float new_val;

    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = threadID * n;
    int idx = param_idx - n;

    new_val = old_cov[offset + idx] + rand_cov_num;

    if (new_val > 0) {
        new_cov[offset + idx] = new_val;
    }
}


__device__ void perturb_params(float* old_cov, float* old_mu, float* new_cov, float* new_mu, int* rand_ints, float* rand_mu, float* rand_cov, int n) {
    // Perturbs a random parameter from array params from a Gaussian distribution 
    // with a standard deviation of its step-size in array step_size. Returns a new
    // parameter vector params.
    
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = threadID * n;
    // Pick parameter to perturb
    int param_idx = rand_ints[threadID];
    // printf("Random interger: %d.\n", param_idx);

    if (param_idx < n) {
        new_mu[param_idx + offset] = old_mu[param_idx + offset] + rand_mu[threadID];
    } else {
        perturb_cov(old_cov, new_cov, param_idx, rand_cov[threadID], n);
    }
}


__device__ void mcmc_step(float* curr_L, float* new_cov, float* new_mu, float* old_cov, float* old_mu, int* rand_ints, float* rand_mu, float* rand_cov, float* rand_num, int n, int n_data_per_thread, int n_data, int* take_step, float* data) {

    int threadID = threadIdx.x + blockDim.x * blockIdx.x;

    float old_L = curr_L[threadID];

    float new_L = get_total_log_likelihood(new_cov, new_mu, data, n_data_per_thread, n_data, n);

    float threshold;

    // Note: might need to restrict elements of cov to keep matrix positive semi-definite
    if (new_L > old_L) {

        take_step[threadID] = 1;
        curr_L[threadID] = new_L;

    } else {

        threshold = exp(new_L - old_L);

        if (rand_num[threadID] < threshold) {
            take_step[threadID] = 1;
            curr_L[threadID] = new_L;
        } else {
            take_step[threadID] = 0;
        }
    }
}


__device__ float calc_l2_norm(float* mu, float* true_mu, int n) {

    int i;
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = n * threadID;

    float diff;
    float cum_sum = 0.;
    float result;

    for (i = offset; i < offset + n; i++) {
        diff = true_mu[i] - mu[i];
        cum_sum += diff * diff;
    }

    result = sqrt(cum_sum);

    return result;
}


__device__ void initialize_cov(float* cov, int n) {

    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = threadID * n;
    float val = 5.0f;

    int i;

    for (i = offset; i < offset + n; i++) {
        cov[i] = val;
    }
}


__device__ void initialize_means(float initial_mean, float* curr_mu, int n) {

    int i;
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = threadID * n;

    for (i = offset; i < offset + n; i++) {
        curr_mu[i] = initial_mean;
    }
}


__host__ int count_n_data(int n) {

    FILE *input;
    unsigned int i = 0;

    float temp;
   
    // input = fopen("data_10.txt", "r");
    char input_file [50];
    sprintf(input_file, "data_%d.txt", n);
    input = fopen(input_file, "r");

    while ((fscanf(input, "%f", &temp)) != EOF) {
        i++;
    }
        
    fclose(input);

    return i / n;
}

__host__ void read_data(int n_data, int n, float* data) {

    FILE *input;
    unsigned int i = 0;
   
    // input = fopen("data_10.txt", "r");
    char input_file [50];
    sprintf(input_file, "data_%d.txt", n);
    input = fopen(input_file, "r");

    while ((fscanf(input, "%f", &data[i])) != EOF) {
        i++;
    }
        
    fclose(input);
}


__device__ void vec_cpy(float* parent_mat, float* child_mat, int n) {

    int threadID = threadIdx.x + blockDim.x * blockIdx.x;

    int offset = threadID * n;

    int i;

    for (i = offset; i < offset + n; i++) {
        child_mat[i] = parent_mat[i];
    }
}


__global__ void kernel_rand_nums(curandState* state, float* rand_mu, float* rand_cov, float mu_step, float cov_step, float* rand_num, int* rand_ints, int n) {

        generate_random_nums(state, rand_mu, rand_cov, mu_step, cov_step, rand_num, rand_ints, n);

}


__global__ void mcmc_shared(int n, int n_data, int n_data_per_thread, int n_threads, int n_steps, int spacing, float mu_step, float cov_step, curandState* state, float* curr_cov, float* new_cov, float* curr_mu, float* new_mu, float* rand_num, int* rand_ints, float* rand_mu, float* rand_cov, float* curr_L, int* take_step, float* data, float* mu_inv_arr, float* cov_inv_arr, float* mu_est, float* cov_est, float* all_mu_est) {

    int step_count = 0;
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float s_data[];

    float initial_mean = 1.0f;
    int local_take_step;

    int estimation_offset;
    int i;
    int copy_offset = threadID * n * n_data_per_thread;

    for (i = copy_offset; i < copy_offset + n_data_per_thread * n; i++) {
        s_data[i] = data[i];
    }

    // begin code
    initialize_cov(curr_cov, n);
    initialize_means(initial_mean, curr_mu, n);
    rand_init(state);

    curr_L[threadID] = get_total_log_likelihood(curr_cov, curr_mu, s_data, n_data_per_thread, n_data, n);

    vec_cpy(curr_cov, new_cov, n);
    vec_cpy(curr_mu, new_mu, n);

    while (step_count < n_steps) {

        generate_random_nums(state, rand_mu, rand_cov, mu_step, cov_step, rand_num, rand_ints, n);

        perturb_params(curr_cov, curr_mu, new_cov, new_mu, rand_ints, rand_mu, rand_cov, n);

        mcmc_step(curr_L, new_cov, new_mu, curr_cov, curr_mu, rand_ints, rand_mu, rand_cov, rand_num, n, n_data_per_thread, n_data, take_step, s_data);

        local_take_step = take_step[threadID];

        if (local_take_step == 1) {

            vec_cpy(new_cov, curr_cov, n);
            vec_cpy(new_mu, curr_mu, n);
        } 

        if (step_count % spacing == 0) {
           
            calc_cov_est(curr_cov, cov_inv_arr, cov_est, n, n_threads);

            calc_mu_est(curr_mu, cov_inv_arr, mu_inv_arr, cov_est, mu_est, n, n_threads);

            if (threadID == 0) {
                estimation_offset = step_count / spacing * n;

                // ERROR
                for (i = estimation_offset; i < estimation_offset + n; i++) {
                    all_mu_est[i] = mu_est[i - estimation_offset];
                }
            }

            __syncthreads();

        }

        step_count += 1;
    }
}



__global__ void mcmc(int n, int n_data, int n_data_per_thread, int n_threads, int n_steps, int spacing, float mu_step, float cov_step, curandState* state, float* curr_cov, float* new_cov, float* curr_mu, float* new_mu, float* rand_num, int* rand_ints, float* rand_mu, float* rand_cov, float* curr_L, int* take_step, float* data, float* mu_inv_arr, float* cov_inv_arr, float* mu_est, float* cov_est, float* all_mu_est) {

    int step_count = 0;
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;

    float initial_mean = 1.0f;
    int local_take_step;

    int estimation_offset;
    int i;

    // begin code
    initialize_cov(curr_cov, n);
    initialize_means(initial_mean, curr_mu, n);
    rand_init(state);

    curr_L[threadID] = get_total_log_likelihood(curr_cov, curr_mu, data, n_data_per_thread, n_data, n);

    vec_cpy(curr_cov, new_cov, n);
    vec_cpy(curr_mu, new_mu, n);

    while (step_count < n_steps) {

        generate_random_nums(state, rand_mu, rand_cov, mu_step, cov_step, rand_num, rand_ints, n);

        perturb_params(curr_cov, curr_mu, new_cov, new_mu, rand_ints, rand_mu, rand_cov, n);

        mcmc_step(curr_L, new_cov, new_mu, curr_cov, curr_mu, rand_ints, rand_mu, rand_cov, rand_num, n, n_data_per_thread, n_data, take_step, data);

        local_take_step = take_step[threadID];

        if (local_take_step == 1) {

            vec_cpy(new_cov, curr_cov, n);
            vec_cpy(new_mu, curr_mu, n);

        } 

        if (step_count % spacing == 0) {
           
            calc_cov_est(curr_cov, cov_inv_arr, cov_est, n, n_threads);

            calc_mu_est(curr_mu, cov_inv_arr, mu_inv_arr, cov_est, mu_est, n, n_threads);

            if (threadID == 0) {
                estimation_offset = step_count / spacing * n;

                // ERROR
                for (i = estimation_offset; i < estimation_offset + n; i++) {
                    all_mu_est[i] = mu_est[i - estimation_offset];
                }
            }

            __syncthreads();

        }

        step_count += 1;
    }
}


void print_usage() {

    printf("Usage: MCMC options are...\n");
    printf("    Number of steps: --n_steps=n_steps\n");
    printf("    Number of dimensions: --n_dim=n_dim\n");
    printf("    Evaluation frequency: --eval_freq=eval_freq\n");
    printf("    Store data in shared memory (requires small datasets): --shared_memory=1\n");
}


int query_flag_idx(char* arg, const char* query_array[]) {

    int unique_idx = 5;
    int i;

    for (i = 0; i < 4; i++) {
        if (arg[unique_idx] == query_array[i][unique_idx]) {
            return i;
        }
    }

    return -1;
}


int* parse_args(int argc, char *argv[]) {

    const char* query_array[4];
    query_array[0] = "--n_steps";
    query_array[1] = "--n_dim";
    query_array[2] = "--eval_freq";
    query_array[3] = "--shared_memory";

    int* arg_vals = (int*)malloc(4 * sizeof(int));

    int default_n_steps = 10000;
    int default_n = 10;
    int default_spacing = 1000;
    int default_shared = 0;

    int i;
    int arg_idx;
    int init_idx;

    for (i = 0; i < 4; i++) {
        arg_vals[i] = -1;
    }

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage();
            exit(0);
        } else {
            arg_idx = query_flag_idx(argv[i], query_array);

            if (arg_idx != -1) {
                init_idx = strlen(query_array[arg_idx]) + 1;
                argv[i] += init_idx;
                sscanf(argv[i], "%d", &arg_vals[arg_idx]);
            } else {
                print_usage();
                exit(0);
            }
        }
    }

    if (arg_vals[0] == -1) {
        printf("Number of steps not provided. Defaulting to %d steps.\n", default_n_steps);
        arg_vals[0] = default_n_steps;
    }
    if (arg_vals[1] == -1) {
        printf("Number of dimensions not provided. Defaulting to %d dimensions.\n", default_n);
        arg_vals[1] = default_n;
    }
    if (arg_vals[2] == -1) {
        printf("Evaluation frequency not provided. Defaulting to evaluating every %d steps.\n", default_spacing);
        arg_vals[2] = default_spacing;
    }
    if (arg_vals[3] == -1) {
        printf("Shared memory not specified. Defaulting to using global memory.\n");
        arg_vals[3] = default_shared;
    }

    return arg_vals;
}


int main(int argc, char *argv[]) {

    int* arg_vals = parse_args(argc, argv);
    int n_steps = arg_vals[0];
    int n = arg_vals[1];
    int spacing = arg_vals[2];
    int use_shared = arg_vals[3];

    int n_threads = 1024;
    int n_data = count_n_data(n);
    int n_data_per_thread = n_data / n_threads;
    int n_blocks = 1;

    int n_sample_points = n_steps / spacing;
    int i;

    float mu_step = 0.2;
    float cov_step = 0.2;

    FILE* f;

    cudaEvent_t start, stop;
    curandState* state;

    float* rand_num;
    float* rand_mu;
    float* rand_cov;
    int* rand_ints;

    float* curr_mu;
    float* new_mu;
    float* curr_cov;
    float* new_cov;

    float* curr_L;
    int* take_step;

    float* data;

    float* mu_inv;
    float* cov_inv;

    float* mu_est;
    float* cov_est;

    float* all_mu_est;

    float* h_data = (float*)malloc(n * n_data * sizeof(float));
    float* h_cov = (float*)malloc(n * n_threads * n_blocks * sizeof(float));
    float* h_mu = (float*)malloc(n * n_threads * n_blocks * sizeof(float));
    float* h_rand = (float*)malloc(n_threads * n_blocks * sizeof(float));
    
    float* h_mu_est = (float*)malloc(n * n_sample_points * sizeof(float));

    CudaSafeCall(cudaMalloc(&state, n_blocks * n_threads * sizeof(curandState)));

    CudaSafeCall(cudaMalloc((void**)&rand_num, n_blocks * n_threads * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&rand_mu, n_blocks * n_threads * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&rand_cov, n_blocks * n_threads * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&rand_ints, n_blocks * n_threads * sizeof(int)));

    CudaSafeCall(cudaMalloc((void**)&curr_L, n_threads * n_blocks * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&take_step, n_threads * n_blocks * sizeof(int)));

    CudaSafeCall(cudaMalloc((void**)&curr_mu, n * n_threads * n_blocks * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&new_mu, n * n_threads * n_blocks * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&curr_cov, n * n_threads * n_blocks * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&new_cov, n * n_threads * n_blocks * sizeof(float)));

    CudaSafeCall(cudaMalloc((void**)&data, n * n_data * sizeof(float)));

    CudaSafeCall(cudaMalloc((void**)&mu_inv, n * n_blocks * n_threads * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&cov_inv, n * n_threads * n_blocks * sizeof(float)));

    CudaSafeCall(cudaMalloc((void**)&mu_est, n * sizeof(float)));
    CudaSafeCall(cudaMalloc((void**)&cov_est, n * sizeof(float)));

    CudaSafeCall(cudaMalloc((void**)&all_mu_est, n * n_sample_points * sizeof(float)));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Array for covariance elements held by every thread
    read_data(n_data, n, h_data);

    CudaSafeCall(cudaMemcpy(data, h_data, n * n_data * sizeof(float), cudaMemcpyHostToDevice));

    // kernel_rand_nums<<<1, 1024>>>(state, rand_mu, rand_cov, mu_step, cov_step, rand_num, rand_ints, n);

    //cudaEventRecord(start);
    //mcmc<<<n_blocks, n_threads>>>(n, n_data, n_data_per_thread, n_threads, n_steps, spacing, mu_step, cov_step, state, curr_cov, new_cov, curr_mu, new_mu, rand_num, rand_ints, rand_mu, rand_cov, curr_L, take_step, data, mu_inv, cov_inv, mu_est, cov_est, all_mu_est);
    //cudaEventRecord(stop);
    //CudaCheckError();

    if (use_shared == 1) {
        cudaEventRecord(start);
        mcmc_shared<<<n_blocks, n_threads, n * n_data * sizeof(float)>>>(n, n_data, n_data_per_thread, n_threads, n_steps, spacing, mu_step, cov_step, state, curr_cov, new_cov, curr_mu, new_mu, rand_num, rand_ints, rand_mu, rand_cov, curr_L, take_step, data, mu_inv, cov_inv, mu_est, cov_est, all_mu_est);
        cudaEventRecord(stop);
        CudaCheckError();
    } else {
        cudaEventRecord(start);
        mcmc<<<n_blocks, n_threads>>>(n, n_data, n_data_per_thread, n_threads, n_steps, spacing, mu_step, cov_step, state, curr_cov, new_cov, curr_mu, new_mu, rand_num, rand_ints, rand_mu, rand_cov, curr_L, take_step, data, mu_inv, cov_inv, mu_est, cov_est, all_mu_est);
        cudaEventRecord(stop);
        CudaCheckError();
    }


    CudaSafeCall(cudaMemcpy(h_cov, curr_cov, n * n_threads * sizeof(float), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(h_mu, curr_mu, n * n_threads * sizeof(float), cudaMemcpyDeviceToHost));

    CudaSafeCall(cudaMemcpy(h_mu_est, all_mu_est, n * n_sample_points * sizeof(float), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(h_rand, rand_num, n_threads * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);

//    for (i = 0; i < n * n_threads; i++) {
//        printf("%f ", h_mu[i]);
//    }

    float milliseconds = 0.f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Code executed in %f ms.\n", milliseconds);

    f = fopen("mu_data.txt", "w");
    for (i = 0; i < n * n_threads; i++) {
        fprintf(f, "%f ", h_mu[i]);
    }
    fclose(f);

    f = fopen("cov_data.txt", "w");
    for (i = 0; i < n * n_threads; i++) {
        fprintf(f, "%f ", h_cov[i]);
    }
    fclose(f);

    f = fopen("rand_data.txt", "w");
    for (i = 0; i < n_threads; i++) {
        fprintf(f, "%f ", h_rand[i]);
    }
    fclose(f);

    f = fopen("mu_evolution.txt", "w");
    for (i = 0; i < n_sample_points * n; i++) {
        fprintf(f, "%f ", h_mu_est[i]);
        // printf("%f ", h_mu_est[i]);
    }
    fclose(f);


    free(h_data);
    free(h_cov);
    free(h_mu);
    free(h_rand);

    cudaFree(state);

    free(h_mu_est);

    cudaFree(mu_inv);
    cudaFree(cov_inv);

    cudaFree(mu_est);
    cudaFree(cov_est);

    cudaFree(all_mu_est);

    cudaFree(rand_num);
    cudaFree(rand_mu);
    cudaFree(rand_cov);
    cudaFree(rand_ints);

    cudaFree(curr_L);
    cudaFree(take_step);

    cudaFree(curr_mu);
    cudaFree(new_mu);
    cudaFree(curr_cov);
    cudaFree(new_cov);

    cudaFree(data);

    return 0;
}
