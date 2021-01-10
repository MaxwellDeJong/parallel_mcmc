#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fstream>
#include <mpi.h>
#include "mcmc_cpu.h"

using namespace std;


float alt_calc_det(float** a, int n) {

    int i;
    float det = 0;

    for (i = 0; i < n; i++) {
        det += a[i][i] * a[i][i];
    }

    return det;
}


void calc_mat3_inv(float** mat3, float** mat_inv, int n) {

    int i;

    for (i = 0; i < n; i++) {
        mat_inv[i][i] = 1. / mat3[i][i];
    }
}


void calc_mat_vec_prod(float** mat, float* x, float* mu, float* diff_vec, float* new_vec, int n) {

    int i, j;

    for (i = 0; i < n; i++) {
        diff_vec[i] = x[i] - mu[i];
        new_vec[i] = 0.;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            new_vec[i] += mat[i][j] * diff_vec[j];
        }
    }
}


float calc_dot_prod(float* vec1, float* vec2, int n) {

    int i;
    float cum_sum = 0;

    for (i = 0; i < n; i++) {
        cum_sum += vec1[i] * vec2[i];
    }

    return cum_sum;
}


float get_log_det(float** A, int n) {

    float det = alt_calc_det(A, n);

    return log(det);
}


float get_log_likelihood(float* x, float* mu, float* new_vec, float* diff_vec, float** cov, float** cov_inv, float cov_det, int n) {
    // Log likelihood, assuming Gaussian errors.
    
    float t1, t2, t3;
    float L;
    float fl_inf = 1000000000000;

    calc_mat_vec_prod(cov_inv, x, mu, diff_vec, new_vec, n);

    t1 = -0.5 * cov_det;
    t2 = -0.5 * calc_dot_prod(diff_vec, new_vec, n);
    t3 = -0.5 * n * log(2 * M_PI);

    L = t1 + t2 + t3;

    if (isnan(L)) {
        return -1 * fl_inf;
    } else {
        return L;
    }

    return L;
}


float get_total_log_likelihood(float** cov, float** cov_inv, float* curr_x, float* mu, float* diff_vec, float* new_vec, float** data, int n_data, int n) {
    // Gets the log likelihood for all of the data files, summing the result.

    float cov_det = get_log_det(cov, n);

    int i;
    float cum_L = 0.;

    calc_mat3_inv(cov, cov_inv, n);

    for (i = 0; i < n_data; i++) {
        curr_x = data[i];
        cum_L += get_log_likelihood(curr_x, mu, new_vec, diff_vec, cov, cov_inv, cov_det, n);
    }

    return cum_L;
}


void host_generate_random_nums(float* mu_rand, float* cov_rand, float mu_step, float cov_step, float* rand_num, int* rand_ints, int max_steps, gsl_rng *r, int n) {

    int i;
    int n_params = 2 * n;

    for (i = 0; i < max_steps; i++) {
        mu_rand[i] = gsl_ran_gaussian(r, mu_step);
    }

    for (i = 0; i < max_steps; i++) {
        cov_rand[i] = gsl_ran_gaussian(r, cov_step);
    }

    for (i = 0; i < max_steps; i++) {
        rand_num[i] = gsl_rng_uniform(r);
    }

    for (i = 0; i < max_steps; i++) {
        rand_ints[i] = (int) (n_params * gsl_rng_uniform(r));
    }

}


void perturb_cov(float** old_cov, float** new_cov, int param_idx, float cov_rand_num, int n) {

    float new_val;

    int idx = param_idx - n;

    new_val = old_cov[idx][idx] + cov_rand_num;

    if (new_val > 0) {
        new_cov[idx][idx] = new_val;
    }
}


void perturb_params(float** old_cov, float* old_mu, float** new_cov, float* new_mu, int* rand_ints, float* mu_rand, float* cov_rand, int n, int iter) {
    // Perturbs a random parameter from array params from a Gaussian distribution 
    // with a standard deviation of its step-size in array step_size. Returns a new
    // parameter vector params.
    
    // Pick parameter to perturb
    int param_idx = rand_ints[iter];
    // printf("Random interger: %d.\n", param_idx);

    if (param_idx < n) {
        new_mu[param_idx] = old_mu[param_idx] + mu_rand[iter];
    } else {
        perturb_cov(old_cov, new_cov, param_idx, cov_rand[iter], n);
    }
}


void mcmc_step(float &curr_L, float** new_cov, float* new_mu, float** old_cov, float* old_mu, float** cov_inv, float* curr_x, float* diff_vec, float* new_vec, int* rand_ints, float* mu_rand, float* cov_rand, float* rand_num, int n, int n_data, int &take_step, int iter, float** data) {

    float old_L = curr_L;
    float new_L = get_total_log_likelihood(new_cov, cov_inv, curr_x, new_mu, diff_vec, new_vec, data, n_data, n);

    float threshold;

    // Note: might need to restrict elements of cov to keep matrix positive semi-definite
    if (new_L > old_L) {

        take_step = 1;
        curr_L = new_L;

    } else {

        threshold = exp(new_L - old_L);

        if (rand_num[iter] < threshold) {
            take_step = 1;
            curr_L = new_L;
        } else {
            take_step = 0;
        }
    }
}


float calc_l2_norm(float* mu, float* true_mu, int n) {

    int i;
    float diff;
    float cum_sum = 0.;

    for (i = 0; i < n; i++) {
        diff = true_mu[i] - mu[i];
        cum_sum += diff * diff;
    }

    return sqrt(cum_sum);
}


void initialize_cov(float** cov, gsl_rng *r) { 

    int i, j;

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            cov[i][j] = 0.;
        }
        cov[i][i] = 3.0;
    }
}


void read_data(int n_data, int n, float** data) {

    int i, j;

    ifstream in("data_3.txt");

    if (!in) {
        printf("Cannot open file.\n");
        return;
    }

    for (i = 0; i < n_data; i++) {
        for (j = 0; j < n; j++) {
            in >> data[i][j];
        }
    }

    in.close();
}


void mat_cpy(float** parent_mat, float** child_mat, int n) {

    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            child_mat[i][j] = parent_mat[i][j];
        }
    }
}


void vec_cpy(float* parent_vec, float* child_vec, int n) {
    
    int i;

    for (i = 0; i < n; i++) {
        child_vec[i] = parent_vec[i];
    }
}

void send_data(int n_data, int n_local_data, int n, int n_procs, float** data, float** local_data, float* temp_arr) {

    int i, j;
    int row, col;
    int offset;

    for (i = 1; i < n_procs; i++) {

        offset = i * n_local_data * n;

        for (j = 0; j < n_local_data * n; j++) {

            row = (offset + j) / n;
            col = (offset + j) % n;

            temp_arr[j] = data[row][col];

        }

        MPI_Send(&temp_arr[0], n_local_data * n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }

    // Distribute local data
    for (i = 0; i < n_local_data; i++) {
        for (j = 0; j < n; j++) {
            local_data[i][j] = data[i][j];
        }
    }
}


void receive_data(int n_local_data, int n, float** local_data, float* tmp_arr) {

    int i, j, idx;

    MPI_Recv(&tmp_arr[0], n_local_data * n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (i = 0; i < n_local_data; i++) {
        for (j = 0; j < n; j++) {

            idx = i * n + j;
            local_data[i][j] = tmp_arr[idx];
        }
    }

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


void print_usage() {

    printf("Usage: MCMC options are...\n");
    printf("    Number of steps: --n_steps=n_steps\n");
    printf("    Number of dimensions: --n_dim=n_dim\n");
}


int* parse_args(int argc, char *argv[], int ID) {

    const char* query_array[2];
    query_array[0] = "--n_steps";
    query_array[1] = "--n_dim";

    int* arg_vals = (int*)malloc(2 * sizeof(int));

    int default_n_steps = 10000;
    int default_n = 10;

    int i;
    int arg_idx;
    int init_idx;

    for (i = 0; i < 2; i++) {
        arg_vals[i] = -1;
    }

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            if (ID == 0) {
                print_usage();
            }
            MPI_Finalize();
            exit(0);
        } else {
            arg_idx = query_flag_idx(argv[i], query_array);

            if (arg_idx != -1) {
                init_idx = strlen(query_array[arg_idx]) + 1;
                argv[i] += init_idx;
                sscanf(argv[i], "%d", &arg_vals[arg_idx]);
            } else {
                if (ID == 0) {
                    print_usage();
                }
                MPI_Finalize();
                exit(0);
            }
        }
    }

    if (arg_vals[0] == -1) {
        if (ID == 0) {
            printf("Number of steps not provided. Defaulting to %d steps.\n", default_n_steps);
        }
        arg_vals[0] = default_n_steps;
    }
    if (arg_vals[1] == -1) {
        if (ID == 0) {
            printf("Number of dimensions not provided. Defaulting to %d dimensions.\n", default_n);
        }
        arg_vals[1] = default_n;
    }

    return arg_vals;
}


int count_n_data(int n) {

    FILE *input;
    unsigned int i = 0;

    float temp;
   
    char input_file [50];
    sprintf(input_file, "data_%d.txt", n);
    input = fopen(input_file, "r");

    while ((fscanf(input, "%f", &temp)) != EOF) {
        i++;
    }
        
    fclose(input);

    return i / n;
}


void mcmc(int argc, char* argv[], float mu_step, float cov_step) {

    int ID;
    int n_procs;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &n_procs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &ID);

    int* arg_vals = parse_args(argc, argv, ID);
    int n_steps = arg_vals[0];
    int n = arg_vals[1];

    if (ID == 0) {
        printf("Beginning MCMC. Total number of processors: %d\n", n_procs);
    }


    fflush(stdout);

    int step_count = 1;

    float initial_mean = 1.0;

    float curr_L;
    int take_step;

    int n_data = count_n_data(n);

    int n_local_data = n_data / n_procs;

    float* rand_num = (float*) malloc(n_steps * sizeof(float));
    float* mu_rand = (float*) malloc(n_steps * sizeof(float));
    float* cov_rand = (float*) malloc(n_steps * sizeof(float));

    int* rand_ints = (int*) malloc(n_steps * sizeof(int));
    float* norm_arr = (float*) malloc(n_steps * sizeof(float));

    const gsl_rng_type *T;
    gsl_rng *r;

    int i, j;

    float* true_mu = (float*)malloc(n * sizeof(float));

    float* curr_mu = (float*)malloc(n * sizeof(float));
    float* new_mu = (float*)malloc(n * sizeof(float));

    float* curr_x = (float*)malloc(n * sizeof(float));
    float* diff_vec = (float*)malloc(n * sizeof(float));
    float* new_vec = (float*)malloc(n * sizeof(float));

    float** curr_cov = (float**)malloc(n * sizeof(float*));
    float** new_cov = (float**)malloc(n * sizeof(float*));
    float** cov_inv = (float**)malloc(n * sizeof(float*));

    float** data;
    float** local_data = (float**)malloc(n_local_data * sizeof(float*));
    float* temp_arr = (float*)malloc(n_local_data * n * sizeof(float));

    float** means_arr = (float**)malloc(n * sizeof(float*));

    char buffer[30];
    int n_char;

    for (i = 0; i < n; i++) {
        curr_cov[i] = (float*)malloc(n * sizeof(float));
        new_cov[i] = (float*)malloc(n * sizeof(float));
        cov_inv[i] = (float*)malloc(n * sizeof(float));
    }

    if (ID == 0) {
        data = (float**)malloc(n_data * sizeof(float*));

        for (i = 0; i < n_data; i++) {
            data[i] = (float*)malloc(n * sizeof(float));
        }
    }

    for (i = 0; i < n_local_data; i++) {
        local_data[i] = (float*)malloc(n * sizeof(float));
    }

    for (i = 0; i < n; i++) {
        means_arr[i] = (float*)malloc(n_steps * sizeof(float));
    }

    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    for (i = 0; i < n; i++) {
        // true_mu[i] = i + 1.;
        true_mu[i] = 5.f;
        curr_mu[i] = initial_mean;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            cov_inv[i][j] = 0.f;
        }
    }

    host_generate_random_nums(mu_rand, cov_rand, mu_step, cov_step, rand_num, rand_ints, n_steps, r, n);

    initialize_cov(curr_cov, r);

    if (ID == 0) {

        read_data(n_data, n, data);

        send_data(n_data, n_local_data, n, n_procs, data, local_data, temp_arr);

    } else {

        receive_data(n_local_data, n, local_data, temp_arr);
    }

//    printf("Local data for processor %d:\n", ID);
//    fflush(stdout);
//    for (i = 0; i < n_local_data; i++) {
//        for (j = 0; j < n; j++) {
//            printf("%f ", local_data[i][j]);
//            fflush(stdout);
//        }
//        printf("\n");
//        fflush(stdout);
//    }

    curr_L = get_total_log_likelihood(curr_cov, cov_inv, curr_x, curr_mu, diff_vec, new_vec, local_data, n_local_data, n);

    mat_cpy(curr_cov, new_cov, n);
    vec_cpy(curr_mu, new_mu, n);

    norm_arr[0] = calc_l2_norm(curr_mu, true_mu, n);

    while (step_count < n_steps) {

        perturb_params(curr_cov, curr_mu, new_cov, new_mu, rand_ints, mu_rand, cov_rand, n, step_count);

        mcmc_step(curr_L, new_cov, new_mu, curr_cov, curr_mu, cov_inv, curr_x, diff_vec, new_vec, rand_ints, mu_rand, cov_rand, rand_num, n, n_local_data, take_step, step_count, local_data);

        if (take_step == 1) {

            mat_cpy(new_cov, curr_cov, n);
            vec_cpy(new_mu, curr_mu, n);

            norm_arr[step_count] = calc_l2_norm(curr_mu, true_mu, n);
//            printf("new norm: %f\n", norm_arr[step_count]);

        } else {
            norm_arr[step_count] = norm_arr[step_count-1];
        }

        for (i = 0; i < n; i++) {
            means_arr[i][step_count-1] = curr_mu[i];
        }

        step_count += 1;
    }

//    printf("final norm: %f\n", norm_arr[n_steps-1]);
//
//    printf("final vector:");
//    for (i = 0; i < n; i++) {
//        printf("%f ", curr_mu[i]);
//    }
//    printf("\n");
//
//    printf("final covariance:");
//    for (i = 0; i < n; i++) {
//        for (j = 0; j < n; j++) {
//            printf("%f ", curr_cov[i][j]);
//        }
//        printf("\n");
//    }
//
//    printf("Final L: %f\n", curr_L);

    n_char = sprintf(buffer, "l2_norm_%d.txt", ID);

    ofstream output(buffer);
    if (output.is_open()) {

        for (i = 0; i < n_steps; i++) {
            output << norm_arr[i] << " ";
        }

        output.close();
    } else {
        printf("Unable to open output file for norm.\n");
    }

    for (j = 0; j < n; j++) {
        n_char = sprintf(buffer, "means_ID-%d_%d.txt", ID, j);

        ofstream mean_output(buffer);
        if (mean_output.is_open()) {

            for (i = 0; i < n_steps; i++) {
                mean_output << means_arr[j][i] << " ";
            }
    
            mean_output.close();
        } else {
            printf("Unable to open output file for means.\n");
        }
    }

//    for (i = 0; i < n_steps; i++) {
//        printf("%d ", rand_ints[i]);
//    }

    gsl_rng_free(r);

    free(norm_arr);

    free(true_mu);
    free(curr_mu);
    free(new_mu);

    free(curr_x);
    free(diff_vec);
    free(new_vec);

    free(mu_rand);
    free(cov_rand);
    free(rand_ints);
    free(rand_num);

    for (i = 0; i < n; i++) {
        free(curr_cov[i]);
        free(new_cov[i]);
        free(cov_inv[i]);
    }

    free(curr_cov);
    free(new_cov);
    free(cov_inv);

    for (i = 0; i < n; i++) {
        free(means_arr[i]);
    }

    free(means_arr);

    for (i = 0; i < n_local_data; i++) {
        free(local_data[i]);
    }

    free(local_data);

    if (ID == 0) {
        for (i = 0; i < n_data; i++) {
            free(data[i]);
        }

        free(data);
    }

    free(temp_arr);

    MPI_Finalize();

}


int main(int argc, char *argv[]) {

    float mu_step = 0.2;
    float cov_step = 0.2;

    mcmc(argc, argv, mu_step, cov_step);

    return 0;
}
