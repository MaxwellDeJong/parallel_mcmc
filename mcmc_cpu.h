#ifndef MCMC_CPU_H
#define MCMC_CPU_H  

float alt_calc_det(float**, int);
void calc_mat3_inv(float**, float);
void calc_mat_vec_prod(float**, float*, float*, float*, float*, int);
float calc_dot_prod(float*, float);
float get_log_det(float**, int) ;
float get_log_likelihood(float*, float*, float*, float*, float**, float**, float, int);
float get_total_log_likelihood(float**, float**, float*, float*, float*, float*, float**, int, int);
void host_generate_random_nums(float*, float*, float, float, float*, int*, int, gsl_rng*, int);
void perturb_cov(float**, float**, int, float, int);
void perturb_params(float**, float*, float**, float*, int*, float*, float*, int, int);
void mcmc_step(float&, float**, float*, float**, float*, float**, float*, float*, float*, int*, float*, float*, float*, int, int, int&, int, float**);
float calc_l2_norm(float*, float*, int);
void initialize_cov(float**, gsl_rng*);
void read_data(int, int, float**);
void mat_cpy(float**, float**, int);
void vec_cpy(float*, float*, int);
void send_data(int, int, int, int, float**, float**, float*);
void receive_data(int, int, float**, float*);
void mcmc(int, char* [], float, float);
void print_usage();
int query_flag_idx(char*, const char* []);
int count_n_data(int);
int* parse_args(int, char* [], int);

#endif
