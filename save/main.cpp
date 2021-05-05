#include <iostream>
#include <cmath>
#include <random>
#include "eigen/Eigen/Dense"
#include "new_util.h"
#include <cstdio>
#include <fstream>
#include <omp.h>

using eigmat = Eigen::MatrixXd;
using value_t = double;
using state_t = Eigen::VectorXd;
using eigmat = Eigen::MatrixXd;

using de_system = Lorenz;

int main() {

    double s = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    double start, end;
    start = omp_get_wtime();
    /*---Setup---*/
    std::default_random_engine engine;
    std::normal_distribution<double> n01(0, 1.0);
    eigmat y_observe, X_analysis_mean, x_background_mean, x_background_deviation, X_error, x_observe, observe_covariance,
            x_analysis, L, Z_b, x_background, H, z_b, z_b_mean, Y, K_intermediate, tmp, x_truth, L_model, exact_L;

    /*---Parameters---*/
    double t = 0.0;
    double t_end = 0.12;
    double dt = 0.002;
    double inflation = 1.01;
    int N_ens = 32;
    double invSqrtNens = 1 / sqrt(N_ens - 1); //inverse square root of number of ensembles
    double sqrtNens = sqrt(N_ens - 1);
    int o = 1; // Number of observed variables
    long steps = 1000;
    double eta = sqrt(1);
    int numThreads = 16;

    /*---INIT---*/
    eigmat x_true(vars, 1); //set definition at the top for vars
    x_true << -5.9, -5.6, 24.4;

    eigmat X_analysis(x_true.size(), steps + 1);
    eigmat X_True(x_true.size(), steps + 1);
    x_truth = x_true;
    x_truth += perturbedM(x_true.size(), 1, n01, engine);
    setColumn(X_True, 0, x_truth);
    observe_covariance = (eta * eta) * I(o); //precalc
    L = cholesky(observe_covariance); //precalc

    x_background = x_truth.replicate(1, N_ens) + perturbedM(x_true.size(), N_ens, n01, engine);
    setColumn(X_analysis, 0, mean(x_background));
    x_analysis = x_background;
    H = I(o, x_true.size()); //precalc


    Eigen::initParallel();
    omp_set_nested(1);
    omp_set_num_threads(64);
    Eigen::setNbThreads(64);
    printf("\nNumber of threads: %d\n", omp_get_num_threads());

    end = omp_get_wtime();
    s += ms_difference(start, end);
    #pragma omp parallel num_threads(64)
    {
        printf("\nNumber of threads: %d\n", omp_get_num_threads());

        for (long long i = 1; i <= steps; i++) {

            /*---------1st Section---------*/
            start = omp_get_wtime();
            #pragma omp single
            {
                x_truth = ode(vectorize(x_truth), t, t_end, dt);
            }
            end = omp_get_wtime();
            s1 += ms_difference(start, end);
            /*-----------------------------*/


            /*---------2nd Section---------*/
            start = omp_get_wtime();
            #pragma omp sections
            {
                #pragma omp section
                {
                    setColumn(X_True, i, x_truth);
                }

                #pragma omp section
                {
                    y_observe = (H * x_truth) + (L * perturbedM(o, 1, n01, engine));
                    Y = y_observe.replicate(1, N_ens);
                    exact_L = L * perturbedM(o, N_ens, n01, engine);
                    Y += exact_L - mean(exact_L).replicate(1, N_ens);
                }
            }
            end = omp_get_wtime();
            s2 += ms_difference(start, end);
            /*-----------------------------*/


            /*---------3rd Section---------*/
            start = omp_get_wtime();
            #pragma omp for
            for (long long j = 0; j < N_ens; j++) {
                setColumn(x_background, j, ode(vectorize(col(x_analysis, j)), t, t_end, dt));
            }
            end = omp_get_wtime();
            s3 += ms_difference(start, end);
            /*-----------------------------*/


            /*---------4th Section---------*/
            start = omp_get_wtime();
            #pragma omp single
            {
                x_background_mean = mean(x_background);
                x_background_deviation = invSqrtNens * (x_background - x_background_mean * ones(1, N_ens)) * inflation;
                x_background = x_background_mean.replicate(1, N_ens) + sqrtNens * x_background_deviation;
                x_observe = H * x_background;

                Z_b = invSqrtNens * (x_observe - mean(x_observe) * ones(1, N_ens));
                K_intermediate = (Z_b * Z_b.transpose() + observe_covariance).colPivHouseholderQr().solve(
                        Y - x_observe);
                x_analysis = x_background + (x_background_deviation * Z_b.transpose()) * K_intermediate;
                X_analysis_mean = mean(x_analysis);
                setColumn(X_analysis, i, X_analysis_mean);
            }
            end = omp_get_wtime();
            s4 += ms_difference(start, end);
            /*-----------------------------*/
        }
    }
    //p(x_truth);
    //p(x_analysis);
    //p(mean(x_analysis));
    X_error = (X_analysis - X_True);

    double rmse = sqrt(((double) X_error.squaredNorm()) / (double) X_error.size());//throw first 500-1000
    print_result(s, s1, s2, s3, s4, steps, N_ens, numThreads, rmse, X_analysis, X_True, X_error);

    return 0;
}
