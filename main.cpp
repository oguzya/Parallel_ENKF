#include "utilities.h"

using namespace Eigen;
using M = MatrixXd;
#define vars 3

int main() {

    /*---Setup---*/
    initParallel();
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    double start, end;
    std::default_random_engine engine;
    std::normal_distribution<double> n01(0, 1.0);
    M y_observe, X_analysis_mean, x_background_mean, x_background_deviation, X_error, x_observe, observe_covariance,
            x_analysis, L, Z_b, x_background, H, z_b, z_b_mean, Y, K_intermediate, tmp, x_perturbed, L_model;

    /*---Parameters---*/
    double t = 0.0;
    double dt = 0.0001;
    double t_end = 0.01;
    double inflation = 1.05;
    int N_ens = 50;
    double invSqrtNens = 1 / sqrt(N_ens - 1); //inverse square root of number of ensembles
    double sqrtNens = sqrt(N_ens - 1);
    int o = 1; // Number of observed variables
    long steps = 1000;
    double eta = sqrt(10);
    int numT = 16;

    /*---INIT---*/
    M x_true(vars, 1); //set definition at the top
    M X_analysis(vars, steps + 1);
    M X_True(vars, steps + 1);
    x_true << 1, 1, 1;
    x_perturbed = x_true;
    x_perturbed += perturbedM(x_true.size(), 1, n01, engine);
    setColumn(X_True, 0, x_perturbed);
    observe_covariance = (eta * eta) * I(o); //precalc
    L = cholesky(observe_covariance); //precalc
    x_background = perturbedM(x_true.size(), N_ens, n01, engine);
    setColumn(X_analysis, 0, mean(x_background));
    x_analysis = x_background;
    H = I(o, x_true.size()); //precalc

    #pragma omp parallel num_threads(numT)
    for (long long i = 1; i <= steps; i++) {


        /*---------1st Section---------*/
        start = omp_get_wtime();
        #pragma omp single
        {
            x_perturbed = ode(x_perturbed, t, t_end, dt);
        }
        #pragma omp barrier
        end = omp_get_wtime();
        s1 += ms_difference(start, end);
        /*-----------------------------*/


        /*---------2nd Section---------*/
        start = omp_get_wtime();
        #pragma omp sections
        {
            #pragma omp section
            {
                setColumn(X_True, i, x_perturbed);
            }

            #pragma omp section
            {
                y_observe = (H * x_perturbed) + (L * perturbedM(o, 1, n01, engine));
                Y = y_observe.replicate(1, N_ens);
                Y += L * M::Random(o, N_ens);
            }
        }
        end = omp_get_wtime();
        s2 += ms_difference(start, end);
        /*-----------------------------*/


        /*---------3rd Section---------*/
        start = omp_get_wtime();
        #pragma omp for schedule(static)
        for (long long j = 0; j < N_ens; j++) {
            setColumn(x_background, j, ode(col(x_analysis, j), t, t_end, dt));
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
            x_background = x_background_mean * ones(1, N_ens) + sqrtNens * x_background_deviation;

            x_observe = H * x_background;
            Z_b = invSqrtNens * (x_observe - mean(x_observe) * ones(1, N_ens));
            K_intermediate = (Z_b * Z_b.transpose() + observe_covariance).colPivHouseholderQr().solve(Y - x_observe);
            x_analysis = x_background + (x_background_deviation * Z_b.transpose()) * K_intermediate;
            X_analysis_mean = mean(x_analysis);
            setColumn(X_analysis, i, X_analysis_mean);
        }
        end = omp_get_wtime();
        s4 += ms_difference(start, end);
        /*-----------------------------*/
    }

    X_error = (X_analysis - X_True);
    double rmse = sqrt(((double) X_error.squaredNorm()) / (double) X_error.size());
    print_result(s1, s2, s3, s4, steps, N_ens, numT, rmse, X_analysis, X_True, X_error);

    return 0;
}
