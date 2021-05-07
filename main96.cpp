#include "eigen/Eigen/Dense"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <omp.h>
#include <random>

using M = Eigen::MatrixXd;

struct Lorenz96{
    void operator()(const Eigen::VectorXd &x, Eigen::VectorXd &xd, const double){
        static constexpr double F = 8;
        int N = x.size();  // Number of states

        xd(0) = (x(1) - x(N-2))*x(N-1) -x(0) + F;
        xd(1) = (x(2) - x(N-1))*x(0) - x(1) + F;
        xd(N-1) = (x(0) - x(N-3))*x(N-2) - x(N-1) + F;
        for(int i = 2; i < N-1; i++){
            xd(i) = (x(i+1) - x(i-2))*x(i-1) - x(i) + F;
        }
    }
};

constexpr double cx(long double v) { return static_cast<double>(v); }

constexpr double operator "" _v(long double v) { return static_cast<double>(v); }

struct RK4T {
    template<typename System>
    void operator()(System &&system, Eigen::VectorXd &x, double &t, const double dt) {
        const double t0 = t;
        const double dt_2 = 0.5_v * dt;
        const double dt_6 = cx(1.0 / 6.0) * dt;

        const size_t n = x.size();
        if (xd.size() < n) {
            xd.resize(n);
            xd_temp.resize(n);
        }

        x0 = x;
        system(x0, xd, t);
        size_t i{};
        for (; i < n; ++i)
            x(i) = dt_2 * xd(i) + x0(i);
        t += dt_2;

        system(x, xd_temp, t);
        for (i = 0; i < n; ++i) {
            xd(i) += 2 * xd_temp(i);
            x(i) = dt_2 * xd_temp(i) + x0(i);
        }

        system(x, xd_temp, t);
        for (i = 0; i < n; ++i) {
            xd(i) += 2 * xd_temp(i);
            x(i) = dt * xd_temp(i) + x0(i);
        }
        t = t0 + dt;

        system(x, xd_temp, t);
        for (i = 0; i < n; ++i)
            x(i) = dt_6 * (xd(i) + xd_temp(i)) + x0(i);
    }

    Eigen::VectorXd xd;

private:
    Eigen::VectorXd x0, xd_temp;
};

double ms_difference(double a, double b) {
    return b - a;
}

__attribute__((unused)) void mm(double **A, double **BB, double **C, long long I, long long J, long long K) {
    double B[K][J];
    double temp;
    #pragma omp for
    for (int j = 0; j < J; j++)
        for (int k = 0; k < K; k++) {
            temp = BB[j][k];
            B[k][j] = temp;
        }

    double sum;
    #pragma omp barrier
    #pragma omp for collapse(2) //check the collapse performance
    for (long long i = 0; i < I; i++)
        for (long long k = 0; k < K; k++) {
            sum = 0;
            for (long long j = 0; j < J; j++) {
                sum += A[i][j] * B[k][j];
            }
            C[i][k] = sum;
        }
}

Eigen::VectorXd ode(Eigen::VectorXd x, double t, double t_end, double dt) {
    Lorenz96 system;
    RK4T integrator;
    while (t < t_end) {
        integrator(system, x, t, dt);
    }
    return x;
}

Eigen::MatrixXd I(long long i, long long j) {
    return Eigen::MatrixXd::Identity(i, j);
}

Eigen::MatrixXd I(long long i) {
    return Eigen::MatrixXd::Identity(i, i);
}

Eigen::MatrixXd cholesky(Eigen::MatrixXd A) {
    Eigen::LLT<Eigen::MatrixXd> lltOfA(A);
    return lltOfA.matrixL();
}

Eigen::MatrixXd ones(long long i, long long j) {
    return Eigen::MatrixXd::Constant(i, j, 1);
}

Eigen::MatrixXd col(Eigen::MatrixXd &A, long long j) {
    return A.block(0, j, A.rows(), 1);
}

__attribute__((unused)) void setCol(Eigen::MatrixXd &A, long long j, const Eigen::MatrixXd &B) {
    A.block(0, j, A.rows(), 1) = B;
}

Eigen::MatrixXd mean(Eigen::MatrixXd A) {
    return A.rowwise().mean();
}

void p(const Eigen::MatrixXd &A) {
    std::cout << A << std::endl;
}

void pf(const Eigen::MatrixXd &A, const std::string &filename) {
    std::ofstream out(("./output/" + filename));
    out << A;
    out.close();
}

void setColumn(Eigen::MatrixXd &A, long long J, Eigen::MatrixXd B) {
    for (long long i = 0; i < A.rows(); i++) {
        A(i, J) = B(i, 0);
    }
}

void tt(int x) {
    printf("%d-", x);
}

void print_result(const double s, const double s1, const double s2, const double s3, const double s4,
                  const long steps, const int N_ens, const int numT, const double rmse,
                  const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C) {

    printf("Number of threads   = %7d \n", numT);
    printf("Number of steps     = %7ld \n", steps);
    printf("Number of ensembles = %7d \n", N_ens);
    printf("Mean S : %14.8f \t Total S : %14.8f \n", s / steps, s);
    printf("Mean S1: %14.8f \t Total S1: %14.8f \n", s1 / steps, s1);
    printf("Mean S2: %14.8f \t Total S2: %14.8f \n", s2 / steps, s2);
    printf("Mean S3: %14.8f \t Total S3: %14.8f \n", s3 / steps, s3);
    printf("Mean S4: %14.8f \t Total S4: %14.8f \n", s4 / steps, s4);
    printf("Total execution time: %14.8f \n", s + s1 + s2 + s3 + s4);
    printf("RMSE = %11.8f\n", rmse);

    FILE *f2 = fopen("./output/results.txt", "a");

    fprintf(f2, "\n------------------------------------------------------------\n");
    fprintf(f2, "Number of threads   = %7d \n", numT);
    fprintf(f2, "Number of steps     = %7ld \n", steps);
    fprintf(f2, "Number of ensembles = %7d \n", N_ens);
    fprintf(f2, "Mean S : %14.8f \t Total S : %14.8f \n", s / steps, s);
    fprintf(f2, "Mean S1: %14.8f \t Total S1: %14.8f \n", s1 / steps, s1);
    fprintf(f2, "Mean S2: %14.8f \t Total S2: %14.8f \n", s2 / steps, s2);
    fprintf(f2, "Mean S3: %14.8f \t Total S3: %14.8f \n", s3 / steps, s3);
    fprintf(f2, "Mean S4: %14.8f \t Total S4: %14.8f \n", s4 / steps, s4);
    fprintf(f2, "Total execution time: %14.8f \n", s + s1 + s2 + s3 + s4);
    fprintf(f2, "RMSE = %11.8f\n", rmse);
    fprintf(f2, "------------------------------------------------------------\n");

    fclose(f2);

    pf(A, "analysis.txt");
    pf(B, "true.txt");
    pf(C, "error.txt");

}

Eigen::VectorXd vectorize(Eigen::MatrixXd A) {
    return Eigen::Map<Eigen::VectorXd>(A.data(), A.cols() * A.rows());
}

Eigen::MatrixXd perturbedM(long long row, long long col,
                           std::normal_distribution<double> &n01,
                           std::default_random_engine &engine) {
    Eigen::MatrixXd A(row, col);
    for (long long i = 0; i < row; i++)
        for (long long j = 0; j < col; j++)
            A(i, j) = n01(engine);
    return A;
}

int main(int argc, char *argv[]) {
    int mode;
    if (argc > 2)
        mode = atoi(argv[2]);
    double s = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    double start, end;
    start = omp_get_wtime();
    /*---Setup---*/
    std::default_random_engine engine;
    std::normal_distribution<double> n01(0, 1.0);
    M y_observe, X_analysis_mean, x_background_mean, x_background_deviation, X_error, x_observe, observe_covariance,
            x_analysis, L, Z_b, x_background, H, z_b, z_b_mean, Y, K_intermediate, tmp, x_truth, L_model, exact_L;

    /*---Parameters---*/
    int num_var = 40;
    double t = 0.0;
    double t_end = 0.05;
    double dt = 0.001;
    double inflation = 1.01;
    int N_ens = 64; // Number of Ensembles
    double inv_sqrt_ens = 1 / sqrt(N_ens - 1); //inverse square root of number of ensembles
    double sqrt_ens = sqrt(N_ens - 1);
    int o = 40; // Number of observed variables
    long steps = 5000;
    double eta = sqrt(1);
    int numThreads = atoi(argv[1]);

    /*---INITIALIZE VAlUES---*/
    M x_true(num_var, 1); //set definition at the top for vars
    x_true = 8 * M::Ones(num_var, 1);
    x_true(0,0) += n01(engine);
    M X_analysis(x_true.size(), steps + 1);
    M X_True(x_true.size(), steps + 1);
    x_truth = x_true;
    //x_truth += perturbedM(x_true.size(), 1, n01, engine);
    setColumn(X_True, 0, x_truth);
    observe_covariance = (eta * eta) * I(o); //precalc
    L = cholesky(observe_covariance); //precalc

    x_background = x_truth.replicate(1, N_ens) + perturbedM(x_true.size(), N_ens, n01, engine);
    setColumn(X_analysis, 0, mean(x_background));
    x_analysis = x_background;
    H = I(o, x_true.size()); //precalc

    /*----INITIALIZE EIGEN PARALLEL ENGINE
     * SET NUMBER OF THREADS for OMP----*/
    Eigen::initParallel();
    omp_set_nested(1);
    omp_set_num_threads(numThreads);
    Eigen::setNbThreads(numThreads);

    end = omp_get_wtime();
    s += ms_difference(start, end);
    #pragma omp parallel num_threads(numThreads)
    {
        for (long long i = 1; i <= steps; i++) {

            /*---------1st Section
             * Propagation in time for True value---------*/
            start = omp_get_wtime();
            #pragma omp single
            {
                x_truth = ode(vectorize(x_truth), t, t_end, dt);
            }
            end = omp_get_wtime();
            s1 += ms_difference(start, end);
            /*-----------------------------*/


            /*---------2nd Section
             * Updating X_True
             * Creating an Ensemble of Perturbed Observations ---------*/
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


            /*---------3rd Section
             * Ensemble Propagation through State Equation---------*/
            start = omp_get_wtime();
            #pragma omp for
            for (long long j = 0; j < N_ens; j++) {
                setColumn(x_background, j, ode(vectorize(col(x_analysis, j)), t, t_end, dt));
            }
            end = omp_get_wtime();
            s3 += ms_difference(start, end);
            /*-----------------------------*/


            /*---------4th Section
             * Analysis/Assimilation Step---------*/
            start = omp_get_wtime();
            #pragma omp single
            {
                x_background_mean = mean(x_background);
                x_background_deviation = inv_sqrt_ens * (x_background - x_background_mean * ones(1, N_ens)) * inflation;
                x_background = x_background_mean.replicate(1, N_ens) + sqrt_ens * x_background_deviation;
                x_observe = H * x_background;

                Z_b = inv_sqrt_ens * (x_observe - mean(x_observe) * ones(1, N_ens));
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

    X_error = (X_analysis - X_True);

    double rmse = sqrt(((double) X_error.squaredNorm()) / (double) X_error.size());//throw first 500-1000
    if (argc > 2 && mode == 2) {
        printf("\nNumber of Threads: %5d\n",numThreads);
        printf("S: %14.8f\n", s);
        printf("S1: %14.8f \n", s1);
        printf("S2: %14.8f \n", s2);
        printf("S3: %14.8f \n", s3);
        printf("S4: %14.8f \n", s4);
        printf("Total time: %14.8f\n", s + s1 + s2 + s3 + s4);
        printf("RMSE: %14.8f \n", rmse);
    } else {
        print_result(s, s1, s2, s3, s4, steps, N_ens, numThreads, rmse, X_analysis, X_True, X_error);
    }

    return 0;
}
