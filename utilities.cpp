#include "utilities.h"

using namespace Eigen;
using value_t = double;
using state_t = VectorXd;
using M = MatrixXd;

double ms_difference(double a, double b) {
    return b - a;
}

struct Lorenz {
    void operator()(const state_t &x, state_t &xd, const double) {
        static constexpr double sigma = 10.0;
        static constexpr double R = 28.0;
        static constexpr double b = 8.0 / 3.0;

        xd(0) = sigma * (x(1) - x(0));
        xd(1) = R * x(0) - x(1) - x(0) * x(2);
        xd(2) = -b * x(2) + x(0) * x(1);
    }
};

using de_system = Lorenz;

constexpr value_t

cx(long double v) { return static_cast<value_t>(v); }

constexpr value_t

operator "" _v(long double v) { return static_cast<value_t>(v); }

template<typename state_t>
struct RK4T {
    using value_t = typename state_t::value_type;

    template<typename System>
    void operator()(System &&system, state_t &x, value_t &t, const value_t dt) {
        const value_t t0 = t;
        const value_t dt_2 = 0.5_v * dt;
        const value_t dt_6 = cx(1.0 / 6.0) * dt;

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

    state_t xd;

private:
    state_t x0, xd_temp;
};

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

VectorXd ode(state_t x, double t, double t_end, double dt) {
    de_system system;
    RK4T<state_t> integrator;
    while (t < t_end) {
        integrator(system, x, t, dt);
    }
    return x;
}

M perturbedM(long long row, long long col, std::normal_distribution<double> &n01, std::default_random_engine &engine) {
    M A(row, col);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            A(i, j) += n01(engine);
    return A;
}

M I(long long i, long long j) {
    return M::Identity(i, j);
}

M I(long long i) {
    return M::Identity(i, i);
}

M cholesky(M A) {
    LLT<M> lltOfA(A);
    return lltOfA.matrixL();
}

M ones(long long i, long long j) {
    return M::Constant(i, j, 1);
}

M col(M &A, long long j) {
    return A.block(0, j, A.rows(), 1);
}

__attribute__((unused)) void setCol(M &A, long long j, const M &B) {
    A.block(0, j, A.rows(), 1) = B;
}

M mean(M A) {
    return A.rowwise().mean();
}

void p(const M &A) {
    std::cout << A << std::endl;
}

void pf(const M &A, const std::string &filename) {
    std::ofstream out(("./output/" + filename));
    out << A;
    out.close();
}

void setColumn(M &A, long long J, M B) {
    for (long long i = 0; i < A.rows(); i++) {
        A(i, J) = B(i, 0);
    }
}

void tt(int x) {
    printf("%d-", x);
}

void print_result(const double s1, const double s2, const double s3, const double s4, const long steps, const int N_ens,
                  const int numT, const double rmse, const M &A, const M &B, const M &C) {

    printf("Number of threads   = %7d \n", numT);
    printf("Number of steps     = %7ld \n", steps);
    printf("Number of ensembles = %7d \n", N_ens);
    printf("Mean S1: %14.6f \t Total S1: %14.6f \n", s1 / steps, s1);
    printf("Mean S2: %14.6f \t Total S2: %14.6f \n", s2 / steps, s2);
    printf("Mean S3: %14.6f \t Total S3: %14.6f \n", s3 / steps, s3);
    printf("Mean S4: %14.6f \t Total S4: %14.6f \n", s4 / steps, s4);
    printf("Total execution time: %14.6f \n", s1 + s2 + s3 + s4);
    printf("RMSE = %11.8f\n", rmse);

    FILE *f2 = fopen("./output/results.txt", "a");

    fprintf(f2, "\n------------------------------------------------------------\n");
    fprintf(f2, "Number of threads   = %7d \n", numT);
    fprintf(f2, "Number of steps     = %7ld \n", steps);
    fprintf(f2, "Number of ensembles = %7d \n", N_ens);
    fprintf(f2, "Mean S1: %14.6f \t Total S1: %14.6f \n", s1 / steps, s1);
    fprintf(f2, "Mean S2: %14.6f \t Total S2: %14.6f \n", s2 / steps, s2);
    fprintf(f2, "Mean S3: %14.6f \t Total S3: %14.6f \n", s3 / steps, s3);
    fprintf(f2, "Mean S4: %14.6f \t Total S4: %14.6f \n", s4 / steps, s4);
    fprintf(f2, "Total execution time: %14.6f \n", s1 + s2 + s3 + s4);
    fprintf(f2, "RMSE = %11.8f\n", rmse);
    fprintf(f2, "------------------------------------------------------------\n");

    fclose(f2);

    pf(A, "analysis.txt");
    pf(B, "true.txt");
    pf(C, "error.txt");

}

