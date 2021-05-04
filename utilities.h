#ifndef ENKF_UTILITIES_H
#define ENKF_UTILITIES_H

#include <iostream>
#include <cmath>
#include <random>
#include "eigen/Eigen/Dense"
#include <omp.h>
#include <cstdio>
#include <fstream>

#endif //ENKF_UTILITIES_H

using namespace Eigen;
using value_t = double;
using state_t = VectorXd;
using M = MatrixXd;

double ms_difference(double, double);

__attribute__((unused)) void mm(double **A, double **BB, double **C, long long I, long long J, long long K);

VectorXd ode(state_t x, double t, double t_end, double dt);

M perturbedM(long long row, long long col, std::normal_distribution<double> &n01, std::default_random_engine &engine);

M I(long long i, long long j);

M I(long long i);

M cholesky(M A);

M ones(long long i, long long j);

M col(M &A, long long j);

__attribute__((unused)) void setCol(M &A, long long j, const M &B);

M mean(M A);

void p(const M &A);

void pf(const M &A, const std::string &filename);

void setColumn(M &A, long long J, M B);

void tt(int x);

void print_result(const double s1, const double s2, const double s3, const double s4, const long steps, const int N_ens,
                  const int numT, const double rmse, const M &A, const M &B, const M &C);
