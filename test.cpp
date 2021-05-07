int main(int argc, char *argv[]) {
    M x_true(3, 1);
    x_true << -5.9, -5.6, 24.4;
    M X_analysis(x_true.size(), steps + 1);
    M X_True(x_true.size(), steps + 1);
    x_truth = x_true;
    x_truth += perturbedM(x_true.size(), 1, n01, engine);
    setColumn(X_True, 0, x_truth);
    observe_covariance = (eta * eta) * I(o); /
    L = cholesky(observe_covariance);

    x_background = x_truth.replicate(1, N_ens) + perturbedM(x_true.size(), N_ens, n01, engine);
    setColumn(X_analysis, 0, mean(x_background));
    x_analysis = x_background;
    H = I(o, x_true.size());

    end = omp_get_wtime();
    s += ms_difference(start, end);
    #pragma omp parallel num_threads(numThreads)
    {
        for (long long i = 1; i <= steps; i++) {

            start = omp_get_wtime();
            /*1st Section*/
            #pragma omp single
            {
                x_truth = ode(vectorize(x_truth), t, t_end, dt);
            }
            /*2nd Section*/
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

            /*3rd Section*/
            #pragma omp for
            for (long long j = 0; j < N_ens; j++) {
                setColumn(x_background, j, ode(vectorize(col(x_analysis, j)), t, t_end, dt));
            }

            /*4th Section*/
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
        }
    }
    return 0;
}