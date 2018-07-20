# TODO: rename the estimators
import time
import numpy as np
from itertools import product
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import KFold
from tools import (generate_data, get_lambdas, OR, L_RCV, SC_CV, SC_CV_LS,
                   L_CV, SBvG_CV, SQRT_Lasso_CV, L_CV_LS, L_U, L_U_LS, SZ_LS,
                   D2, estimator_LS_CV, SZ_CV)

try:
    import mkl
    mkl.set_num_threads(1)
except Exception as e:
    pass


n_folds = 2  # Number of fold used in each cross validation
scale_sigma_0 = 1e-2
eps = 1e-4
sigma = 1
max_iter = 50
n_simulations = 5
n_lambdas = 10
n_samples = 10

# n_folds = 5  # Number of fold used in each cross validation
# scale_sigma_0 = 1e-2
# eps = 1e-4
# sigma = 1
# max_iter = 5000
# n_simulations = 50
# n_lambdas = 100
# n_samples = 100

# Done
# sparsitys = [0.9]
# correlations = [0.6]
# snrs = [5.]
# n_features_grid = [500]
# s9c6n100p500snr50

# To run
sparsitys = [0.8]
correlations = [0.]
snrs = [0.5]
n_features_grid = [500]
# s8c0n100p500snr5

# sparsitys = [0.9]
# correlations = [0.8]
# snrs = [10.]
# n_features_grid = [500]
# s9c8n100p500snr100

N_JOBS = -1
N_JOBS_ITER = -1

N_JOBS = 1
N_JOBS_ITER = -1


def run_one(n_s, n_samples, n_features, n_lambdas, sigma, snr, sparsity,
            correlation, n_simulations, n_folds, scale_sigma_0, eps, max_iter):

    seed = 42 * n_s
    results = []

    # Generate data set
    # The data are standardized (x_j ~ N(0,1))
    X, y, true_beta, true_sigma = generate_data(n_samples, n_features,
                                                sigma, snr, sparsity,
                                                correlation,
                                                random_state=seed)

    # Regularized parameter for the concomitants
    sigma_0 = np.linalg.norm(y) / np.sqrt(n_samples) * scale_sigma_0

    # The fold used to perform the cross-validations
    KF = KFold(n_samples, n_folds, shuffle=True, random_state=seed)

    print("Oracle")
    tic = time.time()
    sigma_OR, sigma_LS_OR = OR(X, y, true_beta)
    toc = time.time() - tic
    results.append(['OR', sigma_LS_OR, toc])
    # results.append(['LS_OR', sigma_LS_OR, toc])

    print("Lasso")
    print("Getting lambdas for lassos")
    lambdas = get_lambdas(X, y, sigma_0=sigma_0, n_lambdas=n_lambdas, delta=2,
                          method="lasso")
    tic = time.time()
    beta_lasso, sigma_L = L_CV(X, y, lambdas, sigma_0, eps, KF, max_iter)
    toc = time.time() - tic
    results.append(['L_CV', sigma_L, toc])
    sigma_LS_L = L_CV_LS(X, y, beta_lasso)[1]
    toc = time.time() - tic
    results.append(['L_CV_LS', sigma_LS_L, toc])

    tic = time.time()
    beta_ulasso, sigma_U_L = L_U(X, y, eps=eps, max_iter=max_iter)
    toc = time.time() - tic
    results.append(['L_U', sigma_U_L, toc])
    sigma_LS_U_L = L_U_LS(X, y, beta_ulasso)[1]
    toc = time.time() - tic
    results.append(['L_U_LS', sigma_LS_U_L, toc])

    print("RCV Lasso")
    tic = time.time()
    sigma_RCV_L = L_RCV(X, y, lambdas, eps, max_iter)[1]
    toc = time.time() - tic
    results.append(['L_RCV', sigma_RCV_L, toc])

    print("cv_ls estimators")
    tic = time.time()
    sigma_cv_ls_lasso = \
        estimator_LS_CV("ls_lasso", X, y, lambdas, sigma_0=sigma_0, eps=eps,
                        max_iter=max_iter)[1]

    toc = time.time() - tic
    results.append(['L_LS_CV', sigma_cv_ls_lasso, toc])

    print("Square-Root Lasso")
    print("Getting lambdas for concomitants")
    lambdas = get_lambdas(X, y, sigma_0=sigma_0, n_lambdas=n_lambdas, delta=2,
                          method="no_lasso")
    tic = time.time()
    beta_sc, sigma_SC = SC_CV(X, y, lambdas, sigma_0, eps, KF, max_iter)
    toc = time.time() - tic
    results.append(['SC_CV', sigma_SC, toc])
    sigma_SC_LS = SC_CV_LS(X, y, beta_sc)
    toc = time.time() - tic
    results.append(['SC_CV_LS', sigma_SC_LS, toc])

    print("Scaled lasso")
    tic = time.time()
    sigma_SZ, sigma_SZ_LS = SZ_LS(X, y, eps, max_iter)
    toc = time.time() - tic
    results.append(['SZ', sigma_SZ, toc])
    results.append(['SZ_LS', sigma_SZ_LS, toc])

    print("Moment estimators")
    tic = time.time()
    sigma_D1, sigma_D2 = D2(X, y)
    toc = time.time() - tic
    # results.append(['D1', sigma_D1, toc])
    results.append(['D2', sigma_D2, toc])

    tic = time.time()
    sigma_SC_LS_CV = \
        estimator_LS_CV("ls_smoothed_concomitant", X, y, lambdas,
                        sigma_0=sigma_0, eps=eps, max_iter=max_iter, KF=KF)[1]
    toc = time.time() - tic
    results.append(['SC_LS_CV', sigma_SC_LS_CV, toc])

    tic = time.time()
    sigma_sz_cv = SZ_CV(X, y, lambdas, sigma_0, eps, KF)[1]
    toc = time.time() - tic
    results.append(['SZ_CV', sigma_sz_cv, toc])

    print("Staedler estimator")
    tic = time.time()
    sigma_SBvG = SBvG_CV(X, y, lambdas)[1]
    toc = time.time() - tic
    results.append(['SBvG_CV', sigma_SBvG, toc])

    print("belloni estimator")
    tic = time.time()
    sigma_sqrt_lasso = SQRT_Lasso_CV(X, y, lambdas)[1]
    toc = time.time() - tic
    results.append(['SQRT-Lasso_CV', sigma_sqrt_lasso, toc])

    return results


def estim_sigma(n_samples=100, n_features=200, n_lambdas=30,
                sigma=1, snr=2, sparsity=0.80, correlation=0.5,
                n_simulations=30, n_folds=5, sigma_0=1e-2, eps=1e-4,
                max_iter=5000):

    results = []
    params = [n_samples, n_features, n_lambdas,
              sigma, snr, sparsity, correlation,
              n_simulations, n_folds, sigma_0, eps,
              max_iter]

    parallel = Parallel(n_jobs=N_JOBS_ITER, verbose=2)
    results = parallel(delayed(run_one)(n_s, *params)
                       for n_s in range(n_simulations))
    results = sum(results, [])

    estimators = pd.DataFrame(results, columns=['method', 'sigma', 'time'])
    return estimators


def run_estim_sigma(n_features, snr, sparsity, corr):
    print("p = ", n_features, " - sparsity = ", sparsity,
          " - correlation = ", corr, " - snr = ", snr)

    estimators = estim_sigma(n_samples, n_features, n_lambdas, sigma,
                             snr, sparsity, corr, n_simulations, n_folds,
                             scale_sigma_0, eps, max_iter)

    name = ("s" + str(int(10 * sparsity)) +
            "c" + str(int(10 * corr)) +
            "n" + str(n_samples) +
            "p" + str(n_features) +
            "snr" + str(int(10 * snr)))

    estimators.to_csv('results/' + name + '.csv', index=False)


Parallel(n_jobs=N_JOBS, verbose=2)(delayed(run_estim_sigma)(*params)
                                   for params in product(n_features_grid,
                                                         snrs, sparsitys,
                                                         correlations))
