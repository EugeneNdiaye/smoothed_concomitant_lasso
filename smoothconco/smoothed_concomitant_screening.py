import numpy as np
from numpy.linalg import norm
from cd_smoothed_concomitant_screening import cd_smoothed_concomitant_fast

NO_SCREENING = 0
GAPSAFE = 1
WSTRT_SIGMA_0 = 2
BOUND = 3


def SC_path_screening(X, y, lambdas, sigma_0=None, eps=1e-4, max_iter=5000,
                      f=10, screening=0, warm_start_plus=False):

    if screening == WSTRT_SIGMA_0:
        screening = GAPSAFE

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)

    n_samples, n_features = X.shape
    sqrt_n = np.sqrt(n_samples)
    betas = np.zeros((n_lambdas, n_features))
    beta_init = np.zeros(n_features, order='F')
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    residual = y.copy()
    sigmas = np.zeros(n_lambdas)
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)

    nrm_y = norm(y, ord=2)
    norm_residual = nrm_y
    norm_X2 = np.sum(X ** 2, axis=0)

    if sigma_0 is None:
        sigma_0 = (norm_residual / sqrt_n) * 1e-2

    residual = np.asfortranarray(y - np.dot(X, beta_init))
    XTR = np.asfortranarray(np.dot(X.T, residual))

    abs_XTy = np.abs(XTR / (np.sqrt(norm_X2) * nrm_y))
    bound_cte = np.sqrt(1 - abs_XTy ** 2)

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    for t in range(n_lambdas):

        beta_init, sigmas[t], gaps[t], norm_residual, n_iters[t], \
            n_active_features[t] = \
            cd_smoothed_concomitant_fast(X, y, beta_init, abs_XTy, XTR,
                                         residual,
                                         disabled_features,
                                         norm_residual, sqrt_n, nrm_y,
                                         norm_X2, lambdas[t], eps,
                                         sigma_0, max_iter, f, screening,
                                         bound_cte,
                                         wstr_plus=0)

        betas[t, :] = beta_init.copy()

        if warm_start_plus and t < n_lambdas - 1 and \
           n_active_features[t] < n_features:

            beta_init, _, _, norm_residual, _, _ = \
                cd_smoothed_concomitant_fast(X, y, beta_init, abs_XTy,
                                             XTR, residual, disabled_features,
                                             norm_residual, sqrt_n, nrm_y,
                                             norm_X2, lambdas[t + 1], eps,
                                             sigma_0, max_iter, f,
                                             screening=screening,
                                             bound_cte=bound_cte,
                                             wstr_plus=1)

        if gaps[t] > eps * nrm_y:

            print("Warning: did not converge, t = %d" % t)
            print("gap = %f, eps = %f" % (gaps[t], eps))

    return betas, sigmas, gaps, n_iters, n_active_features
