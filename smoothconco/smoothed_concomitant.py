import numpy as np
from numpy.linalg import norm
from cd_smoothed_concomitant import cd_smoothed_concomitant_fast


def SC_path(X, y, lambdas, sigma_0=None, eps=1e-4, max_iter=5000, f=10):

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)

    n_samples, n_features = X.shape
    sqrt_n = np.sqrt(n_samples)
    betas = np.zeros((n_lambdas, n_features))
    beta_init = np.zeros(n_features, order='F')
    residual = y.copy()
    sigmas = np.zeros(n_lambdas)
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)

    nrm_y = norm(y, ord=2)
    norm_residual = nrm_y
    norm_X2 = np.sum(X ** 2, axis=0)

    if sigma_0 is None:
        sigma_0 = (norm_residual / sqrt_n) * 1e-2

    residual = np.asfortranarray(y - np.dot(X, beta_init))
    XTR = np.asfortranarray(np.dot(X.T, residual))

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    for t in range(n_lambdas):

        sigmas[t], gaps[t], norm_residual, n_iters[t] = \
            cd_smoothed_concomitant_fast(X, y, beta_init, XTR, residual,
                                         norm_residual, n_samples,
                                         n_features, sqrt_n, nrm_y,
                                         norm_X2, lambdas[t], eps,
                                         sigma_0, max_iter, f)

        betas[t, :] = beta_init.copy()

        if gaps[t] > eps * nrm_y:

            print("Warning: did not converge, t = %d, max_iter = %d" %
                  (t, max_iter))
            print("gap = %f" % gaps[t])

    return betas, sigmas, gaps, n_iters
