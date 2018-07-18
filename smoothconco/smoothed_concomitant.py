import numpy as np
from numpy.linalg import norm
from .cd_smoothed_concomitant import cd_smoothed_concomitant_fast

NO_SCREENING = 0
GAPSAFE = 1
WSTRT_SIGMA_0 = 2
BOUND = 3


def SC_path(X, y, lambdas, beta_init=None, sigma_0=None, eps=1e-4,
            max_iter=5000, f=10, screening=1, warm_start_plus=False):
    """ Compute smoothed concomitant Lasso path with coordinate descent.
    The objective function is
    P(beta, sigma) = 0.5 * norm(y - X beta, 2)^2 / sigma + sigma / 2 +
                     lambda * norm(beta, 1)
    and we solve min_{beta, sigma >= sigma_0} P(beta, sigma)

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape = (n_samples,)
        Target values

    beta_init : array, shape (n_features, ), optional
        The initial values of the coefficients.
    lambdas : ndarray
        List of lambdas where to compute the models.
    f : float, optional
        The screening rule will be execute at each f pass on the data
    eps : float, optional
        Prescribed accuracy on the duality gap.
    Returns
    -------
    betas : array, shape (n_features, n_alphas)
        Coefficients along the path.
    sigmas : array, shape (n_alphas,)
        The estimated noises sigma for each lambda.
    gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.
    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.
    """

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    if screening == WSTRT_SIGMA_0:
        screening = GAPSAFE

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape
    sqrt_n = np.sqrt(n_samples)
    residual = y.copy()
    nrm_y = norm(y, ord=2)
    norm_residual = nrm_y
    norm_X2 = np.sum(X ** 2, axis=0)

    if beta_init is None:
        beta_init = np.zeros(n_features, order='F')

    if sigma_0 is None:
        sigma_0 = (norm_residual / sqrt_n) * 1e-2

    betas = np.zeros((n_lambdas, n_features))
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    sigmas = np.zeros(n_lambdas)
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)

    residual = np.asfortranarray(y - np.dot(X, beta_init))
    XTR = np.asfortranarray(np.dot(X.T, residual))

    if screening == BOUND:
        abs_XTy = np.abs(XTR / (np.sqrt(norm_X2) * nrm_y))
        bound_cte = np.sqrt(1 - abs_XTy ** 2)

    else:
        abs_XTy = None
        bound_cte = None

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
