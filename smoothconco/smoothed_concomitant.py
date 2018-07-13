import numpy as np
from numpy.linalg import norm
from cd_smoothed_concomitant import cd_smoothed_concomitant_fast


def SC_path(X, y, lambdas, beta_init=None, sigma_0=None, eps=1e-4,
            max_iter=5000, f=10):

    """ Compute smoothed concomitant Lasso path with coordinate descent. The objective functions

    P(beta, sigma) = 0.5 * norm(y - X beta, 2)^2 / sigma + sigma / 2 + lambda * norm(beta, 1)
    
    argmin_{beta, sigma >= sigma_0} P(beta, sigma)
    
    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.
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
        The estimated noises sigma at the end of the optimization for each alpha.
    gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.
    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.
    """

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape
    sqrt_n = np.sqrt(n_samples)
    residual = y.copy()
    sigmas = np.zeros(n_lambdas)
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    betas = np.zeros((n_lambdas, n_features))
    if beta_init is None:
        beta_init = np.zeros(n_features, order='F')

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
