import numpy as np
from numpy.linalg import norm
from .cd_smoothed_concomitant import cd_smoothed_concomitant_fast
from sklearn.base import BaseEstimator, RegressorMixin

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


class SCRegressor(BaseEstimator, RegressorMixin):

    """ sklearn compatible estimator for the Smoothed Concomitant Lasso.
    """

    def __init__(self, lambdas=None, eps=1e-4, max_iter=5000, f=10.):

        self.eps = eps
        self.max_iter = max_iter
        self.f = f
        self.lambdas = lambdas
        self.sigma_0 = None
        self.beta_init = None

    def fit(self, X, y):
        """ Fit smooth_conco according to X and y.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data.
        y : ndarray, shape = (n_samples,)
            Target values

        Returns
        -------
        self : regressor
            Returns self.
        """

        n_samples, n_features = X.shape

        if self.sigma_0 is None:
            self.sigma_0 = (np.linalg.norm(y) / np.sqrt(n_samples)) * 1e-2

        if self.lambdas is None:  # default values
            n_lambdas = 30
            sigstar = max(self.sigma_0, np.linalg.norm(y) / np.sqrt(n_samples))
            lambda_max = np.linalg.norm(
                np.dot(X.T, y), ord=np.inf) / (n_samples * sigstar)
            self.lambdas = np.logspace(np.log10(lambda_max / 10.),
                                       np.log10(lambda_max), n_lambdas)[::-1]

        model = SC_path(X, y, self.lambdas, self.beta_init, self.sigma_0,
                        self.eps, self.max_iter, self.f)

        self.betas = model[0]
        self.sigmas = model[1]
        self.gaps = model[2]
        self.n_iters = model[3]

        return self

    def predict(self, X):
        """ Compute a prediction vector based on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Testing data.

        Returns
        -------
        pred : ndarray, shape = (n_samples, n_lambdas)
            prediction of target values for different parameter lambda
        """

        pred = np.dot(X, self.betas.T)
        return pred

    def score(self, X, y):
        """ Compute a prediction error wrt y.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Testing data.
        y : ndarray, shape = (n_samples,)
            Testing target values

        Returns
        -------
        pred_error : ndarray, shape = (n_lambdas,)
            Prediction error wrt target values for different parameter lambda.
        """

        n_lambdas = self.betas.shape[0]
        pred_error = [np.linalg.norm(np.dot(X, self.betas.T)[:, l] - y)
                      for l in range(n_lambdas)]
        return np.array(pred_error)
