
from smoothconco.smoothed_concomitant import SC_path
from sklearn.linear_model import lasso_path
import numpy as np
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.linear_model import LassoCV
import scipy.stats
from smoothconco.SBvG import SBvG_path
from smoothconco.sqrt_lasso_cvxpy import belloni_path

"""
Tools used in our numerical experiments in https://arxiv.org/abs/1606.02702
"""


def generate_data(n_samples, n_features, sigma=1., snr=2., sparsity=0.8,
                  corr=0.5, random_state=42):

    rng = check_random_state(random_state)
    vect = corr ** np.arange(n_features)
    covar = toeplitz(vect, vect)
    X = rng.multivariate_normal(np.zeros(n_features), covar, n_samples)
    beta = rng.laplace(size=n_features)
    n_zeros = int(n_features * sparsity)
    mask = rng.choice(range(n_features), n_zeros, replace=False)
    beta[mask] = np.zeros(n_zeros)

    # scale to precribed snr
    # scale = np.sqrt(np.dot(beta.T, np.dot(covar, beta)) / (snr * sigma ** 2))
    scale = np.sqrt(sigma ** 2 * snr / np.dot(beta.T, np.dot(covar, beta)))
    beta *= scale

    y = np.dot(X, beta) + sigma * rng.normal(0, 1, n_samples)

    return X, y, beta, sigma


def ST(x, tau):
    """
        Vectorial soft-thresholding at level u.
    """

    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


def bp_cp(X, y, sigma=None, tau=None, MAX_ITER=10000):
    """
    Solve Basis Pursuit with Chambolle-Pock algorithm
    """

    if sigma is None and tau is None:
        normX = np.linalg.norm(X, ord=2)
        sigma = 1. / float(normX)
        tau = 0.99 / float(normX)

    n_samples, n_features = X.shape
    theta = np.zeros(n_samples)
    beta = np.zeros(n_features)

    for k in range(MAX_ITER):
        theta_old = theta
        theta += -sigma * np.dot(X, beta) + sigma * y
        beta = ST(beta + tau * np.dot(X.T, 2 * theta - theta_old), tau)

    return beta, theta


def get_lambdas(X, y, sigma_0, n_lambdas=100, delta=2, method="lasso"):
    """ Compute a list of regularization parameters for Lasso and sconco which 
        decrease geometrically.
    """

    n_samples = X.shape[0]
    # Compute lambda_max

    if method == "lasso":
        lambda_max = np.linalg.norm(np.dot(X.T, y), ord=np.inf) / n_samples

    else:
        tmp = max(sigma_0, np.linalg.norm(y) / np.sqrt(n_samples))
        lambda_max = \
            np.linalg.norm(np.dot(X.T, y), ord=np.inf) / (n_samples * tmp)

    # Generate grid
    lambdas = np.logspace(np.log10(lambda_max / float(10 ** delta)),
                          np.log10(lambda_max), n_lambdas)[::-1]

    return lambdas


def refit(X, y, beta):
    """
        Compute a least squares solution by restricting the features onto the 
        support of beta
    """

    lin_reg = LinearRegression()
    support = np.where(beta != 0)[0]
    X_support = X[:, support]
    size = len(support)

    if size != 0:
        res = lin_reg.fit(X_support, y)
        refit_beta = res.coef_
        norm_residual = np.linalg.norm(y - lin_reg.predict(X_support))
        return refit_beta, norm_residual, size

    return beta, np.linalg.norm(y), size


def cross_val(X, y, lambdas, sigma_0, eps=1e-4, method="lasso", KF=None):
    """
        Perform a 5-fold cross-validation and return the mean square errors for
        different parameters lambdas. 
    """

    n_samples, n_features = X.shape
    n_lambdas = len(lambdas)
    if KF is None:
        KF = KFold(n_samples, 5, shuffle=True, random_state=42)
    n_folds = KF.n_folds
    errors = np.zeros((n_lambdas, n_folds))
    i_fold = 0

    for train_index, test_index in KF:

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        if method == "smoothed_concomitant":
            betas, sigmas, gaps, n_iters, _ = \
                SC_path(X_train, y_train, lambdas, eps=eps,
                        sigma_0=sigma_0)

        elif method == "lasso":
            betas = lasso_path(X_train, y_train, alphas=lambdas, tol=eps)[1]
            betas = betas.T

        elif method in ["ls_smoothed_concomitant", "ls_lasso"]:
            betas = estimator_LS(X_train, y_train, lambdas, sigma_0=sigma_0,
                                 eps=eps, method=method)

        elif method == "SZ_path":
            betas, sigmas = SZ_path(X_train, y_train, eps, lambdas)

        elif method == "SBvG":
            betas, sigmas = SBvG_path(X_train, y_train, lambdas)

        elif method == "belloni":
            betas, sigmas = belloni_path(X_train, y_train, lambdas)

        else:
            1 / 0  # BOOM !

        for l in range(n_lambdas):
            y_pred = np.dot(X_test, betas[l, :])
            errors[l, i_fold] = np.mean((y_pred - y_test) ** 2)

        i_fold += 1

    return np.mean(errors, axis=1)


def OR(X, y, true_beta):
    """
    Compute oracle estimators of the noise level
    """

    n_samples = X.shape[0]
    sgm2_OR = np.linalg.norm(y - np.dot(X, true_beta)) ** 2 / float(n_samples)

    # Least square on the support of the oracle
    refit_beta, norm_residual, size = refit(X, y, true_beta)
    sigma2_Ls_oracle = (norm_residual ** 2) / float(n_samples - size)

    return np.sqrt(sgm2_OR), np.sqrt(sigma2_Ls_oracle)


def SC_CV(X, y, lambdas, sigma_0, eps, KF, max_iter):
    """
    Compute sconco estimator at the best regularizer lambda selected by 
    cross-validation.
    """

    cv_errors = cross_val(X, y, lambdas, sigma_0, eps=eps,
                          method="smoothed_concomitant", KF=KF)

    lsqrt = lambdas[np.argmin(cv_errors)]
    beta, sigma, _, _, _ = \
        SC_path(X, y, [lsqrt], sigma_0=sigma_0, eps=eps, max_iter=max_iter)

    return beta.ravel(), sigma[0]


def SC_CV_LS(X, y, beta):
    """
    Compute a refitted estimator from sconco
    """

    n_samples = X.shape[0]
    # Least square on the support of best smoothed_concomitant
    refit_beta, norm_residual, size = refit(X, y, beta)
    sigma = norm_residual / np.sqrt(n_samples - size)

    return refit_beta, sigma


def L_CV(X, y, lambdas, sigma_0, eps=1e-4, KF=None, max_iter=5000):
    """
    Compute Lasso estimator at the best regularizer lambda selected by 
    cross-validation.
    """

    n_samples, n_features = X.shape
    cv_errors = cross_val(X, y, lambdas, sigma_0, eps=eps, method="lasso",
                          KF=KF)

    l_lasso = lambdas[np.argmin(cv_errors)]
    beta_lasso = lasso_path(X, y, alphas=[l_lasso], tol=eps,
                            max_iter=max_iter)[1]
    beta = beta_lasso.ravel()
    size = np.sum(beta != 0)
    sigma = np.linalg.norm(y - np.dot(X, beta))
    sigma /= np.sqrt(n_samples - size)

    return beta, sigma


def L_CV_LS(X, y, beta_lasso):
    """
    Compute a refitted estimator from Lasso
    """
    ### !!! no cv here, it is directly perfomed is benchmark.
    # TODO: merge CV_LS procedure
    n_samples = X.shape[0]
    # Least square on the support of best lasso
    beta, norm_residual, size = refit(X, y, beta_lasso)
    sigma = norm_residual / np.sqrt(n_samples - size)

    return beta, sigma


def L_U(X, y, eps=1e-4, max_iter=5000):
    """
    Compute Lasso estimator at a universal regularizer lambda recommended by 
    the theory.
    """

    n_samples, n_features = X.shape
    # Lasso with universal lambda
    univ_lambda = np.sqrt(2. * np.log(n_features) / float(n_samples))
    beta_ulasso = lasso_path(X, y, alphas=[univ_lambda], tol=eps,
                             max_iter=max_iter)[1]
    beta_ulasso = beta_ulasso.ravel()
    size = np.sum(beta_ulasso != 0)
    sigma_ulasso = np.linalg.norm(y - np.dot(X, beta_ulasso))
    sigma_ulasso /= np.sqrt(n_samples - size)
    return beta_ulasso, sigma_ulasso


def L_U_LS(X, y, beta_ulasso):
    """
    Compute a refitted estimator from Lasso with a universal regularizer lambda
    """

    n_samples, n_features = X.shape
    # Least square on the support of univ_lasso
    beta, norm_residual, size = refit(X, y, beta_ulasso)
    sigma = norm_residual / np.sqrt(n_samples - size)

    return beta, sigma


def SC_RCV(X, y, lambdas, sigma_0, eps, max_iter):
    """
    Compute sconco estimator by refitting cross-validation procedure 
    https://arxiv.org/abs/1004.5178
    """

    n_samples = X.shape[0]
    KF = KFold(n_samples, 2, shuffle=True, random_state=42)
    sigma2 = 0.

    for idx1, idx2 in KF:

        cv_errors = cross_val(X, y, lambdas, sigma_0, eps=1e-4,
                              method="smoothed_concomitant")
        lsqrt = lambdas[np.argmin(cv_errors)]
        beta, sigma_sqrt, _, _, _= \
            SC_path(X[idx1], y[idx1], [lsqrt], sigma_0=sigma_0, eps=eps,
                    max_iter=max_iter)
        refit_beta, norm_residual, size = refit(X[idx2], y[idx2], beta.ravel())
        sigma2 += (norm_residual ** 2) / (n_samples / 2. - size)
        sigma = np.sqrt(sigma2 / 2.)

    return beta, sigma


def L_RCV(X, y, lambdas, eps, max_iter):
    """
    Compute Lasso estimator by refitting cross-validation procedure 
    https://arxiv.org/abs/1004.5178
    """

    n_samples = X.shape[0]
    KF = KFold(n_samples, 2, shuffle=True, random_state=42)
    sigma2 = 0

    for idx1, idx2 in KF:

        model = LassoCV(alphas=lambdas, tol=eps, max_iter=max_iter,
                        cv=5).fit(X[idx1], y[idx1])
        beta, norm_residual, size = refit(X[idx2], y[idx2], model.coef_)
        sigma2 += (norm_residual ** 2) / (n_samples / 2. - size)
        sigma = np.sqrt(sigma2 / 2.)
    return beta, sigma


def SZ(X, y, eps, max_iter=100):
    """
    Compute a Scaled-Lasso estimator following the selection of lambda 
    described in https://arxiv.org/abs/1104.4595
    """

    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    L = 0.1
    Lold = 0

    while abs(L - Lold) > 0.001:

        k = L ** 4 + 2 * L ** 2
        Lold = L
        L = - scipy.stats.norm.ppf(min(k / float(n_features), 0.99))
        L = (L + Lold) / 2.

        if n_features == 1:
            L = 0.5

    lambda_0 = np.sqrt(2 / float(n_samples)) * L
    beta, sigma = oneSZ(X, y, eps, lambda_0, max_iter)

    return beta, sigma


def oneSZ(X, y, eps, lambda_0, max_iter=100):
    """
    Compute a Scaled-Lasso estimator by alternating a Lasso path and adjustment 
    of the noise level as in https://arxiv.org/abs/1104.4595
    """

    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    sigma_old = 5
    sigma_new = 0.1
    stop = 0

    while abs(sigma_old - sigma_new) > 0.0001 and stop < max_iter:

        stop += 1
        sigma_old = sigma_new
        lambda_ = sigma_new * lambda_0

        beta = lasso_path(X, y, alphas=[lambda_], tol=eps)[1]
        beta = beta.ravel()
        sigma_new = np.linalg.norm(y - np.dot(X, beta)) / np.sqrt(n_samples)

    return beta, sigma_new


def SZ_path(X, y, eps, lambdas, max_iter=100):
    """
    Compute a Scaled-Lasso estimator following over a grid of parameters lambda
    """

    n_samples, n_features = X.shape
    n_lambdas = len(lambdas)
    sigmas = np.zeros(n_lambdas)
    betas = np.zeros((n_lambdas, n_features))

    for l in range(n_lambdas):
        betas[l, :], sigmas[l] = oneSZ(X, y, eps, lambdas[l], max_iter)

    return betas, sigmas


def SZ_CV(X, y, lambdas, sigma_0, eps, KF):
    """
    Compute a Scaled-Lasso estimator with parameter lambda selected by 
    cross validation
    """

    cv_errors = cross_val(X, y, lambdas, sigma_0, eps=eps,
                          method="SZ_path", KF=KF)
    best_l = lambdas[np.argmin(cv_errors)]
    beta, sigma = oneSZ(X, y, eps, best_l)

    return beta, sigma


def SBvG_CV(X, y, lambdas, KF=None):
    """
    Compute a concomitant estimator following a re-parameterization of the 
    linear model described in https://arxiv.org/abs/1202.6046
    """

    n_samples, n_features = X.shape
    sigma_0 = np.nan
    cv_errors = cross_val(X, y, lambdas, sigma_0, method="SBvG", KF=KF)

    l_ = lambdas[np.argmin(cv_errors)]
    beta_opt, sigma_opt = SBvG_path(X, y, [l_])
    beta_opt = beta_opt.ravel()

    return beta_opt, sigma_opt[0]


def SQRT_Lasso_CV(X, y, lambdas, KF=None):
    """
    Compute a Square-Root Lasso estimator https://arxiv.org/abs/1009.5689 with 
    cvx solver for different regularizer and return the best selected by 
    cross-validation.
    """

    n_samples, n_features = X.shape
    sigma_0 = np.nan
    cv_errors = cross_val(X, y, lambdas, sigma_0, method="belloni", KF=KF)

    l_ = lambdas[np.argmin(cv_errors)]
    beta_opt, sigma_opt = belloni_path(X, y, [l_])

    return beta_opt.ravel(), sigma_opt[0]


def SZ_LS(X, y, eps, max_iter):
    """
    Compute a refitted estimator from Scaled-Lasso
    """

    n_samples = X.shape[0]
    beta, sigma = SZ(X, y, eps, max_iter)

    # Least square on the support of the scaled lasso
    refit_beta, norm_residual, size = refit(X, y, beta)
    sigma_Ls = norm_residual / np.sqrt(n_samples - size)

    # return refit_beta, sigma_Ls
    return sigma, sigma_Ls


def D2(X, y):
    """
    Compute the variance estimators proposed by (Lee H. Dicker, 2014)
    """

    n_samples, n_features = X.shape
    nX = float(n_samples)
    pX = float(n_features)

    norm_y = np.linalg.norm(y)
    norm_Xy = np.linalg.norm(np.dot(X.T, y))

    # With gram
    gramX = np.dot(X.T, X) / nX
    m1 = np.trace(gramX) / pX
    m2 = np.trace(np.dot(gramX, gramX)) / pX - (pX / nX) * (m1 ** 2)

    c1 = (pX + nX + 1.) / (nX * (nX + 1.))
    c2 = 1. / (nX * (nX + 1.))
    sigma2_id = c1 * norm_y ** 2 - c2 * norm_Xy ** 2

    c3 = (1 + (pX * m1 ** 2) / ((nX + 1) * m2)) / nX
    c4 = m1 / ((nX + 1) * nX * m2)
    sigma2_no_id = c3 * norm_y ** 2 - c4 * norm_Xy ** 2

    return np.sqrt(sigma2_id), np.sqrt(sigma2_no_id)


def estimator_LS(X, y, lambdas, sigma_0=1e-2, eps=1e-4, max_iter=5000,
                 method="ls_lasso"):

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape
    refit_betas = np.zeros((n_lambdas, n_features))

    if method == "ls_lasso":
        betas = lasso_path(X, y, alphas=lambdas, tol=eps, max_iter=max_iter)[1]
        betas = betas.T
    elif method == "ls_smoothed_concomitant":
        betas = SC_path(X, y, lambdas, sigma_0=sigma_0,
                        eps=eps, max_iter=max_iter)[0]
    else:
        1 / 0  # BOOM !!!

    for l in range(n_lambdas):
        refit_beta, _, size = refit(X, y, betas[l, :])
        if size != 0:
            # TODO avoid double computation of support
            support = np.where(betas[l, :] != 0)[0]
            refit_betas[l, support] = refit_beta

    return refit_betas


def estimator_LS_CV(method, X, y, lambdas, sigma_0, eps=1e-4, max_iter=5000,
                    KF=None):

    # TODO do better
    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_samples = X.shape[0]
    cv_errors = cross_val(X, y, lambdas, sigma_0, eps=eps, method=method)
    best_lambda = lambdas[np.argmin(cv_errors)]
    beta = estimator_LS(X, y, best_lambda, sigma_0=sigma_0, eps=eps,
                        max_iter=max_iter, method=method).ravel()
    size = len(np.where(beta != 0)[0])
    sigma = np.linalg.norm(y - np.dot(X, beta)) / np.sqrt(n_samples - size)

    return beta, sigma
