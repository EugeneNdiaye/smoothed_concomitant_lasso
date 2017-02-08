
from smoothed_concomitant import smoothed_concomitant_path  # Cython version
from sklearn.linear_model import lasso_path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.linear_model import LassoCV
import scipy.stats
from SBvG import SBvG_path
from sqrt_lasso_cvxpy import belloni_path


def ST(x, tau):

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
            betas, sigmas, gaps, n_iters = \
                smoothed_concomitant_path(X_train, y_train, lambdas, eps=eps,
                                          sigma_0=sigma_0)
        elif method == "lasso":
            betas = lasso_path(X_train, y_train, alphas=lambdas, tol=eps)[1]
            betas = betas.T
        elif method in ["ls_smoothed_concomitant", "ls_lasso"]:
            betas = ls_lassolike(X_train, y_train, lambdas, sigma_0=sigma_0,
                                 eps=eps, method=method)
        elif method == "grid_concomitant":
            betas, sigmas = grid_concomitant(X_train, y_train, eps, lambdas)
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


def SC_CV(X, y, lambdas, sigma_0, eps, KF, max_iter):
    cv_errors = cross_val(X, y, lambdas, sigma_0, eps=eps,
                          method="smoothed_concomitant", KF=KF)

    lsqrt = lambdas[np.argmin(cv_errors)]
    beta, sigma, _, _ = \
        smoothed_concomitant_path(X, y, [lsqrt], sigma_0=sigma_0, eps=eps,
                                  max_iter=max_iter)

    return beta.ravel(), sigma[0]


def SC_CV_LS(X, y, beta_sqrt):
    n_samples = X.shape[0]
    # Least square on the support of best smoothed_concomitant
    refit_beta, norm_residual, size = refit(X, y, beta_sqrt)
    sigma = norm_residual / np.sqrt(n_samples - size)

    return refit_beta, sigma


def L_CV(X, y, lambdas, sigma_0, eps=1e-4, KF=None, max_iter=5000):

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
    n_samples = X.shape[0]
    # Least square on the support of best lasso
    beta, norm_residual, size = refit(X, y, beta_lasso)
    sigma = norm_residual / np.sqrt(n_samples - size)

    return beta, sigma


def L_U(X, y, eps=1e-4, max_iter=5000):
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
    n_samples, n_features = X.shape
    # Least square on the support of univ_lasso
    beta, norm_residual, size = refit(X, y, beta_ulasso)
    sigma = norm_residual / np.sqrt(n_samples - size)

    return beta, sigma


def SC_RCV(X, y, lambdas, sigma_0, eps, max_iter):

    n_samples = X.shape[0]
    KF = KFold(n_samples, 2, shuffle=True, random_state=42)
    sigma2 = 0.

    for idx1, idx2 in KF:

        cv_errors = cross_val(X, y, lambdas, sigma_0, eps=1e-4,
                              method="smoothed_concomitant")
        lsqrt = lambdas[np.argmin(cv_errors)]
        beta, sigma_sqrt, _, _ = \
            smoothed_concomitant_path(X[idx1], y[idx1], [lsqrt],
                                      sigma_0=sigma_0, eps=eps,
                                      max_iter=max_iter)
        refit_beta, norm_residual, size = refit(X[idx2], y[idx2], beta.ravel())
        sigma2 += (norm_residual ** 2) / (n_samples / 2. - size)
        sigma = np.sqrt(sigma2 / 2.)
    return beta, sigma


def L_RCV(X, y, lambdas, eps, max_iter):

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


def grid_concomitant(X, y, eps, lambdas, max_iter=100):

    n_samples, n_features = X.shape
    n_lambdas = len(lambdas)
    sigmas = np.zeros(n_lambdas)
    betas = np.zeros((n_lambdas, n_features))

    for l in range(n_lambdas):
        betas[l, :], sigmas[l] = oneSZ(X, y, eps, lambdas[l], max_iter)

    return betas, sigmas


def cv_grid_concomitant(X, y, lambdas, sigma_0, eps, KF):

    cv_errors = cross_val(X, y, lambdas, sigma_0, eps=eps,
                          method="grid_concomitant", KF=KF)
    best_l = lambdas[np.argmin(cv_errors)]
    beta, sigma = oneSZ(X, y, eps, best_l)[0]

    return beta, sigma


def SBvG_CV(X, y, lambdas, KF=None):
    n_samples, n_features = X.shape
    sigma_0 = np.nan
    cv_errors = cross_val(X, y, lambdas, sigma_0, method="SBvG", KF=KF)

    l_ = lambdas[np.argmin(cv_errors)]
    beta_opt, sigma_opt = SBvG_path(X, y, [l_])
    beta_opt = beta_opt.ravel()

    return beta_opt, sigma_opt[0]


def SQRT_Lass_CV(X, y, lambdas, KF=None):
    n_samples, n_features = X.shape
    sigma_0 = np.nan
    cv_errors = cross_val(X, y, lambdas, sigma_0, method="belloni", KF=KF)

    l_ = lambdas[np.argmin(cv_errors)]
    beta_opt, sigma_opt = belloni_path(X, y, [l_])
    beta_opt = beta_opt.ravel()

    return beta_opt, sigma_opt[0]


def SC_LS(X, y, eps, max_iter):

    n_samples = X.shape[0]
    beta, _ = SZ(X, y, eps, max_iter)

    # Least square on the support of the scaled lasso
    refit_beta, norm_residual, size = refit(X, y, beta)
    sigma_Ls = norm_residual / np.sqrt(n_samples - size)

    return refit_beta, sigma_Ls


def D2(X, y):
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


# ??????
def ls_lassolike(X, y, lambdas, sigma_0=1e-2, eps=1e-4, max_iter=5000,
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
        betas = smoothed_concomitant_path(X, y, lambdas, sigma_0=sigma_0,
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


def sigmas_cv_ls_lassolike(method, X, y, lambdas, sigma_0, eps=1e-4,
                           max_iter=5000, KF=None):

    # TODO do better
    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_samples = X.shape[0]
    cv_errors = cross_val(X, y, lambdas, sigma_0, eps=eps, method=method)
    best_lambda = lambdas[np.argmin(cv_errors)]
    beta = ls_lassolike(X, y, best_lambda, sigma_0=sigma_0, eps=eps,
                        max_iter=max_iter, method=method).ravel()
    size = len(np.where(beta != 0)[0])
    sigma = np.linalg.norm(y - np.dot(X, beta)) / np.sqrt(n_samples - size)

    return beta, sigma
