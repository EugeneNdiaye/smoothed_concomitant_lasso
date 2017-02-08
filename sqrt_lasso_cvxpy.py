from __future__ import division
import numpy as np
import cvxpy as cvx


def belloni_path(X, y, lambda_grid, solver='ECOS'):
    '''
    solve:
     min ||y - X*beta|| / sqrt(n_samples) + lambda ||beta||_1
    '''

    n_samples, n_features = X.shape
    lambda_ = cvx.Parameter(sign="Positive")
    beta = cvx.Variable(n_features)

    objective = cvx.Minimize(cvx.norm(X * beta - y, 2) / np.sqrt(n_samples) +
                             lambda_ * cvx.norm(beta, 1))
    prob = cvx.Problem(objective)

    betas = np.zeros((len(lambda_grid), n_features))
    sigmas = np.zeros(len(lambda_grid))

    for i, l in enumerate(lambda_grid):
        lambda_.value = l
        prob.solve(solver=solver)
        betas[i] = np.ravel(beta.value)
        sigmas[i] = \
            np.linalg.norm(y - np.dot(X, betas[i])) / np.sqrt(n_samples)

    return sigmas, betas
