from __future__ import division
import numpy as np


def SBvG_path(X, y, lambda_grid, solver='SCS'):
    """
    solve:

    min - log(rho) + 1/(2n) ||rho y - x phi||^2_2 + lambda_ ||phi||_1

    for lambda in lambda_grid
    """
    import cvxpy as cvx  # nested import to make it optional

    n_samples, n_featuresures = X.shape

    lambda_ = cvx.Parameter(sign="Positive")
    phi = cvx.Variable(n_featuresures)
    rho = cvx.Variable(1)

    objective = \
        cvx.Minimize(- cvx.log(rho) +
                     cvx.sum_squares(rho * y - X * phi) / (2. * n_samples) +
                     lambda_ * cvx.norm(phi, 1))
    prob = cvx.Problem(objective)
    betas = np.zeros((len(lambda_grid), n_featuresures))
    sigmas = np.zeros(len(lambda_grid))

    for i, l in enumerate(lambda_grid):
        lambda_.value = l
        prob.solve(solver=solver)
        this_sigma = 1. / rho.value
        sigmas[i] = this_sigma
        betas[i] = np.ravel(this_sigma * phi.value)

    return np.array(betas), np.array(sigmas)
