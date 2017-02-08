import numpy as np
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state


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