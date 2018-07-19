import numpy as np
import matplotlib.pyplot as plt
from smoothconco.smoothed_concomitant import SCRegressor
from SBvG import SBvG_path
from tools import generate_data

# Generate dataset

n_samples = 100
n_features = 100
sigma = 2
sparsity = 0.9
snr = 1
correlation = 0.5
random_state = 42

X, y, true_beta, true_sigma = generate_data(n_samples, n_features, sigma, snr,
                                            sparsity, correlation,
                                            random_state=random_state)

# regularization parameter for sigma estimation
sigma_0 = (np.linalg.norm(y) / np.sqrt(n_samples)) * 1e-2
sigstar = max(sigma_0, np.linalg.norm(y) / np.sqrt(n_samples))
lambda_max = np.linalg.norm(np.dot(X.T, y), ord=np.inf) / (n_samples * sigstar)

# SC
clf = SCRegressor(lambdas=[lambda_max / 1.5], eps=1e-4)
clf.fit(X, y)
betas, sigmas = clf.betas, clf.sigmas

# SBvG
betas_SBvG, sigmas_SBvG = SBvG_path(X, y, [lambda_max / 1.5])


fig, axes = plt.subplots(ncols=3, sharey=True)

axes[0].stem(true_beta)
axes[0].set_title("True signal")
axes[1].stem(betas[0])
axes[1].set_title("SC")
axes[2].stem(betas_SBvG[0])
axes[2].set_title("SBvG")

plt.show()


plt.savefig("test_SC.png", format="png")

# Sigma performance:
print("Sigma estimation:")
print("True sigma")
print(sigma)

print("hatsigma for SC")
print(sigmas[0])

print("hatsigma for SBvG")
print(sigmas_SBvG[0])

# Beta performance
print("Estimation risk")
print("SC")
print(np.linalg.norm(true_beta - betas[0]) / n_features)
print("SBvG")
print(np.linalg.norm(true_beta - betas_SBvG[0]) / n_features)
