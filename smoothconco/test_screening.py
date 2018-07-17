from itertools import cycle, islice
import numpy as np
from data_generation import generate_data
from smoothed_concomitant import SC_path
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.mldata import fetch_mldata

try:
    import mkl
    mkl.set_num_threads(1)
except Exception as e:
    pass


# Synthetic or leukemia dataset
dataset = "leukemia"

if dataset == "synthetic":
    # Generate data set
    n_samples = 100
    n_features = 200
    sigma = 1.
    sparsity = 0.9
    corr = 0.5
    random_state = np.random.randint(0, 100)

    X, y, true_beta, true_sigma = generate_data(n_samples, n_features, sigma,
                                                sparsity, corr,
                                                random_state=random_state)

if dataset == "leukemia":

    data = fetch_mldata('leukemia')
    X = data.data
    y = data.target
    X = X.astype(float)
    y = y.astype(float)
    n_samples, n_features = X.shape

NO_SCREENING = 0
GAPSAFE = 1
WSTRT_SIGMA_0 = 2
BOUND = 3
BOUND2 = 4


# Number of elements in the path (set to 100 for papers results)
n_lambdas = 10

# regularization parameter for sigma estimation
sigma_0 = (np.linalg.norm(y) / np.sqrt(n_samples)) * 1e-2
sigstar = max(sigma_0, np.linalg.norm(y) / np.sqrt(n_samples))
lambda_max = np.linalg.norm(np.dot(X.T, y), ord=np.inf) / (n_samples * sigstar)
lambdas = np.logspace(np.log10(lambda_max / 10.), np.log10(lambda_max),
                      n_lambdas)[::-1]


screenings = [NO_SCREENING, BOUND, GAPSAFE, WSTRT_SIGMA_0]
screenings_names = ["No Screening", "Bound", "Gap Safe", "Gap Safe ++"]


eps_ = range(4, 12, 2)
times = np.zeros((len(screenings), len(eps_)))

for ieps, eps in enumerate(eps_):
    for iscreening, screening in enumerate(screenings):

        if screening == WSTRT_SIGMA_0:
            screening = GAPSAFE
            wstp = True
        else:
            wstp = False

        begin = time.time()
        betas, sigmas, gaps, n_iters, screening_sizes = \
            SC_path(X, y, lambdas, sigma_0=sigma_0, eps=10**(-eps),
                    max_iter=5000, f=10, screening=screening,
                    warm_start_plus=wstp)
        duration = time.time() - begin

        times[iscreening, ieps] = duration
        print(screenings_names[iscreening])


cols = ["#3498db",  # blue
        "#e74c3c",  # red
        [0.984375, 0.7265625, 0],  # dark yellow
        "#2ecc71"]  # green

my_colors = list(islice(cycle(cols), None, times.size))

df = pd.DataFrame(times.T, columns=screenings_names)
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
df.plot(kind='bar', ax=ax, rot=0, color=my_colors)
plt.xticks(range(len(eps_)), [r"%s" % t for t in eps_])
plt.xlabel(r"-log$_{10}$(duality gap)")
plt.ylabel("Time (s)")
plt.grid(color='w')
leg = plt.legend(loc='best')
plt.tight_layout()
# plt.savefig("bench_screening.svg", format="svg")
# plt.savefig("bench_screening.png", format="png")
plt.show()
