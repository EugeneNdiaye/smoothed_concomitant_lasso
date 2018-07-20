from itertools import product
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.6)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


set_style()
n_folds = 5  # Number of fold used in each cross validation
scale_sigma_0 = 1e-2
eps = 1e-4
sigma = 1
max_iter = 5000
n_simulations = 50
n_lambdas = 100
n_samples = 100
sparsitys = [0.8]
correlations = [0.]
snrs = [10]
n_features_grid = [200]

N_JOBS = -1
N_JOBS_ITER = -1

N_JOBS = 1
N_JOBS_ITER = -1


for n_features, snr, sparsity, corr in \
        product(n_features_grid, snrs, sparsitys, correlations):
    if ((1 - sparsity) * n_features) > n_samples:
        print("skipping")
        continue

    name = ("s" + str(int(10 * sparsity)) +
            "c" + str(int(10 * corr)) +
            "n" + str(n_samples) +
            "p" + str(n_features) +
            "snr" + str(int(10 * snr)))

    estimators = pd.read_csv("results/" + name + ".csv")
    # plot stuff in the same order as Reid
    est_ordered = ['OR', 'L_CV', 'L_LS_CV', 'L_CV_LS', 'L_RCV',
                   'SC_CV', 'SC_CV_LS', 'SC_LS_CV',
                   'SZ', 'SZ_LS', 'SZ_CV',
                   'L_U', 'L_U_LS',
                   'D2', 'SBvG_CV', 'SQRT-Lasso_CV']
    estimators[estimators['method'].isin(est_ordered)]

    fig = plt.figure(figsize=(13., 4.5))
    plt.axhline(y=sigma, color='r', linestyle="-", zorder=0, linewidth=2)
    df = estimators.copy()
    df.columns = ['method', r'$\hat{\sigma}$', 'time']
    pal = dict()
    for est in est_ordered:
        if "U" in est:
            pal[est] = "#3498db"  # blue
        elif "L" == est[:1]:
            pal[est] = "#e74c3c"  # red
        elif "SC" in est:
            pal[est] = "#2ecc71"  # green
        elif "SZ" in est:
            pal[est] = [0.984375, 0.7265625, 0]  # dark yellow
            # pal[est] = "#34495e"  # grey
        elif "D" in est:
            pal[est] = [1, 1, 0.9]  # light yellow
        elif "SBvG" in est:
            pal[est] = "#ffaf00"
        else:
            pal[est] = "#95a5a6"  # light grey

    ax = sns.boxplot(data=df, x='method', y=r'$\hat{\sigma}$', palette=pal,
                     order=est_ordered)
    ax.text(0.01, 0.06, 'SNR=%s' % snr, bbox={'pad': 10, 'facecolor': 'white'},
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='k', fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
    patches = [c for c in ax.get_children()
               if isinstance(c, matplotlib.patches.PathPatch)]
    for p, est in zip(patches, est_ordered):
        if "LS" in est:
            p.set_hatch('//')

    ax.set_xlabel('')
    # ax.set_ylim([0.5, 1.5])
    sns.despine(left=True)
    # plt.title(name)
    plt.tight_layout()
    plt.savefig("Images/" + name + ".svg", format="svg")
    plt.savefig("Images/" + name + ".pdf", format="pdf")
    plt.savefig("Images/" + name + ".png", format="png")

    est_ordered_cv = [est for est in est_ordered if 'CV' in est]
    time_L_CV = df[df.method == 'L_CV']['time'].mean()
    df_cv = df[df['method'].isin(est_ordered_cv)]
    df_cv = df_cv.assign(time=lambda x: np.log10(x.time / time_L_CV))
    df_cv = df_cv[df_cv.method != 'L_CV']
    est_ordered_cv.pop(est_ordered_cv.index('L_CV'))
    fig = plt.figure(figsize=(8., 4.5))
    ax = sns.barplot(data=df_cv, x='method', y='time', palette=pal,
                     order=est_ordered_cv)
    ax.set_xlabel('')
    ax.set_ylabel(r'$\log_{10}($time / time$_{L\_CV})$')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
    sns.despine(left=True)
    patches = [c for c in ax.get_children()
               if isinstance(c, matplotlib.patches.Rectangle)]
    for p, est in zip(patches, est_ordered_cv):
        if "LS" in est:
            p.set_hatch('//')
    plt.tight_layout()
    plt.savefig("Images/time_" + name + ".svg", format="svg")
    plt.savefig("Images/time_" + name + ".pdf", format="pdf")
    plt.savefig("Images/time_" + name + ".png", format="png")
    plt.show()
