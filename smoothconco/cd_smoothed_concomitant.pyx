# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fabs, sqrt
from libc.stdlib cimport qsort
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal
import numpy as np
cimport numpy as np
cimport cython


cdef:
    int inc = 1  # Default array increment for cython_blas operation
    int NO_SCREENING = 0
    int GAPSAFE = 1
    int WSTRT_SIGMA_0 = 2
    int BOUND = 3


cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef double abs_max(int n, double * a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef double max(int n, double * a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef double primal_value(int n_samples,
                         int n_features,
                         double sigma,
                         double * beta_data,
                         double norm_residual,
                         double lambda_) nogil:

    cdef:
        double l1_norm = 0
        double fval = 0
        int j = 0

    # TODO disabled_features
    for j in range(n_features):
        l1_norm += fabs(beta_data[j])

    fval = 0.5 * ((1 / (n_samples * sigma)) * norm_residual ** 2 + sigma)

    return fval + lambda_ * l1_norm


cdef double dual(int n_samples,
                 int n_features,
                 double * residual_data,
                 double * y_data,
                 double dual_scale,
                 double norm_residual,
                 double sigma_0,
                 double lambda_) nogil:

    cdef double Ry = ddot(& n_samples, residual_data, & inc, y_data, & inc)

    if dual_scale != 0:
        dval = (lambda_ * Ry / dual_scale + 0.5 * sigma_0 *
          (1 - n_samples * (lambda_ ** 2) * (norm_residual / dual_scale) ** 2))

    return dval


cdef double segment_project(double x, double a, double b) nogil:
    return -fmax(-b, -fmax(x, a))


cdef double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0)


def cd_smoothed_concomitant_fast(double[::1, :] X, double[:] y, double[:] beta,
                                 double[:] abs_XTy, double[:] XTR,
                                 double[:] residual,
                                 int[:] disabled_features,
                                 double norm_residual, double sqrt_n,
                                 double nrm_y, double[:] norm_X2,
                                 double lambda_, double eps,
                                 double sigma_0, int max_iter, int f,
                                 int screening,
                                 double[:] bound_cte,
                                 int wstr_plus=0):
    """
        Solve the concomitant estimation ie We jointly minimize in beta and
        sigma 1/(2 n sigma) ||y - X beta||^2 + sigma/2 + lambda_ ||beta||_1
    """

    cdef:
        int i = 0
        int k = 0
        int j = 0
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_active_features = n_features
        int yes_screen = False
        double gap_t = 1
        double nrm2_y = nrm_y * nrm_y
        double double_tmp = 0
        double mu = 0
        double beta_old_j = 0
        double * X_ptr = &X[0, 0]
        double sigma = fmax(norm_residual / sqrt_n, sigma_0)
        double norm2_residual = norm_residual ** 2
        double l_sqn = lambda_ * sqrt_n
        double l_n_sigma0 = lambda_ * n_samples * sigma_0
        double gamma_sup = 0.
        double gamma_inf = 0.
        double p_obj = 0.
        double d_obj = 0.
        double bound = 0.
        double sqrtn_over_nrmy = sqrt_n / nrm_y
        double[:] l_sqn_over_norm_X = np.zeros(n_features, order='F')

    with nogil:
        if wstr_plus == 0:
            for j in range(n_features):
                disabled_features[j] = 0

        if screening == BOUND:
            l_sqn_over_norm_X[j] = l_sqn / sqrt(norm_X2[j])

        for k in range(max_iter):

            if f != 0 and k % f == 0:

                # Compute dual point by dual scaling :
                # theta_k = residual / dual_scale
                double_tmp = 0
                for j in range(n_features):

                    if disabled_features[j] == 1:
                        continue
                    else:
                        double_tmp = fmax(double_tmp, fabs(XTR[j]))

                dual_scale = fmax(fmax(l_sqn * norm_residual, l_n_sigma0),
                                  double_tmp)

                pobj = primal_value(n_samples, n_features, sigma, & beta[0],
                                    norm_residual, lambda_)

                dobj = dual(n_samples, n_features, & residual[0], & y[0],
                            dual_scale, norm_residual, sigma_0, lambda_)

                gap_t = pobj - dobj

                if gap_t <= eps * nrm_y:
                    break

                if screening == GAPSAFE:

                    r = sqrt(2 * gap_t / (lambda_ ** 2 * sigma_0 * n_samples))

                for j in range(n_features):

                    if disabled_features[j] == 1:
                        continue

                    if screening == GAPSAFE:

                        r_normX_j = r * sqrt(norm_X2[j])
                        if r_normX_j >= 1:
                            # screening test obviously will fail
                            continue

                        yes_screen = fabs(XTR[j] / dual_scale) + r_normX_j < 1

                    if screening == BOUND:

                        gamma_sup = p_obj * sqrtn_over_nrmy
                        gamma_inf = (d_obj - sigma_0 / 2.) * sqrtn_over_nrmy

                        if abs_XTy[j] > gamma_sup:

                            double_tmp = sqrt(1 - gamma_sup ** 2) * bound_cte[j]
                            yes_screen = gamma_sup * abs_XTy[j] + \
                                         double_tmp < l_sqn_over_norm_X[j]

                        elif abs_XTy[j] < gamma_inf:

                            double_tmp = sqrt(1 - gamma_inf ** 2) * bound_cte[j]
                            yes_screen = gamma_inf * abs_XTy[j] + \
                                         double_tmp < l_sqn_over_norm_X[j]

                        else:
                            yes_screen = 1 < l_sqn_over_norm_X[j]

                    if yes_screen:
                        # Update residual
                        if beta[j] != 0:
                            # residual -= X[:, j] * (beta[j] - beta_old[j])
                            daxpy(& n_samples, & beta[j],
                                  X_ptr + j * n_samples, & inc, & residual[0], & inc)
                            norm_residual = dnrm2(& n_samples, & residual[0], & inc)
                            sigma = fmax(norm_residual / sqrt_n, sigma_0)
                            beta[j] = 0

                        # we "set" x_j to zero since the j_th feature is inactive
                        XTR[j] = 0
                        disabled_features[j] = 1
                        n_active_features -= 1

            for j in range(n_features):

                if disabled_features[j] == 1:
                    continue

                mu = lambda_ * n_samples * sigma / norm_X2[j]
                beta_old_j = beta[j]
                XTR[j] = ddot(& n_samples, X_ptr + j * n_samples, & inc,
                              & residual[0], & inc)
                beta[j] = ST(mu, beta[j] + XTR[j] / norm_X2[j])

                if beta[j] != beta_old_j:

                    double_tmp = beta_old_j - beta[j]
                    # Update residual
                    daxpy(& n_samples, & double_tmp, X_ptr + j * n_samples, & inc,
                          & residual[0], & inc)
                    # norm_residual = dnrm2(& n_samples, & residual[0], & inc)
                    norm2_residual += 2. * double_tmp * XTR[j] + norm_X2[j] * (double_tmp ** 2)
                    norm_residual = sqrt(norm2_residual)
                    sigma = fmax(norm_residual / sqrt_n, sigma_0)

    return beta, sigma, gap_t, norm_residual, k, n_active_features
