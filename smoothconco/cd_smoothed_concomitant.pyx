
from libc.math cimport fabs, sqrt
from libc.stdlib cimport qsort
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double primal_value(int n_samples,
                         int n_features,
                         double sigma,
                         double * beta_data,
                         double norm_residual,
                         double lambda_) nogil:

    cdef double l1_norm = 0
    cdef double fval = 0
    cdef int i = 0
    cdef int inc = 1

    for i in range(n_features):
        l1_norm += fabs(beta_data[i])

    fval = 0.5 * ((1 / (n_samples * sigma)) * norm_residual ** 2 + sigma)

    return fval + lambda_ * l1_norm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual(int n_samples,
                 int n_features,
                 double * residual_data,
                 double * y_data,
                 double dual_scale,
                 double norm_residual,
                 double sigma_0,
                 double lambda_) nogil:

    cdef int inc = 1
    cdef double Ry = ddot(& n_samples, residual_data, & inc, y_data, & inc)

    if dual_scale != 0:
        dval = (lambda_ * Ry / dual_scale + 0.5 * sigma_0 *
          (1 - n_samples * (lambda_ ** 2) * (norm_residual / dual_scale) ** 2))

    return dval


cdef double segment_project(double x, double a, double b) nogil:
    return -fmax(-b, -fmax(x, a))


cdef double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0)


cdef double dual_gap(int n_samples,
                     int n_features,
                     double * residual_data,
                     double * y_data,
                     double * beta_data,
                     double dual_scale,
                     double norm_residual,
                     double sigma,
                     double sigma_0,
                     double lambda_) nogil:

    cdef double pobj = primal_value(n_samples, n_features, sigma, beta_data,
                                    norm_residual, lambda_)

    cdef double dobj = dual(n_samples, n_features, residual_data, y_data,
                            dual_scale, norm_residual, sigma_0, lambda_)

    return pobj - dobj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cd_smoothed_concomitant_fast(double[::1, :] X, double[:] y, double[:] beta,
                                 double[:] XTR, double[:] residual,
                                 double norm_residual, int n_samples,
                                 int n_features, double sqrt_n, double nrm_y,
                                 double[:] norm_X2, double lambda_, double eps,
                                 double sigma_0, int max_iter, int f):
    """
        Solve the concomitant estimation ie We jointly minimize in beta and
        sigma 1/(2 n sigma) ||y - X beta||^2 + sigma/2 + lambda_ ||beta||_1
    """

    cdef int i = 0
    cdef int k = 0
    cdef int j = 0
    cdef int inc = 1

    cdef double gap_t = 1
    cdef double nrm2_y = nrm_y * nrm_y
    cdef double double_tmp = 0
    cdef double mu = 0
    cdef double beta_old_j = 0
    cdef double * X_ptr = &X[0, 0]
    cdef double sigma = fmax(norm_residual / sqrt_n, sigma_0)
    cdef double l_sqn = lambda_ * sqrt_n
    cdef double norm2_residual = norm_residual ** 2
    cdef double l_n_sigma0 = lambda_ * n_samples * sigma_0

    for k in range(max_iter):

        if f != 0 and k % f == 0:

          # Compute dual point by dual scaling :
          # theta_k = residual / dual_scale
          dual_scale = fmax(fmax(l_sqn * norm_residual, l_n_sigma0),
                            abs_max(n_features, & XTR[0]))

          gap_t = dual_gap(n_samples, n_features, & residual[0], & y[0],
                           & beta[0], dual_scale, norm_residual, sigma,
                           sigma_0, lambda_)

          if gap_t <= eps * nrm_y:
              break

        for j in range(n_features):

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

    return (sigma, gap_t, norm_residual, k)
