#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport cython
cimport numpy as cnp


from libc.math cimport sqrt, exp


cdef double dmin(double a, double b):
    if a <= b:
        return a
    return b


def bhattacharya(cnp.ndarray[cnp.float_t, ndim=1] Xa,
                 cnp.ndarray[cnp.float_t, ndim=1] Xb,
                 double gamma=0):

    cdef cnp.intp_t i, N = Xa.shape[1]
    cdef cnp.float_t* Xa_p = <cnp.float_t*>Xa.data
    cdef cnp.float_t* Xb_p = <cnp.float_t*>Xb.data
    cdef cnp.float_t result = 0

    with nogil:
        for i in range(N):
            result += sqrt(Xa_p[i] * Xb_p[i])

    return result


def intersection(cnp.ndarray[cnp.float_t, ndim=1] Xa,
                 cnp.ndarray[cnp.float_t, ndim=1] Xb,
                 double gamma=0):
    cdef cnp.intp_t i, N = Xa.shape[1]
    cdef cnp.float_t* Xa_p = <cnp.float_t*>Xa.data
    cdef cnp.float_t* Xb_p = <cnp.float_t*>Xb.data
    cdef cnp.float_t result = 0
    cdef cnp.float_t a, b

    with nogil:
        for i in range(N):
            a = Xa_p[i]
            b = Xb_p[i]
            if a <= b:
                result += a
            else:
                result += b

    return result


def rbf_kernel(cnp.ndarray[cnp.float_t, ndim=1] Xa,
               cnp.ndarray[cnp.float_t, ndim=1] Xb,
               double gamma=0):
    cdef cnp.intp_t i, N = Xa.shape[1]
    cdef cnp.float_t* Xa_p = <cnp.float_t*>Xa.data
    cdef cnp.float_t* Xb_p = <cnp.float_t*>Xb.data
    cdef cnp.float_t result = 0
    cdef cnp.float_t a, b

    with nogil:
        for i in range(N):
            a = Xa_p[i]
            b = Xb_p[i]
            result += (a - b)**2

    return exp(-gamma * result)


def chi2_kernel(cnp.ndarray[cnp.float_t, ndim=1] Xa,
                cnp.ndarray[cnp.float_t, ndim=1] Xb,
                double gamma=0):
    cdef cnp.intp_t i, N = Xa.shape[1]
    cdef cnp.float_t* Xa_p = <cnp.float_t*>Xa.data
    cdef cnp.float_t* Xb_p = <cnp.float_t*>Xb.data
    cdef cnp.float_t result = 0
    cdef cnp.float_t denom, nom

    with nogil:
        for i in range(N):
            denom = (Xa_p[i] - Xb_p[i])
            nom = (Xa_p[i] + Xb_p[i])
            if nom != 0:
                result += denom * denom / nom

    return exp(-gamma * result)
