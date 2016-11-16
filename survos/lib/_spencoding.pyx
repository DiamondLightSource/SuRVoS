#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


import numpy as np
cimport numpy as np


def _sp_labels(int[::1] sp, short[::1] labels, int nsp, int nlbl, double scale):
    cdef Py_ssize_t N = sp.shape[0]
    cdef Py_ssize_t i, s, cmax, curr
    cdef short j, l
    cdef double smin

    cdef int[::1] sizes = np.zeros(nsp, dtype=np.int32)
    cdef int[:, ::1] counts = np.zeros((nsp, nlbl), dtype=np.int32)
    cdef short[::1] out = np.full(nsp, -1, dtype=np.int16)

    for i in range(N):
        l = labels[i]
        s = sp[i]
        sizes[s] += 1
        if l >= 0:
            counts[s, l] += 1

    for i in range(nsp):
        cmax = 0
        smin = sizes[i] * scale
        for j in range(nlbl):
            curr = counts[i, j]
            if curr > cmax and <double>curr >= smin:
                cmax = curr
                out[i] = j

    return np.asarray(out)


def _spmeans(float[:, ::1] data, int[::1] labels, int nsp):
    cdef int n
    cdef int N = labels.shape[0]
    cdef int K = data.shape[1]
    cdef float[:, ::1] F = np.zeros((nsp, K), np.float32)
    cdef int[::1] sizes = np.zeros(nsp, np.int32)
    cdef int l, b
    cdef float t

    for n in range(N):
        l = labels[n]
        sizes[l] += 1

        for z in range(K):
            t = data[n, z]
            F[l, z] += t

    for n in range(nsp):
        if sizes[n] > 0:
            for z in range(K):
                F[n, z] /= sizes[n]

    return np.asarray(F)


def _sphist(int[::1] data, int[::1] labels, int nsp, int nbins):
    cdef int n
    cdef int N = labels.shape[0]
    cdef float[:, ::1] F = np.zeros((nsp, nbins), np.float32)
    cdef int l, t

    for n in range(N):
        l = labels[n]
        t = data[n]
        F[l, t] += 1

    return np.asarray(F)


def _spstats(float[:, ::1] data, int[::1] labels, int nsp):
    cdef int i, j, k, sp
    cdef int N = data.shape[0], K = data.shape[1]
    cdef int[::1] sizes = np.zeros(nsp, dtype=np.int32)
    cdef float diff
    cdef float[:, ::1] means = np.zeros((nsp, K), np.float32)
    cdef float[:, :, ::1] covars = np.zeros((nsp, K, K), np.float32)

    for n in range(N):
        sp = labels[n]
        sizes[sp] += 1
        for z in range(K):
            means[sp, z] += data[n, z]

    for n in range(nsp):
        for z in range(K):
            means[n, z] /= sizes[n]

    for i in range(N):
        sp = labels[i]
        for j in range(K):
            covars[sp, j, j] += (data[i, j] - means[sp, j]) * (data[i, j] - means[sp, j]) / sizes[sp]
            for k in range(j+1, K):
                diff = (data[i, j] - means[sp, j]) * (data[i, k] - means[sp, k]) / sizes[sp]
                covars[sp, j, k] += diff
                covars[sp, k, j] += diff

    return np.asarray(means), np.asarray(covars)


def _spcenter2d(int[:, ::1] labels, int nsp):
    cdef int i, j
    cdef int H = labels.shape[0], W = labels.shape[1]
    cdef float[:, ::1] centers = np.zeros((nsp, 2), np.float32)
    cdef int[::1] count = np.zeros(nsp, np.int32)
    cdef int n

    for i in range(H):
        for j in range(W):
            n = labels[i, j]
            centers[n, 0] += i
            centers[n, 1] += j
            count[n] += 1

    for n in range(nsp):
        centers[n, 0] /= count[n]
        centers[n, 1] /= count[n]

    return np.asarray(centers)


def _spcenter3d(int[:, :, ::1] labels, int nsp):
    cdef int i, j, k
    cdef int D = labels.shape[0], H = labels.shape[1], W = labels.shape[2]
    cdef float[:, ::1] centers = np.zeros((nsp, 3), np.float32)
    cdef int[::1] count = np.zeros(nsp, np.int32)
    cdef int n

    for k in range(D):
        for i in range(H):
            for j in range(W):
                n = labels[k, i, j]
                centers[n, 0] += k
                centers[n, 1] += i
                centers[n, 2] += j
                count[n] += 1

    for n in range(nsp):
        centers[n, 0] /= count[n]
        centers[n, 1] /= count[n]
        centers[n, 2] /= count[n]

    return np.asarray(centers)
