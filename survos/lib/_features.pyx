#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


import numpy as np
cimport numpy as np

def find_boundaries(int[:, ::1] spslice):
    cdef int H = spslice.shape[0]
    cdef int W = spslice.shape[1]
    cdef int i, j

    cdef int[:, ::1] result = np.zeros((H, W), dtype=np.int32)

    for i in range(H-1):
        for j in range(W-1):
            if spslice[i, j] != spslice[i+1, j]:
                result[i, j] = 1
                result[i+1, j] = 1
            if spslice[i, j] != spslice[i, j+1]:
                result[i, j] = 1
                result[i, j+1] = 1

    return np.asarray(result)

def _position_2d(float[:, ::1] X):
    cdef int rows = X.shape[0]
    cdef int cols = X.shape[1]
    cdef int i, j
    cdef float[:, :, ::1] pos = np.zeros((rows, cols, 2), np.float32)

    for i in range(rows):
        for j in range(cols):
            pos[i, j, 0] = i
            pos[i, j, 1] = j

    return np.asarray(pos)


def _position_3d(float[:, :, ::1] X):
    cdef int dept = X.shape[0]
    cdef int rows = X.shape[1]
    cdef int cols = X.shape[2]
    cdef int i, j, k
    cdef float[:, :, :, ::1] pos = np.zeros((dept, rows, cols, 3), np.float32)

    for k in range(dept):
        for i in range(rows):
            for j in range(cols):
                pos[k, i, j, 0] = k
                pos[k, i, j, 1] = i
                pos[k, i, j, 2] = j

    return np.asarray(pos)

def _mask_values(int[::1] X, int[::1] values):
    cdef int N = X.shape[0]
    cdef int K = values.shape[0]
    cdef int i, j
    cdef unsigned char[::1] r = np.zeros(N, np.uint8)

    for i in range(N):
        for j in range(K):
            if X[i] == values[j]:
                r[i] = 1
                break

    return np.array(r, dtype=np.bool)


def _mask_values_set(int[::1] X, set values):
    cdef int N = X.shape[0]
    cdef int i, j
    cdef unsigned char[::1] r = np.zeros(N, np.uint8)

    for i in range(N):
        if X[i] in values:
            r[i] = 1

    return np.array(r, dtype=np.bool)
