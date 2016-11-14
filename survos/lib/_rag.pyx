#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


import numpy as np
cimport numpy as np


def _create_rag_2d(np.ndarray[np.int32_t, ndim=2, mode='c'] splabels, int connectivity):
    cdef:
        Py_ssize_t i, j, idx
        Py_ssize_t rows = splabels.shape[0]
        Py_ssize_t cols = splabels.shape[1]
        Py_ssize_t N = rows * cols
        int target, current
        np.ndarray[np.int32_t, ndim=1, mode='c'] nodes = np.empty(N, dtype=np.int32)
        np.ndarray[np.int32_t, ndim=2, mode='c'] edges = np.full((N, connectivity//2), -1, dtype=np.int32)
        bint it, jt

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            nodes[idx] = splabels[i, j]
            if i < rows - 1:
                edges[idx, 0] = splabels[i+1, j]
            if j < cols - 1:
                edges[idx, 1] = splabels[i, j+1]

            if connectivity == 8:
                if i < rows - 1 and j < cols - 1:
                    edges[idx, 2] = splabels[i+1, j+1]
                if i > 0 and j < cols - 1:
                    edges[idx, 3] = splabels[i-1, j+1]

    return nodes, edges


def _create_rag_3d(np.ndarray[np.int32_t, ndim=3, mode='c'] splabels, int connectivity):
    cdef:
        Py_ssize_t i, j, k, idx
        Py_ssize_t depth = splabels.shape[0]
        Py_ssize_t rows = splabels.shape[1]
        Py_ssize_t cols = splabels.shape[2]
        Py_ssize_t N = depth * rows * cols
        int target, current
        np.ndarray[np.int32_t, ndim=1, mode='c'] nodes = np.empty(N, dtype=np.int32)
        np.ndarray[np.int32_t, ndim=2, mode='c'] edges = np.full((N, connectivity//2), -1, dtype=np.int32)

    for k in range(depth):
        for i in range(rows):
            for j in range(cols):
                idx = k * rows * cols + i * cols + j
                nodes[idx] = splabels[k, i, j]

                if k < depth-1:
                    edges[idx, 0] = splabels[k+1, i, j]
                if i < rows-1:
                    edges[idx, 1] = splabels[k, i+1, j]
                if j < cols-1:
                    edges[idx, 2] = splabels[k, i, j+1]

                if connectivity > 6:
                    if k < depth-1 and i < rows-1:
                        edges[idx, 3] = splabels[k+1, i+1, j]
                    if k > 0 and i < rows-1:
                        edges[idx, 4] = splabels[k-1, i+1, j]

                    if k < depth-1 and j < cols-1:
                        edges[idx, 5] = splabels[k+1, i, j+1]
                    if k > 0 and j < cols-1:
                        edges[idx, 6] = splabels[k-1, i, j+1]

                    if i < rows-1 and j < cols-1:
                        edges[idx, 7] = splabels[k, i+1, j+1]
                    if i > 0 and j < cols-1:
                        edges[idx, 8] = splabels[k, i-1, j+1]

                if connectivity > 18:
                    if k < depth-1 and i < rows-1 and j < cols-1:
                        edges[idx, 9] = splabels[k+1, i+1, j+1]
                    if k < depth-1 and i < rows-1 and j > 0:
                        edges[idx, 10] = splabels[k+1, i+1, j-1]
                    if k < depth-1 and i > 0 and j < cols-1:
                        edges[idx, 11] = splabels[k+1, i-1, j+1]
                    if k > 0 and i < rows-1 and j < cols-1:
                        edges[idx, 12] = splabels[k-1, i+1, j+1]

    return nodes, edges


def _unique_rag3d(int[:,:,::1] splabels, int connectivity, int nsp):
    cdef:
        Py_ssize_t i, j, k, idx = 0
        Py_ssize_t depth = splabels.shape[0]
        Py_ssize_t rows = splabels.shape[1]
        Py_ssize_t cols = splabels.shape[2]
        Py_ssize_t N = depth * rows * cols
        int[:,::1] edges = np.empty((N * connectivity//2, 2), dtype=np.int32)

        int target, current
        set neighbors = set()

    for k in range(depth):
        for i in range(rows):
            for j in range(cols):
                current = splabels[k, i, j]

                if k < depth-1:
                    target = splabels[k+1, i, j]
                    if current != target and (current*nsp+target) not in neighbors:
                        edges[idx, 0] = current
                        edges[idx, 1] = target
                        idx += 1
                        neighbors |= set([current*nsp+target])
                        neighbors |= set([target*nsp+current])
                if i < rows-1:
                    target = splabels[k, i+1, j]
                    if current != target and (current*nsp+target) not in neighbors:
                        edges[idx, 0] = current
                        edges[idx, 1] = target
                        idx += 1
                        neighbors |= set([current*nsp+target])
                        neighbors |= set([target*nsp+current])
                if j < cols-1:
                    target = splabels[k, i, j+1]
                    if current != target and (current*nsp+target) not in neighbors:
                        edges[idx, 0] = current
                        edges[idx, 1] = target
                        idx += 1
                        neighbors |= set([current*nsp+target])
                        neighbors |= set([target*nsp+current])

    return np.asarray(edges[:idx, :])
