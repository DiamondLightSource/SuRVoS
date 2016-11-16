###############################################################
# QPBO algorithm by Vladimir Kolmogorov.
#
# Optimizing binary MRFs via extended roof duality.
# C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer.
# CVPR 2007.
#
# Python QPBO bindings based on pyqpbo:
# https://github.com/pystruct/pyqpbo
###############################################################


import numpy as np
cimport numpy as np
from libcpp cimport bool
from time import time

cdef extern from "stdlib.h":
    void srand(unsigned int seed)

ctypedef int NodeId
ctypedef int EdgeId

cdef extern from "QPBO.h":
    cdef cppclass QPBO[REAL]:
        QPBO(int node_num_max, int edge_num_max) nogil except +
        bool Save(char* filename, int format=0) nogil except +
        bool Load(char* filename) nogil except +
        void Reset() nogil except +
        NodeId AddNode(int num)  nogil except +
        void AddUnaryTerm(NodeId i, REAL E0, REAL E1) nogil except +
        EdgeId AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11)  nogil except +
        void AddPairwiseTerm(EdgeId e, NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11)  nogil except +
        int GetLabel(NodeId i)  nogil except +
        void Solve()  nogil except +
        REAL ComputeTwiceEnergy(int)  nogil except +
        void ComputeWeakPersistencies()  nogil except +
        bool Improve()  nogil except +


def solve_binary(int[:, ::1] edges, float[:, ::1] unary,
                 float[::1] pairwise, float[:, ::1] lcost):

    cdef int n_nodes = unary.shape[0]
    cdef int n_edges = edges.shape[0]

    if unary.shape[1] != 2:
        raise ValueError("Unary_cost must be of shape (n_nodes, 2).")
    if edges.shape[1] != 2:
        raise ValueError("edges must be of shape (n_edges, 2).")
    if lcost.shape[0] != lcost.shape[1]:
        raise ValueError("pairwise_cost must be square matrix.")

    cdef int[::1] result = np.zeros(n_nodes, np.int32)

    cdef float* unary_ptr = <float*>&(unary[0,0])
    cdef int* edge_ptr = <int*>&(edges[0,0])
    cdef float* pairwise_ptr = <float*>&(pairwise[0])
    cdef int* result_ptr = <int*>&(result[0])

    cdef int e, i
    cdef float e00 = lcost[0, 0]
    cdef float e10 = lcost[1, 0]
    cdef float e01 = lcost[0, 1]
    cdef float e11 = lcost[1, 1]
    cdef float w = 0
    cdef int e1, e2

    # create qpbo object
    cdef QPBO[float] * q = new QPBO[float](n_nodes, n_edges)

    with nogil:
        q.AddNode(n_nodes)

        for i in range(n_nodes):
            q.AddUnaryTerm(i, unary_ptr[2 * i], unary_ptr[2 * i + 1])

        for i in range(n_edges):
            e1 = edge_ptr[2 * i]
            e2 = edge_ptr[2 * i + 1]
            w = pairwise_ptr[i]
            q.AddPairwiseTerm(e1, e2, w * e00, w * e10, w * e01, w * e11)

        q.Solve()
        q.ComputeWeakPersistencies()

        for i in range(n_nodes):
            result_ptr[i] = q.GetLabel(i)

    del q

    return np.asarray(result)


def solve_aexpansion(int[:, ::1] edges, float[:, ::1] unary,
                     float[::1] pairwise, float[:, ::1] lcost,
                     int[::1] init_labels=None, int n_iter=5,
                     bool verbose=False, random_seed=42):

    cdef int n_nodes = unary.shape[0]
    cdef int n_labels =  unary.shape[1]
    cdef int n_edges = edges.shape[0]
    cdef int old_label
    cdef int label
    cdef int changes
    cdef float e00, e01, e10, e11
    cdef int edge0, edge1
    cdef bool improve
    cdef int[::1] result

    cdef int i, e, alpha, n

    if random_seed is None:
        rnd_state = np.random.mtrand.RandomState()
        srand(time())
    else:
        rnd_state = np.random.mtrand.RandomState(random_seed)
        srand(random_seed)

    # initial guess
    if init_labels is None:
        result = np.zeros(n_nodes, dtype=np.int32)
    else:
        result = init_labels.copy()

    cdef float* unary_ptr = <float*>&(unary[0,0])
    cdef int* edge_ptr = <int*>&(edges[0,0])
    cdef float* pairwise_ptr = <float*>&(pairwise[0])
    cdef float* lcost_ptr = <float*>&(lcost[0,0])
    cdef int* result_ptr = <int*>&(result[0])

    # create qpbo object
    cdef QPBO[float]* q = new QPBO[float](n_nodes, n_edges)

    #cdef int* data_ptr = <int*> unary_cost.data
    for n in range(n_iter):
        if verbose > 0:
            print("iteration: %d" % n)
        changes = 0
        for alpha in rnd_state.permutation(n_labels):
            q.AddNode(n_nodes)
            unary_ptr_c = unary_ptr
            for i in range(n_nodes):
                if alpha == result_ptr[i]:
                    q.AddUnaryTerm(i, unary_ptr_c[result_ptr[i]], 9999999)
                else:
                    q.AddUnaryTerm(i, unary_ptr_c[result_ptr[i]], unary_ptr_c[alpha])
                unary_ptr_c += n_labels
            for e in range(n_edges):
                edge0 = edge_ptr[2 * e]
                edge1 = edge_ptr[2 * e + 1]
                #down
                e00 = pairwise_ptr[e] * lcost_ptr[result_ptr[edge0] * n_labels + result_ptr[edge1]]
                e01 = pairwise_ptr[e] * lcost_ptr[result_ptr[edge0] * n_labels + alpha]
                e10 = pairwise_ptr[e] * lcost_ptr[alpha * n_labels + result_ptr[edge1]]
                e11 = pairwise_ptr[e] * lcost_ptr[alpha * n_labels + alpha]
                q.AddPairwiseTerm(edge0, edge1, e00, e01, e10, e11)

            q.Solve()
            q.ComputeWeakPersistencies()
            improve = True
            while improve:
                improve = q.Improve()

            for i in range(n_nodes):
                old_label = result_ptr[i]
                label = q.GetLabel(i)
                if label == 1:
                    result_ptr[i] = alpha
                    changes += 1
                if label < 0:
                    print("LABEL <0 !!!")
            # compute energy:
            q.Reset()
        if verbose > 0:
            print("changes: %d" % changes)
        if changes == 0:
            break
    del q
    return np.asarray(result)
