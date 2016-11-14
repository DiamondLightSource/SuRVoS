#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: language='c++'

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cdef extern from "src/zernike/ZernikeDescriptor.h":
    cdef cppclass ZernikeDescriptor[T, Tin]:
        ZernikeDescriptor(T *voxels, int dim, int order) except +
        vector[T] GetInvariants() except +

def zernike_descriptors(np.ndarray[np.float32_t, ndim=3, mode='c'] volume,
                        int order=8):
    cdef int dim = volume.shape[0]
    cdef float *data_ptr= <float*>volume.data

    cdef ZernikeDescriptor[float, float] *zern = new ZernikeDescriptor[float, float](data_ptr, dim, order)
    result = np.array(zern.GetInvariants(), dtype=np.float32)

    del zern
    return result
