

import numpy as np
cimport numpy as np

from cpython cimport bool
from libcpp cimport bool

cdef extern from "src/slic.cuh":
    struct int3:
        int x, y, z
    struct float3:
        float x, y, z
    void slicSupervoxels(const float *h_src, int *h_dest,
                         const float compactness, \
                         const int3 im_shape, const int3 sp_shape, \
                         const int3 window, const float3 spacing, \
                         float min_size_ratio, float max_size_ratio, \
                         unsigned short max_iter, bool enforce_connectivity,\
                         int gpu) nogil except +


def slic_supervoxels(float[:, :, ::1] src, float compactness=0.1,
                     tuple sp_shape=(10,10,10), tuple window=(1,1,1),
                     tuple spacing=(1.,1.,1.), float min_size_ratio=0.5,
                     float max_size_ratio=3., unsigned int max_iter=5,
                     bool enforce_connectivity=True, int gpu=-1):

    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 spshape = {'x': sp_shape[2], 'y': sp_shape[1], 'z': sp_shape[0]}
    cdef int3 iwindow = {'x': window[2], 'y': window[1], 'z': window[0]}
    cdef float3 ispacing = {'x': spacing[2], 'y': spacing[1], 'z': spacing[0]}

    cdef int[:, :, ::1] out = np.zeros((im_shape.z, im_shape.y, im_shape.x), np.int32)
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef int *out_ptr = <int*>&(out[0,0,0])

    with nogil:
        slicSupervoxels(src_ptr, out_ptr, compactness,
                        im_shape, spshape, iwindow, ispacing,
                        min_size_ratio, max_size_ratio,
                        max_iter, enforce_connectivity, gpu)

    return np.asarray(out)
