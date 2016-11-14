

import numpy as np
cimport numpy as np

from cpython cimport bool
from libcpp cimport bool

cdef extern from "src/slic.cuh":
	ctypedef float dtype
	ctypedef int itype
	ctypedef unsigned short uint16
	struct itype3:
		itype x, y, z
	struct dtype3:
		dtype x, y, z
	void slicSupervoxels(const dtype *h_src, itype *h_dest, const dtype compactness, \
                     const itype3 im_shape, const itype3 sp_shape, \
                     const itype3 window, const dtype3 spacing, \
                     const dtype min_size_ratio, const dtype max_size_ratio, \
                     const uint16 max_iter, const bool enforce_connectivity)


def slic_supervoxels(np.ndarray[dtype, ndim=3, mode='c'] src, dtype compactness=0.1,
		   		 	 sp_shape=(10,10,10), window=(1,1,1), spacing=(1.,1.,1.),
		   			 dtype min_size_ratio=0.5, dtype max_size_ratio=3.,
		   			 uint16 max_iter=5, bool enforce_connectivity=True):
	cdef itype3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
	cdef itype3 spshape = {'x': sp_shape[2], 'y': sp_shape[1], 'z': sp_shape[0]}
	cdef itype3 iwindow = {'x': window[2], 'y': window[1], 'z': window[0]}
	cdef dtype3 ispacing = {'x': spacing[2], 'y': spacing[1], 'z': spacing[0]}

	cdef np.ndarray[itype, ndim=3, mode='c'] out = np.zeros((im_shape.z, im_shape.y, im_shape.x), np.int32)

	slicSupervoxels(<dtype*>np.PyArray_DATA(src), <itype*>np.PyArray_DATA(out), compactness,
					im_shape, spshape, iwindow, ispacing,
					min_size_ratio, max_size_ratio,
					max_iter, enforce_connectivity)
	return out
