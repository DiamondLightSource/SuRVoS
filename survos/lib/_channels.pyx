#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


import numpy as np
cimport numpy as np

from cpython cimport bool
from libcpp cimport bool
from libc.math cimport sqrt, acos, cos, M_PI


cdef extern from "src/symmetric_eigvals3S.cuh":
	void symmetric3_eigenvalues(const float *Hzz, const float *Hzy, const float *Hzx,
								const float *Hyy, const float *Hyx, const float *Hxx,
								float* out, size_t total_size, bool doabs) nogil except +


def symmetric_eigvals3S_gpu(np.ndarray[np.float32_t, ndim=3, mode='c'] Hzz,
							np.ndarray[np.float32_t, ndim=3, mode='c'] Hzy,
							np.ndarray[np.float32_t, ndim=3, mode='c'] Hzx,
							np.ndarray[np.float32_t, ndim=3, mode='c'] Hyy,
							np.ndarray[np.float32_t, ndim=3, mode='c'] Hyx,
							np.ndarray[np.float32_t, ndim=3, mode='c'] Hxx,
							bool doabs=False):
	cdef int depth = Hzz.shape[0]
	cdef int height = Hzz.shape[1]
	cdef int width = Hzz.shape[2]
	cdef size_t total = depth * height * width

	cdef np.ndarray[np.float32_t, ndim=4, mode='c'] result
	result = np.zeros((depth, height, width, 3), np.float32)

	cdef float *hzz = <float*>np.PyArray_DATA(Hzz)
	cdef float *hzy = <float*>np.PyArray_DATA(Hzy)
	cdef float *hzx = <float*>np.PyArray_DATA(Hzx)
	cdef float *hyy = <float*>np.PyArray_DATA(Hyy)
	cdef float *hyx = <float*>np.PyArray_DATA(Hyx)
	cdef float *hxx = <float*>np.PyArray_DATA(Hxx)
	cdef float *out = <float*>np.PyArray_DATA(result)

	with nogil:
		symmetric3_eigenvalues(hzz, hzy, hzx, hyy, hyx, hxx, out, total, doabs)

	return result


def symmetric_eig(float[:, :, ::1] Hzz, float[:, :, ::1] Hzy, float[:, :, ::1] Hzx,
				  float[:, :, ::1] Hyy, float[:, :, ::1] Hyx, float[:, :, ::1] Hxx):

	cdef int depth = Hzz.shape[0]
	cdef int height = Hzz.shape[1]
	cdef int width = Hzz.shape[2]
	cdef int k, i, j

	cdef float[:, ::1] tmp = np.zeros((3,3), np.float32)
	cdef np.ndarray[np.float32_t, ndim=4, mode='c'] result
	result = np.zeros((depth, height, width, 3), np.float32)

	for k in range(depth):
		for i in range(height):
			for j in range(width):
				tmp[0, 0] = Hzz[k, i, j]
				tmp[0, 1] = Hzy[k, i, j]
				tmp[0, 2] = Hzx[k, i, j]

				tmp[1, 0] = Hzy[k, i, j]
				tmp[1, 1] = Hyy[k, i, j]
				tmp[1, 2] = Hyx[k, i, j]

				tmp[2, 0] = Hzx[k, i, j]
				tmp[2, 1] = Hyx[k, i, j]
				tmp[2, 2] = Hxx[k, i, j]

				result[k, i, j, :] = eigvalues3S(tmp)

	return result


cpdef float determinant3S(float[:, ::1] A):
	return A[0,0] * (A[1,1]*A[2,2] - A[1,2]*A[2,1]) \
		 - A[0,1] * (A[1,0]*A[2,2] - A[1,2]*A[2,0]) \
		 + A[0,2] * (A[1,0]*A[2,1] - A[1,1]*A[2,0])


cpdef float[::1] eigvalues3S(float[:, ::1] A):
	cdef float[::1] eigv = np.zeros(3, np.float32)
	cdef float[:, ::1] B = np.zeros((3,3), np.float32)
	cdef float q, p, p2, r, phi, p1, invp

	p1 = (A[0,1]**2) + (A[0,2]**2) + (A[1,2]**2)

	if p1 == 0: # A is diagonal
		eigv[0] = A[0,0]
		eigv[1] = A[1,1]
		eigv[2] = A[2,2]
	else:
		q  = (A[0,0] + A[1,1] + A[2,2]) / 3.
		p2 = (A[0,0] - q)**2 + (A[1,1] - q)**2 + (A[2,2] - q)**2 + 2 * p1
		p  = sqrt(p2 / 6.)
		invp = 1. / p

		for i in range(3):
			for j in range(3):
				if i == j:
					B[i, j] = invp * (A[i, j] - q)
				else:
					B[i, j] = invp * A[i, j]

		r = determinant3S(B) / 2.

		if r <= -1:
			phi = M_PI / 3.
		elif r >= 1:
			phi = 0
		else:
			phi = acos(r) / 3.

		eigv[0] = q + 2. * p * cos(phi)
		eigv[2] = q + 2. * p * cos(phi + (2. * M_PI / 3.))
		eigv[1] = 3. * q - eigv[0] - eigv[2]

	return eigv
