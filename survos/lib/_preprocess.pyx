# distutils: language = c++

import numpy as np
cimport numpy as np


from libcpp cimport bool


cdef extern from "src/cuda.cuh":
	int ireduce_gpu "reduce<int>" (const int* src, Py_ssize_t num)
	float freduce_gpu "reduce<float>" (const float* src, Py_ssize_t num)
	long lreduce_gpu "reduce<long>" (const long* src, Py_ssize_t num)
	double dreduce_gpu "reduce<double>" (const double* src, Py_ssize_t num)

	int imax_gpu "reduceMax<int>" (const int* src, Py_ssize_t num)
	float fmax_gpu "reduceMax<float>" (const float* src, Py_ssize_t num)
	long lmax_gpu "reduceMax<long>" (const long* src, Py_ssize_t num)
	double dmax_gpu "reduceMax<double>" (const double* src, Py_ssize_t num)


cdef extern from "src/preprocess.cuh":
	struct int3:
		int x, y, z
	struct float3:
		float x, y, z

	void anidiffusion_gpu "anidiffusion" \
					 (const float* src, float* dst, const float lamda,
				  	  const int3 shape, const float gamma, const int mode,
				  	  const int maxIter, const float eps)
	void tvdenoising_gpu "tvdenoising" \
					(const float* src, float* dst, float lamda,
				 	 float3 spacing, int3 shape, int maxIter, float eps)

	void tvchambolle_gpu "tvchambolle" \
					(const float* src, float* dst, float lamda,
				 	 float tau, int3 shape, int maxIter, float eps)

	void tvbregman_gpu "tvbregman" \
					(const float* src, float* dst, float lamda, float mu,
               		 int3 shape, int maxIter, float eps, bool isotropic, int method)

	void tvchambolle1st_gpu "tvchambolle1st" \
					(const float* src, float* dst,
                     float lamda, float rho, float theta, float sigma, float gamma,
                     int3 shape, int maxIter, float eps, bool l2)


def greduce(data):
	if data.dtype == np.int32:
		return ireduce(data)
	elif data.dtype == np.int64:
		return lreduce(data)
	elif data.dtype == np.float32:
		return freduce(data)
	elif data.dtype == np.float64:
		return dreduce(data)
	else:
		raise Exception("Data type not supported");

cdef float freduce(np.ndarray[np.float32_t, ndim=1, mode='c'] src):
	return freduce_gpu(<float*>src.data, src.size)

cdef int ireduce(np.ndarray[np.int32_t, ndim=1, mode='c'] src):
	return ireduce_gpu(<int*>src.data, src.size)

cdef double dreduce(np.ndarray[np.float64_t, ndim=1, mode='c'] src):
	return dreduce_gpu(<double*>src.data, src.size)

cdef long lreduce(np.ndarray[np.int64_t, ndim=1, mode='c'] src):
	return lreduce_gpu(<long*>src.data, src.size)


def gmax(data):
	if data.dtype == np.int32:
		return imax(data)
	elif data.dtype == np.int64:
		return lmax(data)
	elif data.dtype == np.float32:
		return fmax(data)
	elif data.dtype == np.float64:
		return dmax(data)
	else:
		raise Exception("Data type not supported");

cdef float fmax(np.ndarray[np.float32_t, ndim=1, mode='c'] src):
	return fmax_gpu(<float*>src.data, src.size)

cdef int imax(np.ndarray[np.int32_t, ndim=1, mode='c'] src):
	return imax_gpu(<int*>src.data, src.size)

cdef double dmax(np.ndarray[np.float64_t, ndim=1, mode='c'] src):
	return dmax_gpu(<double*>src.data, src.size)

cdef long lmax(np.ndarray[np.int64_t, ndim=1, mode='c'] src):
	return lmax_gpu(<long*>src.data, src.size)


def tvdenoising(np.ndarray[np.float32_t, ndim=3, mode='c'] src, float lamda=10.,
				spacing=(1.,1.,1.), max_iter=100, eps=1e-6):
	cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
	cdef float3 space = {'x': spacing[2], 'y': spacing[1], 'z': spacing[0]}

	cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] dst = np.zeros((D, H, W), np.float32)
	cdef float *src_ptr = <float*>src.data
	cdef float *dst_ptr = <float*>dst.data

	tvdenoising_gpu(src_ptr, dst_ptr, lamda, space, shape, max_iter, eps)

	return dst


def tvchambolle(np.ndarray[np.float32_t, ndim=3, mode='c'] src, float lamda=10.,
				rho=1./8., max_iter=100, eps=1e-6):
	cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
	cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] dst = np.zeros((D, H, W), np.float32)
	cdef float *src_ptr = <float*>src.data
	cdef float *dst_ptr = <float*>dst.data

	tvchambolle_gpu(src_ptr, dst_ptr, lamda, rho, shape, max_iter, eps)

	return dst

def tvbregman(np.ndarray[np.float32_t, ndim=3, mode='c'] src, float lamda, float mu,
			  max_iter=100, eps=1e-6, bool isotropic=True, int method=2):
	cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
	cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] dst = np.zeros((D, H, W), np.float32)
	cdef float *src_ptr = <float*>src.data
	cdef float *dst_ptr = <float*>dst.data

	tvbregman_gpu(src_ptr, dst_ptr, lamda, mu, shape, max_iter, eps, isotropic, method)

	return dst


def tvchambolle1st(np.ndarray[np.float32_t, ndim=3, mode='c'] src, float lamda=10.,
				   float rho=0.01, float theta=1.0, float sigma=0.5, float gamma=7.,
				   int max_iter=100, float eps=1e-6, bool l2=True):
	cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
	cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] dst = np.zeros((D, H, W), np.float32)
	cdef float *src_ptr = <float*>src.data
	cdef float *dst_ptr = <float*>dst.data

	tvchambolle1st_gpu(src_ptr, dst_ptr, lamda, rho, theta, sigma, gamma,
					   shape, max_iter, eps, l2)

	return dst


def anidiffusion(np.ndarray[np.float32_t, ndim=3, mode='c'] src, float gamma=10.,
				 float lamda=1./8., int mode=3, max_iter=100, eps=1e-6):
	cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}

	cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
	cdef np.ndarray[np.float32_t, ndim=3, mode='c'] dst = np.zeros((D, H, W), np.float32)

	cdef float *src_ptr = <float*>src.data
	cdef float *dst_ptr = <float*>dst.data

	anidiffusion_gpu(src_ptr, dst_ptr, lamda, shape, gamma, mode, max_iter, eps)

	return dst
