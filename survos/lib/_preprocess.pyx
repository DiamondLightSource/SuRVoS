# distutils: language = c++

import numpy as np
cimport numpy as np


from libcpp cimport bool


cdef extern from "preprocess.h":
    int ireduce_gpu "reduce<int>" (const int* src, Py_ssize_t num, int gpu) nogil except +
    float freduce_gpu "reduce<float>" (const float* src, Py_ssize_t num, int gpu) nogil except +
    long lreduce_gpu "reduce<long>" (const long* src, Py_ssize_t num, int gpu) nogil except +
    double dreduce_gpu "reduce<double>" (const double* src, Py_ssize_t num, int gpu) nogil except +

    int imax_gpu "reduceMax<int>" (const int* src, Py_ssize_t num, int gpu) nogil except +
    float fmax_gpu "reduceMax<float>" (const float* src, Py_ssize_t num, int gpu) nogil except +
    long lmax_gpu "reduceMax<long>" (const long* src, Py_ssize_t num, int gpu) nogil except +
    double dmax_gpu "reduceMax<double>" (const double* src, Py_ssize_t num, int gpu) nogil except +


cdef extern:
    struct int3:
        int x, y, z
    struct float3:
        float x, y, z

    void anidiffusion_gpu "anidiffusion" \
                     (const float* src, float* dst, const float lamda,
                      const int3 shape, const float gamma, const int mode,
                      const int maxIter, const float eps, int gpu) nogil except +
    void tvdenoising_gpu "tvdenoising" \
                    (const float* src, float* dst, float lamda,
                     float3 spacing, int3 shape, int maxIter, float eps,
                     int gpu) nogil except +

    void tvchambolle_gpu "tvchambolle" \
                    (const float* src, float* dst, float lamda,
                     float tau, int3 shape, int maxIter, float eps,
                     int gpu) nogil except +

    void tvbregman_gpu "tvbregman" \
                    (const float* src, float* dst, float lamda, float mu,
                     int3 shape, int maxIter, float eps, bool isotropic, int method,
                     int gpu) nogil except +

    void tvchambolle1st_gpu "tvchambolle1st" \
                    (const float* src, float* dst,
                     float lamda, float rho, float theta, float sigma, float gamma,
                     int3 shape, int maxIter, float eps, bool l2, int gpu) nogil except +


def greduce(data, int gpu=-1):
    if data.dtype == np.int32:
        return ireduce(data, gpu)
    elif data.dtype == np.int64:
        return lreduce(data, gpu)
    elif data.dtype == np.float32:
        return freduce(data, gpu)
    elif data.dtype == np.float64:
        return dreduce(data, gpu)
    else:
        raise Exception("Data type not supported");

cdef float freduce(float[::1] src, int gpu=-1):
    cdef size_t size = src.size
    cdef float result
    cdef float *src_ptr = <float*>&(src[0])
    with nogil:
        result = freduce_gpu(src_ptr, size, gpu)
    return result

cdef int ireduce(int[::1] src, int gpu=-1):
    cdef size_t size = src.size
    cdef int result
    cdef int *src_ptr = <int*>&(src[0])
    with nogil:
        result = ireduce_gpu(src_ptr, size, gpu)
    return result

cdef double dreduce(double[::1] src, int gpu=-1):
    cdef size_t size = src.size
    cdef double result
    cdef double *src_ptr = <double*>&(src[0])
    with nogil:
        result = dreduce_gpu(src_ptr, size, gpu)
    return result

cdef long lreduce(long[::1] src, int gpu=-1):
    cdef size_t size = src.size
    cdef long result
    cdef long *src_ptr = <long*>&(src[0])
    with nogil:
        result = lreduce_gpu(src_ptr, size, gpu)
    return result


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

cdef float fmax(float[::1] src, int gpu=-1):
    cdef size_t size = src.size
    cdef float result
    cdef float *src_ptr = <float*>&(src[0])
    with nogil:
        result = fmax_gpu(src_ptr, size, gpu)
    return result

cdef int imax(int[::1] src, int gpu=-1):
    cdef size_t size = src.size
    cdef int result
    cdef int *src_ptr = <int*>&(src[0])
    with nogil:
        result = imax_gpu(src_ptr, size, gpu)
    return result

cdef double dmax(double[::1] src, int gpu=-1):
    cdef size_t size = src.size
    cdef double result
    cdef double *src_ptr = <double*>&(src[0])
    with nogil:
        result = dmax_gpu(src_ptr, size, gpu)
    return result

cdef long lmax(long[::1] src, int gpu=-1):
    cdef size_t size = src.size
    cdef long result
    cdef long *src_ptr = <long*>&(src[0])
    with nogil:
        result = lmax_gpu(src_ptr, size, gpu)
    return result


def tvdenoising(float[:, :, ::1] src, float lamda=10., tuple spacing=(1.,1.,1.),
                int max_iter=100, float eps=1e-6, int gpu=-1):
    cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef float3 space = {'x': spacing[2], 'y': spacing[1], 'z': spacing[0]}

    cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
    cdef float[:, :, ::1] dst = np.zeros((D, H, W), np.float32)
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *dst_ptr = <float*>&(dst[0,0,0])

    with nogil:
        tvdenoising_gpu(src_ptr, dst_ptr, lamda, space, shape, max_iter, eps, gpu)

    return np.asarray(dst)


def tvchambolle(float[:, :, ::1] src, float lamda=10., float rho=1./8.,
                int max_iter=100, float eps=1e-6, int gpu=-1):
    cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
    cdef float[:, :, ::1] dst = np.zeros((D, H, W), np.float32)
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *dst_ptr = <float*>&(dst[0,0,0])

    with nogil:
        tvchambolle_gpu(src_ptr, dst_ptr, lamda, rho, shape, max_iter, eps, gpu)

    return np.asarray(dst)

def tvbregman(float[:, :, ::1] src, float lamda, float mu, int max_iter=100,
              float eps=1e-6, bool isotropic=True, int method=2, int gpu=-1):
    cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
    cdef float[:, :, ::1] dst = np.zeros((D, H, W), np.float32)
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *dst_ptr = <float*>&(dst[0,0,0])

    with nogil:
        tvbregman_gpu(src_ptr, dst_ptr, lamda, mu, shape, max_iter, eps, isotropic,
                      method, gpu)

    return np.asarray(dst)


def tvchambolle1st(float[:, :, ::1] src, float lamda=10., float rho=0.01,
                   float theta=1.0, float sigma=0.5, float gamma=7.,
                   int max_iter=100, float eps=1e-6, bool l2=True, int gpu=-1):
    cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
    cdef float[:, :, ::1] dst = np.zeros((D, H, W), np.float32)
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *dst_ptr = <float*>&(dst[0,0,0])

    tvchambolle1st_gpu(src_ptr, dst_ptr, lamda, rho, theta, sigma, gamma,
                       shape, max_iter, eps, l2, gpu)

    return np.asarray(dst)


def anidiffusion(float[:, :, ::1] src, float gamma=10., float lamda=1./8.,
                 int mode=3, int max_iter=100, float eps=1e-6, int gpu=-1):
    cdef int3 shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}

    cdef int D = src.shape[0], H = src.shape[1], W = src.shape[2]
    cdef float[:, :, ::1] dst = np.zeros((D, H, W), np.float32)
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *dst_ptr = <float*>&(dst[0,0,0])

    with nogil:
        anidiffusion_gpu(src_ptr, dst_ptr, lamda, shape, gamma, mode, max_iter, eps, gpu)

    return np.asarray(dst)
