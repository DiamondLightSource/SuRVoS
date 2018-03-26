# distutils: language=c++
import numpy as np
cimport numpy as np

cdef extern from "convolutions.h":
    struct int3:
        int x, y, z
    void convolution(const float *h_src, const float *h_kernel, float *h_dest,
                     const int3 im_shape, const int3 kernel_shape, int gpu) nogil except +
    void convolution_separable(const float *h_src, const float *h_kernelz,
                               const float *h_kernely, const float *h_kernelx,
                               float *h_dest, const int3 ishape, const int3 kshape,
                               int gpu) nogil except +
    void convolution_separable_shared(const float *h_src, const float *h_kernelz,
                                      const float *h_kernely, const float *h_kernelx,
                                      float *h_dest, const int3 ishape, const int3 kshape,
                                      int gpu) nogil except +
    void n_convolution_separable_shared(const float *h_src, const float *h_kernels, float *h_dest,
                                        const int3 ishape, const int3 kshape, const int n_kernels,
                                        int gpu) nogil except +


def gconv(float[:, :, ::1] src, float[:, :, ::1] kernel, int gpu=-1):
    shape = (src.shape[0] - kernel.shape[0] + 1, \
             src.shape[1] - kernel.shape[1] + 1, \
             src.shape[2] - kernel.shape[2] + 1)
    cdef float[:, :, ::1] out = np.zeros(shape, np.float32)
    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 k_shape = {'x': kernel.shape[2], 'y': kernel.shape[1], 'z': kernel.shape[0]}
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *kernel_ptr = <float*>&(kernel[0,0,0])
    cdef float *out_ptr = <float*>&(out[0,0,0])

    with nogil:
        convolution(src_ptr, kernel_ptr, out_ptr, im_shape, k_shape, gpu)

    return np.asarray(out)


def gconvs(float[:, :, ::1] src, float[::1] kernelz, float[::1] kernely,
           float[::1] kernelx, int gpu=-1):
    shape = (src.shape[0] - kernelz.shape[0] + 1, \
             src.shape[1] - kernely.shape[0] + 1, \
             src.shape[2] - kernelx.shape[0] + 1)

    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 k_shape = {'x': kernelx.shape[0], 'y': kernely.shape[0], 'z': kernelz.shape[0]}

    cdef float[:, :, ::1] out = np.zeros(shape, np.float32)
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *kernelz_ptr = <float*>&(kernelz[0])
    cdef float *kernely_ptr = <float*>&(kernely[0])
    cdef float *kernelx_ptr = <float*>&(kernelx[0])
    cdef float *out_ptr = <float*>&(out[0,0,0])

    with nogil:
        convolution_separable(src_ptr, kernelz_ptr, kernely_ptr, kernelx_ptr,
                              out_ptr, im_shape, k_shape, gpu)

    return np.asarray(out)


def gconvssh(float[:, :, ::1] src, float[::1] kernelz, float[::1] kernely,
             float[::1] kernelx, int gpu=-1):
    shape = (src.shape[0] - kernelz.shape[0] + 1, \
             src.shape[1] - kernely.shape[0] + 1, \
             src.shape[2] - kernelx.shape[0] + 1)

    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 k_shape = {'x': kernelx.shape[0], 'y': kernely.shape[0], 'z': kernelz.shape[0]}

    cdef float[:, :, ::1] out = np.zeros(shape, np.float32)
    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *kernelz_ptr = <float*>&(kernelz[0])
    cdef float *kernely_ptr = <float*>&(kernely[0])
    cdef float *kernelx_ptr = <float*>&(kernelx[0])
    cdef float *out_ptr = <float*>&(out[0,0,0])

    with nogil:
        convolution_separable_shared(src_ptr, kernelz_ptr, kernely_ptr, kernelx_ptr,
                                     out_ptr, im_shape, k_shape, gpu)

    return np.asarray(out)


def ngconvssh(float[:, :, ::1] src, float[::1] kernels, tuple kshape,
              int nkernels, int gpu=-1):
    shape = (src.shape[0] - kshape[0] + 1, \
             src.shape[1] - kshape[1] + 1, \
             src.shape[2] - kshape[2] + 1)

    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 k_shape = {'x': kshape[2], 'y': kshape[1], 'z': kshape[0]}

    cdef float[:, :, ::1] out = np.zeros(shape, np.float32)

    cdef float *src_ptr = <float*>&(src[0,0,0])
    cdef float *kernels_ptr = <float*>&(kernels[0])
    cdef float *out_ptr = <float*>&(out[0,0,0])

    with nogil:
        n_convolution_separable_shared(src_ptr, kernels_ptr, out_ptr,
                                       im_shape, k_shape, nkernels, gpu)

    return out
