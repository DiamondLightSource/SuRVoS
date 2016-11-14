import numpy as np
cimport numpy as np

cdef extern from "src/convolutions.cuh":
    struct int3:
        int x, y, z
    void convolution(const float *h_src, const float *h_kernel, float *h_dest,
                     const int3 im_shape, const int3 kernel_shape)
    void convolution_separable(const float *h_src, const float *h_kernelz,
                               const float *h_kernely, const float *h_kernelx,
                               float *h_dest, const int3 ishape, const int3 kshape)
    void convolution_separable_shared(const float *h_src, const float *h_kernelz,
                                      const float *h_kernely, const float *h_kernelx,
                                      float *h_dest, const int3 ishape, const int3 kshape)
    void n_convolution_separable_shared(const float *h_src, const float *h_kernels, float *h_dest,
                                        const int3 ishape, const int3 kshape, const int n_kernels)


def gconv(np.ndarray[float, ndim=3, mode='c'] src,
                    np.ndarray[float, ndim=3, mode='c'] kernel):
    shape = (src.shape[0] - kernel.shape[0] + 1, \
             src.shape[1] - kernel.shape[1] + 1, \
             src.shape[2] - kernel.shape[2] + 1)
    cdef np.ndarray[float, ndim=3, mode='c'] out = np.zeros(shape, np.float32)

    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 k_shape = {'x': kernel.shape[2], 'y': kernel.shape[1], 'z': kernel.shape[0]}

    convolution(<float*>np.PyArray_DATA(src), <float*>np.PyArray_DATA(kernel),
                <float*>np.PyArray_DATA(out), im_shape, k_shape)

    return out


def gconvs(np.ndarray[float, ndim=3, mode='c'] src,
           np.ndarray[float, ndim=1, mode='c'] kernelz,
           np.ndarray[float, ndim=1, mode='c'] kernely,
           np.ndarray[float, ndim=1, mode='c'] kernelx):
    shape = (src.shape[0] - kernelz.shape[0] + 1, \
             src.shape[1] - kernely.shape[0] + 1, \
             src.shape[2] - kernelx.shape[0] + 1)
    cdef np.ndarray[float, ndim=3, mode='c'] out = np.zeros(shape, np.float32)

    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 k_shape = {'x': kernelx.shape[0], 'y': kernely.shape[0], 'z': kernelz.shape[0]}

    convolution_separable(<float*>np.PyArray_DATA(src), <float*>np.PyArray_DATA(kernelz),
                          <float*>np.PyArray_DATA(kernely), <float*>np.PyArray_DATA(kernelx),
                          <float*>np.PyArray_DATA(out), im_shape, k_shape)

    return out


def gconvssh(np.ndarray[float, ndim=3, mode='c'] src,
             np.ndarray[float, ndim=1, mode='c'] kernelz,
             np.ndarray[float, ndim=1, mode='c'] kernely,
             np.ndarray[float, ndim=1, mode='c'] kernelx):
    shape = (src.shape[0] - kernelz.shape[0] + 1, \
             src.shape[1] - kernely.shape[0] + 1, \
             src.shape[2] - kernelx.shape[0] + 1)
    cdef np.ndarray[float, ndim=3, mode='c'] out = np.zeros(shape, np.float32)

    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 k_shape = {'x': kernelx.shape[0], 'y': kernely.shape[0], 'z': kernelz.shape[0]}

    convolution_separable_shared(<float*>np.PyArray_DATA(src), <float*>np.PyArray_DATA(kernelz),
                                 <float*>np.PyArray_DATA(kernely), <float*>np.PyArray_DATA(kernelx),
                                 <float*>np.PyArray_DATA(out), im_shape, k_shape)

    return out


def ngconvssh(np.ndarray[float, ndim=3, mode='c'] src,
              np.ndarray[float, ndim=1, mode='c'] kernels,
              kshape, int nkernels):
    shape = (src.shape[0] - kshape[0] + 1, \
             src.shape[1] - kshape[1] + 1, \
             src.shape[2] - kshape[2] + 1)
    cdef np.ndarray[float, ndim=3, mode='c'] out = np.zeros(shape, np.float32)

    cdef int3 im_shape = {'x': src.shape[2], 'y': src.shape[1], 'z': src.shape[0]}
    cdef int3 k_shape = {'x': kshape[2], 'y': kshape[1], 'z': kshape[0]}

    n_convolution_separable_shared(<float*>np.PyArray_DATA(src),
                                   <float*>np.PyArray_DATA(kernels),
                                   <float*>np.PyArray_DATA(out),
                                   im_shape, k_shape, nkernels)

    return out
