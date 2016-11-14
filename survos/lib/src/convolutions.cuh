
#ifndef __CONVOLUTIONS_CUDA__
#define __CONVOLUTIONS_CUDA__

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cfloat>

#include "cuda.cuh"

void convolution(const float *h_src, const float *h_kernel, float *h_dest,
                 const int3 im_shape, const int3 kernel_shape);

void convolution_separable(const float *h_src, const float *h_kernelz,
                           const float *h_kernely, const float *h_kernelx,
                           float *h_dest, const int3 ishape, const int3 kshape);

void convolution_separable_shared(const float *h_src, const float *h_kernelz,
                                  const float *h_kernely, const float *h_kernelx,
                                  float *h_dest, const int3 ishape, const int3 kshape);

void n_convolution_separable_shared(const float *h_src, const float *h_kernels, float *h_dest,
                                    const int3 ishape, const int3 kshape, const int n_kernels);

#endif