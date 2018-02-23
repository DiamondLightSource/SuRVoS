
#ifndef __CONVOLUTIONS_CUDA__
#define __CONVOLUTIONS_CUDA__

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cfloat>

#include "cuda.cuh"

#ifdef _WIN32
#define SURVOS_EXPORT __declspec(dllexport)
#else
#define SURVOS_EXPORT
#endif

void SURVOS_EXPORT convolution(const float *h_src, const float *h_kernel, float *h_dest,
                 const int3 im_shape, const int3 kernel_shape, int gpu=-1);

void SURVOS_EXPORT convolution_separable(const float *h_src, const float *h_kernelz,
                           const float *h_kernely, const float *h_kernelx,
                           float *h_dest, const int3 ishape, const int3 kshape,
                           int gpu=-1);

void SURVOS_EXPORT convolution_separable_shared(const float *h_src, const float *h_kernelz,
                                  const float *h_kernely, const float *h_kernelx,
                                  float *h_dest, const int3 ishape, const int3 kshape,
                                  int gpu=-1);

void SURVOS_EXPORT n_convolution_separable_shared(const float *h_src, const float *h_kernels, float *h_dest,
                                    const int3 ishape, const int3 kshape, const int n_kernels,
                                    int gpu=-1);

#endif
