
#ifndef __PREPROCESS_CUDA__
#define __PREPROCESS_CUDA__

#include "cuda.cuh"

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cfloat>

template<typename T> T reduce(const T* h_in, size_t num_items, int gpu);
template<typename T> T reduceMax(const T* h_in, size_t num_items, int gpu);

// Function defines
void anidiffusion(const float* src, float* dst, const float lambda,
                  const int3 shape, const float gamma=1.0f/8.0f, const int mode=3,
                  const int maxIter=100, const float eps=1e-6, int gpu=-1);

void tvdenoising(const float* src, float* dst, float lambda,
                 float3 spacing, int3 shape, int maxIter=100, float eps=1e-6,
                 int gpu=-1);

void tvchambolle(const float* src, float* dst, float lambda, float rho,
                 int3 shape, int maxIter=100, float eps=1e-6, int gpu=-1);

void tvbregman(const float* src, float* dst, float lambda, float mu,
               int3 shape, int maxIter=100, float eps=1e-6, bool isotropic=true,
               int method=2, int gpu=-1);

void tvchambolle1st(const float* src, float* dst,
                    float lambda, float rho, float theta, float sigma, float gamma,
                    int3 shape, int maxIter=100, float eps=1e-6, bool l2=true,
                    int gpu=-1);

#endif
