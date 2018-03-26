
#ifndef __PREPROCESS_CUDA__
#define __PREPROCESS_CUDA__

#include "cuda.cuh"

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cfloat>

#ifdef _WIN32
#define SURVOS_EXPORT __declspec(dllexport)
#else
#define SURVOS_EXPORT
#endif

#ifdef _WIN32
template int SURVOS_EXPORT reduce<int>(const int *h_in, size_t num_items, int gpu=-1);
template int SURVOS_EXPORT reduceMax<int>(const int *h_in, size_t num_items, int gpu=-1);
template float SURVOS_EXPORT reduce<float>(const float *h_in, size_t num_items, int gpu=-1);
template float SURVOS_EXPORT reduceMax<float>(const float *h_in, size_t num_items, int gpu=-1);
template long SURVOS_EXPORT reduce<long>(const long *h_in, size_t num_items, int gpu=-1);
template long SURVOS_EXPORT reduceMax<long>(const long *h_in, size_t num_items, int gpu=-1);
template double SURVOS_EXPORT reduce<double>(const double *h_in, size_t num_items, int gpu=-1);
template double SURVOS_EXPORT reduceMax<double>(const double *h_in, size_t num_items, int gpu=-1);
#else
template<typename T> T reduce(const T* h_in, size_t num_items, int gpu);
template<typename T> T reduceMax(const T* h_in, size_t num_items, int gpu);
#endif 
// Function defines
void SURVOS_EXPORT anidiffusion_gpu(const float* src, float* dst, const float lambda,
                  const int3 shape, const float gamma=1.0f/8.0f, const int mode=3,
                  const int maxIter=100, const float eps=1e-6, int gpu=-1);

void SURVOS_EXPORT tvdenoising_gpu(const float* src, float* dst, float lambda,
                 float3 spacing, int3 shape, int maxIter=100, float eps=1e-6,
                 int gpu=-1);

void SURVOS_EXPORT tvchambolle_gpu(const float* src, float* dst, float lambda, float rho,
                 int3 shape, int maxIter=100, float eps=1e-6, int gpu=-1);

void SURVOS_EXPORT tvbregman_gpu(const float* src, float* dst, float lambda, float mu,
               int3 shape, int maxIter=100, float eps=1e-6, bool isotropic=true,
               int method=2, int gpu=-1);

void SURVOS_EXPORT tvchambolle1st_gpu(const float* src, float* dst,
                    float lambda, float rho, float theta, float sigma, float gamma,
                    int3 shape, int maxIter=100, float eps=1e-6, bool ll2=true,
                    int gpu=-1);

#endif
