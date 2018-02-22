

#ifndef __HESSIAN_EIG_CUDA__
#define __HESSIAN_EIG_CUDA__

#include "cuda.cuh"

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cfloat>

#ifdef WIN32
#define SURVOS_EXPORT __declspec(dllexport)
#else
#define SURVOS_EXPORT
#endif

void SURVOS_EXPORT symmetric3_eigenvalues(const float *Hzz, const float *Hzy, const float *Hzx,
                            const float *Hyy, const float *Hyx, const float *Hxx,
                            float* out, size_t total_size, bool doabs=false,
                            int gpu=-1);

							
#endif
