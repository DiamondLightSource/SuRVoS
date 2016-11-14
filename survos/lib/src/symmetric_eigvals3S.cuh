

#ifndef __HESSIAN_EIG_CUDA__
#define __HESSIAN_EIG_CUDA__

#include "cuda.cuh"

#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <cfloat>

void symmetric3_eigenvalues(const float *Hzz, const float *Hzy, const float *Hzx,
							const float *Hyy, const float *Hyx, const float *Hxx,
							float* out, size_t total_size, bool doabs=false);

#endif
