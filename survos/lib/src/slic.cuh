
#ifndef __SLIC_CUDA__
#define __SLIC_CUDA__

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



struct SLICClusterCenter
{
    float f, x, y, z;
};

// Function defines
void SURVOS_EXPORT slicSupervoxels(const float *h_src, int *h_dest, const float compactness, \
                     const int3 im_shape, const int3 sp_shape, \
                     const int3 window, const float3 spacing, \
                     float min_size_ratio=0.5, float max_size_ratio=3, \
                     unsigned short max_iter=5, bool enforce_connectivity=true,
                     int gpu=-1);

#endif __SLIC_CUDA__
