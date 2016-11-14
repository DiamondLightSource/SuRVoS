
#ifndef __SLIC_CUDA__
#define __SLIC_CUDA__

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cfloat>

#include "types.h"
#include "cuda.cuh"

#define DLIMIT FLT_MAX
#define i(a, b, c) ((c) * shape.y * shape.x + (b) * shape.x + (a))
#define max(a,b)  ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })


struct SLICClusterCenter
{
    dtype f, x, y, z;
};

// Function defines
void slicSupervoxels(const dtype *h_src, itype *h_dest, const dtype compactness, \
                     const itype3 im_shape, const itype3 sp_shape, \
                     const itype3 window, const dtype3 spacing, \
                     const dtype min_size_ratio=0.5, const dtype max_size_ratio=3, \
                     const uint16 max_iter=5, const bool enforce_connectivity=true);

#endif __SLIC_CUDA__