
#include "slic.cuh"


__global__
void initSupervoxels(const dtype *data,
                     SLICClusterCenter* centers,
                     const itype tClusters,
                     const itype3 nClusters,
                     const itype3 sp_shape,
                     const itype3 padding,
                     const itype3 window,
                     const itype3 shape)
{
    size_t lidx  = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( lidx >= tClusters )
        return;

    itype3 idx;
    idx.x       = lidx % nClusters.x;
    itype r     = lidx / nClusters.x;
    idx.y       = r % nClusters.y;
    idx.z       = r / nClusters.y;

    itype x = idx.x * sp_shape.x + sp_shape.x / 2 + padding.x;
    itype y = idx.y * sp_shape.y + sp_shape.y / 2 + padding.y;
    itype z = idx.z * sp_shape.z + sp_shape.z / 2 + padding.z;

    itype u, v, w, cx = x, cy = y, cz = z, cux, cuy, cuz;
    dtype minGradient = DLIMIT, gradient, dx, dy, dz;

    for ( u = -window.x; u <= window.x; u++ )
    {
        for ( v = -window.y; v <= window.y; v++ )
        {
            for ( w = -window.z; w <= window.z; w++ )
            {
                cux = x+u; cuy = y+v; cuz = z+w;
                if ( cux < 1 || cux > shape.x-2 ||
                     cuy < 1 || cuy > shape.y-2 ||
                     cuz < 1 || cuz > shape.z-2 )
                {
                    continue;
                }

                dx = data[i(cux+1, cuy, cuz)] - data[i(cux-1, cuy, cuz)];
                dy = data[i(cux, cuy+1, cuz)] - data[i(cux, cuy-1, cuz)];
                dz = data[i(cux, cuy, cuz+1)] - data[i(cux, cuy, cuz-1)];

                gradient = dx*dx + dy*dy + dz*dz;

                if ( gradient < minGradient )
                {
                    minGradient = gradient;
                    cx = cux;
                    cy = cuy;
                    cz = cuz;
                }
            }
        }
    }

    centers[lidx].f = data[i(cx, cy, cz)];
    centers[lidx].x = cx;
    centers[lidx].y = cy;
    centers[lidx].z = cz;
}

__global__
void assignSupervoxels(const dtype *data,
                       const SLICClusterCenter* centers,
                       itype *labels,
                       const dtype compactness,
                       const itype tClusters,
                       const dtype3 spacing,
                       const itype3 imshape)
{
    itype3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t gidx = idx.z * (imshape.y * imshape.x) + idx.y * imshape.x + idx.x;

    extern __shared__ dtype shared[];
    SLICClusterCenter* _centers = (SLICClusterCenter*)shared;
    itype* _label = (itype*)&_centers[27];

    if ( threadIdx.z < 3 && threadIdx.y < 3 && threadIdx.x < 3 )
    {
        itype3 csp;
        csp.z = blockIdx.z + threadIdx.z - 1;
        csp.y = blockIdx.y + threadIdx.y - 1;
        csp.x = blockIdx.x + threadIdx.x - 1;
        itype lidx = threadIdx.z * 9 + threadIdx.y * 3 + threadIdx.x;

        if ( csp.z < 0 || csp.y < 0 || csp.x < 0 || csp.z >= gridDim.z || csp.y >= gridDim.y || csp.x >= gridDim.x )
        {
            _label[lidx] = -1;
        }
        else
        {
            itype spi = csp.z * (gridDim.y * gridDim.x) + csp.y * gridDim.x + csp.x;
            _centers[lidx] = centers[spi];
            _label[lidx] = spi;
        }
    }

    __syncthreads();

    if ( idx.x >= imshape.x || idx.y >= imshape.y || idx.z >= imshape.z )
        return;

    dtype minDist = DLIMIT;
    itype minIdx = 13;

    for ( int i = 0; i < 27; i++ )
    {
        if ( _label[i] == -1 ) {continue;}

        float dist_g = data[gidx] - _centers[i].f;
        float dx = (idx.x - _centers[i].x) * spacing.x;
        float dy = (idx.y - _centers[i].y) * spacing.y;
        float dz = (idx.z - _centers[i].z) * spacing.z;
        float dist_s = dx*dx + dy*dy + dz*dz;
        float dist = dist_g * dist_g + compactness * dist_s;

        if ( dist < minDist ) {
            minDist = dist;
            minIdx = i;
        }
    }

    labels[gidx] = _label[minIdx];
}

__global__
void updateSupervoxels(const dtype *data,
                       const itype *labels,
                       SLICClusterCenter* centers,
                       const itype tClusters,
                       const itype3 nClusters,
                       const itype3 spshape,
                       const itype3 imshape)
{
    size_t lidx  = threadIdx.x + (blockIdx.x * blockDim.x);

    if ( lidx >= tClusters )
        return;

    dtype cz = centers[lidx].z, cy = centers[lidx].y, cx = centers[lidx].x;

    if ( cz == -1 )
        return;

    itype kinit, kend, jinit, jend, iinit, iend;

    float ratio = 1.5f;
    kinit = cz - spshape.z * ratio;
    if ( kinit < 0 ) kinit = 0;
    kend = cz + spshape.z * ratio;
    if ( kend >= imshape.z ) kend = imshape.z - 1;

    iinit = cy - spshape.y * ratio;
    if ( iinit < 0 ) iinit = 0;
    iend = cy + spshape.y * ratio;
    if ( iend >= imshape.y ) iend = imshape.y - 1;

    jinit = cx - spshape.x * ratio;
    if ( jinit < 0 ) jinit = 0;
    jend = cx + spshape.x * ratio;
    if ( jend >= imshape.x ) jend = imshape.x - 1;

    dtype gray = 0, x = 0, y = 0, z = 0;
    itype count = 0;

    for ( itype k = kinit; k < kend; k++ )
    {
        for ( itype i = iinit; i < iend; i++ )
        {
            for ( itype j = jinit; j < jend; j++ )
            {
                itype offset = k * imshape.y * imshape.x + i * imshape.x + j;
                if ( labels[offset] == lidx ) {
                    x += j;
                    y += i;
                    z += k;
                    gray += data[offset];
                    count += 1;
                }
            }
        }
    }

    centers[lidx].f = gray / count;
    centers[lidx].x = x / count;
    centers[lidx].y = y / count;
    centers[lidx].z = z / count;

    if ( count == 0 )
        centers[lidx].z = -1;
        return;
}


const int dx6[6] = {-1,  0,  0,  1,  0,  0};
const int dy6[6] = { 0, -1,  0,  0,  1,  0};
const int dz6[6] = { 0,  0, -1,  0,  0,  1};

void FindNext(const itype* labels, itype* nlabels,
              const itype3 shape, const const itype lab,
              const itype w, const itype h, const itype d,
              itype* xvec, itype* yvec, itype* zvec,
              itype* count, const itype max_size)
{
    itype z, y, x, ind;
    itype oldlab = labels[d*shape.y*shape.x + h*shape.x + w];

    for ( int i = 0; i < 6; i++ )
    {
        z = d+dz6[i];
        y = h+dy6[i];
        x = w+dx6[i];

        if ( (z < shape.z && z >= 0) && (y < shape.y && y >= 0) &&
             (x < shape.x && x >= 0) )
        {
            ind = z*shape.y*shape.x + y*shape.x + x;
            if ( nlabels[ind] < 0 && labels[ind] == oldlab )
            {
                xvec[*count] = x;
                yvec[*count] = y;
                zvec[*count] = z;
                *count += 1;
                nlabels[ind] = lab;

                if ( *count < max_size - 1 ) {
                    FindNext(labels, nlabels, shape, lab, x, y, z,
                             xvec, yvec, zvec, count, max_size);
                }
            }
        }
    }
}

void enforceConnectivity(itype* labels, const itype3 shape,
                         const itype max_size, const itype min_size)
{
    itype size = shape.z * shape.y * shape.x;
    itype* nlabels = (itype*)malloc(size * sizeof(itype));
    memset(nlabels, -1, size * sizeof(itype));

    //------------------
    // labeling
    //------------------
    itype lab = 0;
    itype i = 0;
    itype adjlabel = 0; //adjacent label
    itype* xvec = (itype*)malloc(max_size * sizeof(itype)); //worst case size
    itype* yvec = (itype*)malloc(max_size * sizeof(itype)); //worst case size
    itype* zvec = (itype*)malloc(max_size * sizeof(itype)); //worst case size

    itype count = 0;

    for ( int d = 0; d < shape.z; d++ )
    {
        for ( int h = 0; h < shape.y; h++ )
        {
            for( int w = 0; w < shape.x; w++ )
            {
                itype idx = d*shape.y*shape.x + h*shape.x + w;

                if ( nlabels[idx] < 0 )
                {
                    nlabels[idx] = lab;
                    //-------------------------------------------------------
                    // Quickly find an adjacent label for use later if needed
                    //-------------------------------------------------------
                    for ( int n = 0; n < 6; n++ )
                    {
                        int x = w + dx6[n];
                        int y = h + dy6[n];
                        int z = d + dz6[n];

                        if( (x >= 0 && x < shape.x) && (y >= 0 && y < shape.y) &&
                            (z >= 0 && z < shape.z) )
                        {
                            int nindex = z*shape.y*shape.x + y*shape.x + x;
                            if ( nlabels[nindex] >= 0 ) {
                                adjlabel = nlabels[nindex];
                                break;
                            }
                        }
                    }
                    xvec[0] = w; yvec[0] = h; zvec[0] = d;
                    count = 1;
                    FindNext(labels, nlabels, shape, lab, w, h, d, xvec, yvec, zvec, &count, max_size);
                    //-------------------------------------------------------
                    // If segment size is less then a limit, assign an
                    // adjacent label found before, and decrement label count.
                    //-------------------------------------------------------
                    if ( count < min_size )
                    {
                        for( int c = 0; c < count; c++ )
                        {
                            int ind = zvec[c]*shape.y*shape.x + yvec[c]*shape.x + xvec[c];
                            nlabels[ind] = adjlabel;
                        }
                        lab--;
                    }
                    lab++;
                }
                i++;
            }
        }
    }
    //------------------
    //numlabels = lab;
    //------------------
    if ( xvec ) free(xvec);
    if ( yvec ) free(yvec);
    if ( zvec ) free(zvec);

    memcpy(labels, nlabels, size * sizeof(itype));

    if ( nlabels ) free(nlabels);
}


// Main function
void slicSupervoxels(const dtype *h_src, itype *h_dest, const dtype compactness, \
                     const itype3 im_shape, const itype3 sp_shape, \
                     const itype3 window, const dtype3 spacing, \
                     const dtype min_size_ratio, const dtype max_size_ratio, \
                     const uint16 max_iter, const bool enforce_connectivity)
{
    if ( sp_shape.x * sp_shape.y * sp_shape.z > 1024 ) {
        printf("SP_SHAPE must be <1024");
        exit(1);
    }

    // Init params
    size_t mem_size = sizeof(dtype) * im_shape.x * im_shape.y * im_shape.z;
    itype3 nsp = {(im_shape.x + sp_shape.x - 1) / sp_shape.x, \
                 (im_shape.y + sp_shape.y - 1) / sp_shape.y, \
                 (im_shape.z + sp_shape.z - 1) / sp_shape.z};
    itype3 shift = {(im_shape.x - nsp.x * sp_shape.x) / 2, \
                   (im_shape.y - nsp.y * sp_shape.y) / 2, \
                   (im_shape.z - nsp.z * sp_shape.z) / 2,};
    size_t total = nsp.x * nsp.y * nsp.z;

    dtype m = compactness / (dtype)(max(max(im_shape.x, im_shape.y), im_shape.z));
    m /= sqrt(im_shape.x * im_shape.y * im_shape.z / total);

    // Init cuda memory
    initCuda();

    dtype *d_src;
    itype *d_dest;
    SLICClusterCenter *d_centers;

    cudaMalloc((dtype **) &d_src, mem_size);
    cudaMemcpy(d_src, h_src, mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((itype **) &d_dest, mem_size);
    cudaMalloc((dtype **) &d_centers, sizeof(SLICClusterCenter) * total);
    cudaCheckErrors("SRC, DST & Centers malloc");

    // bdim and gdim
    dim3 threads(1024, 1, 1);
    dim3 grid((total + 1024 - 1) / 1024, 1, 1);

    dim3 threads2(sp_shape.x, sp_shape.y, sp_shape.z);
    dim3 grid2(nsp.x, nsp.y, nsp.z);

    initSupervoxels<<<grid, threads>>>(d_src, d_centers, total, nsp, sp_shape, \
                                       shift, window, im_shape);

    for ( int i = 0; i < max_iter; i++ )
    {
        assignSupervoxels<<<grid2, threads2, sizeof(dtype)*5*27>>>(d_src, d_centers, d_dest, m, total, spacing, im_shape);
        updateSupervoxels<<<grid, threads>>>(d_src, d_dest, d_centers, total, nsp, sp_shape, im_shape);
    }

    assignSupervoxels<<<grid2, threads2, sizeof(dtype)*5*27>>>(d_src, d_centers, d_dest, m, total, spacing, im_shape);

    cudaMemcpy(h_dest, d_dest, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy back");

    cudaFree(d_src);
    cudaFree(d_dest);
    cudaFree(d_centers);
    cudaDeviceReset();

    if ( enforce_connectivity ) {
        itype spsize = sp_shape.x * sp_shape.y * sp_shape.z;
        itype max_size = (itype)(spsize * max_size_ratio);
        itype min_size = (itype)(spsize * min_size_ratio);
        enforceConnectivity(h_dest, im_shape, max_size, min_size);
    }
}
