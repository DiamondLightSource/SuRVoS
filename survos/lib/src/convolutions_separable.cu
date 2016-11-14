
#include "convolutions.cuh"



__global__
void convolution_deep(const float* in, const float* kernel, float* out,
                      const int3 im_shape, const int3 k_shape)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x >= im_shape.x || y >= im_shape.y || z > im_shape.z - k_shape.z )
        return;

    float sum = 0;
    size_t k;
    size_t pz;

    for ( k = 0; k < k_shape.z; k++ )
    {
        pz = z + k;
        sum += in[pz * im_shape.y * im_shape.x + y * im_shape.x + x] * kernel[k];
    }

    out[(z+k_shape.z/2) * im_shape.y * im_shape.x + y * im_shape.x + x] = sum;
}

__global__
void convolution_rows(const float* in, const float* kernel, float* out,
                      const int3 im_shape, const int3 k_shape)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x >= im_shape.x || y > im_shape.y - k_shape.y || z >= im_shape.z )
        return;

    float sum = 0;
    size_t i;
    size_t py;

    for ( i = 0; i < k_shape.y; i++ )
    {
        py = y + i;
        sum += in[z * im_shape.y * im_shape.x + py * im_shape.x + x] * kernel[i];
    }

    out[z * im_shape.y * im_shape.x + (y+k_shape.y/2) * im_shape.x + x] = sum;
}

__global__
void convolution_cols(const float* in, const float* kernel, float* out,
                      const int3 im_shape, const int3 k_shape)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x > im_shape.x - k_shape.x || y >= im_shape.y || z >= im_shape.z )
        return;

    float sum = 0;
    size_t j;
    size_t px;

    for ( j = 0; j < k_shape.x; j++ )
    {
        px = x + j;
        sum += in[z * im_shape.y * im_shape.x + y * im_shape.x + px] * kernel[j];
    }

    out[z * im_shape.y * im_shape.x + y * im_shape.x + (x+k_shape.x/2)] = sum;
}

__global__
void clamp_result(const float* in, float* out,
                  const int3 im_shape, const int3 k_shape, const int3 r_shape)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x >= r_shape.x || y >= r_shape.y || z >= r_shape.z )
        return;

    size_t pz = z + k_shape.z/2;
    size_t py = y + k_shape.y/2;
    size_t px = x + k_shape.x/2;
    float result = in[pz * im_shape.y * im_shape.x + py * im_shape.x + px];
    out[z * r_shape.y * r_shape.x + y * r_shape.x + x] = result;
}





//*****************************************************************
// MAIN
//*****************************************************************
void convolution_separable(const float *h_src, const float *h_kernelz,
                           const float *h_kernely, const float *h_kernelx,
                           float *h_dest, const int3 ishape, const int3 kshape)
{
    // Init cuda memory
    initCuda();

    int3 total;
    total.x = ishape.x - kshape.x + 1;
    total.y = ishape.y - kshape.y + 1;
    total.z = ishape.z - kshape.z + 1;

    float *d_src1, *d_src2, *d_kernelz, *d_kernely, *d_kernelx, *d_dest;
    size_t d_mem_size = ishape.z * ishape.y * ishape.x * sizeof(float);
    size_t r_mem_size = total.z * total.y * total.x * sizeof(float);
    int3 k_mem_size;
    k_mem_size.z = kshape.z * sizeof(float);
    k_mem_size.y = kshape.y * sizeof(float);
    k_mem_size.x = kshape.x * sizeof(float);

    cudaMalloc((float **) &d_src1, d_mem_size);
    cudaMemcpy(d_src1, h_src, d_mem_size, cudaMemcpyHostToDevice);
    cudaMalloc((float **) &d_src2, d_mem_size);
    cudaMemcpy(d_src2, d_src1, d_mem_size, cudaMemcpyDeviceToDevice);

    cudaMalloc((float **) &d_kernelz, k_mem_size.z);
    cudaMemcpy(d_kernelz, h_kernelz, k_mem_size.z, cudaMemcpyHostToDevice);

    cudaMalloc((float **) &d_kernely, k_mem_size.y);
    cudaMemcpy(d_kernely, h_kernely, k_mem_size.y, cudaMemcpyHostToDevice);

    cudaMalloc((float **) &d_kernelx, k_mem_size.x);
    cudaMemcpy(d_kernelx, h_kernelx, k_mem_size.x, cudaMemcpyHostToDevice);

    cudaMalloc((float **) &d_dest, r_mem_size);
    cudaCheckErrors("SRC & KERNEL & DST");

    // bdim and gdim
    dim3 threads;
    threads.x = total.x < 13? total.x : 13;
    threads.y = total.y < 13? total.y : 13;
    threads.z = total.z < 6? total.z : 6;

    dim3 gridz((ishape.x + threads.x - 1) / threads.x, \
               (ishape.y + threads.y - 1) / threads.y, \
               (total.z + threads.z - 1) / threads.z);
    dim3 gridy((ishape.x + threads.x - 1) / threads.x, \
               (total.y + threads.y - 1) / threads.y, \
               (ishape.z + threads.z - 1) / threads.z);
    dim3 gridx((total.x + threads.x - 1) / threads.x, \
               (ishape.y + threads.y - 1) / threads.y, \
               (ishape.z + threads.z - 1) / threads.z);
    dim3 grida((total.x + threads.x - 1) / threads.x, \
               (total.y + threads.y - 1) / threads.y, \
               (total.z + threads.z - 1) / threads.z);

    convolution_deep<<<gridz, threads>>>(d_src1, d_kernelz, d_src2, ishape, kshape);
    convolution_rows<<<gridy, threads>>>(d_src2, d_kernely, d_src1, ishape, kshape);
    convolution_cols<<<gridx, threads>>>(d_src1, d_kernelx, d_src2, ishape, kshape);
    cudaCheckErrors("Convolution");

    clamp_result<<<gridx, threads>>>(d_src2, d_dest, ishape, kshape, total);

    cudaMemcpy(h_dest, d_dest, r_mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy back");

    cudaFree(d_src1);
    cudaFree(d_src2);
    cudaFree(d_kernelz);
    cudaFree(d_kernely);
    cudaFree(d_kernelx);
    cudaFree(d_dest);

    cudaCheckErrors("Free everything");

    cudaDeviceReset();
}