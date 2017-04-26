

#include "convolutions.cuh"


texture<float, 3> texSrc;
texture<float, 3> texK;
texture<float, 1> texZ, texY, texX;


__global__
void convolution_raw(float* out, const int3 im_shape, const int3 k_shape, const int3 r_shape)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x > im_shape.x - k_shape.x ||
         y > im_shape.y - k_shape.y ||
         z > im_shape.z - k_shape.z )
        return;

    float sum = 0;
    size_t i, j, k;
    size_t px, py, pz;

    for ( k = 0; k < k_shape.z; k++ )
    {
        pz = z + k;
        for ( i = 0; i < k_shape.y; i++ )
        {
            py = y + i;
            for ( j = 0; j < k_shape.x; j++ )
            {
                px = x + j;
                sum += tex3D(texSrc, px, py, pz) * \
                       tex3D(texK, j, i, k);
            }
        }
    }

    out[z * r_shape.y * r_shape.x + y * r_shape.x + x] = sum;
}

//*****************************************************************
// MAIN
//*****************************************************************
void convolution(const float *h_src, const float *h_kernel, float *h_dest,
                 const int3 im_shape, const int3 kernel_shape, int gpu)
{
    // Init cuda memory
    int max_threads = initCuda(gpu);
    int max_threads_dim = (int)(floor(pow(max_threads, 1./3.)));

    int3 total;
    total.x = im_shape.x - kernel_shape.x + 1;
    total.y = im_shape.y - kernel_shape.y + 1;
    total.z = im_shape.z - kernel_shape.z + 1;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
    cudaArray *a_src, *a_kernel;

    float *d_dest;
    //size_t d_mem_size = im_shape.z * im_shape.y * im_shape.x * sizeof(float);
    //size_t k_mem_size = kernel_shape.z * kernel_shape.y * kernel_shape.x * sizeof(float);
    size_t r_mem_size = total.z * total.y * total.x * sizeof(float);

    //cudaMalloc((float **) &d_src, d_mem_size);
    //cudaMemcpy(d_src, h_src, d_mem_size, cudaMemcpyHostToDevice);
    const cudaExtent extent = make_cudaExtent(im_shape.x, im_shape.y, im_shape.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&a_src, &channelDesc, extent);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_src, extent.width*sizeof(float), extent.width, extent.height);
    copyParams.dstArray = a_src;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    texSrc.normalized = false;
    texSrc.filterMode = cudaFilterModePoint;
    texSrc.addressMode[0] = cudaAddressModeClamp;
    texSrc.addressMode[1] = cudaAddressModeClamp;
    texSrc.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(texSrc, a_src, channelDesc);
    cudaCheckErrors("SRC to texture");

    //cudaMalloc((float **) &d_kernel, k_mem_size);
    //cudaMemcpy(d_kernel, h_kernel, k_mem_size, cudaMemcpyHostToDevice);
    const cudaExtent kextent = make_cudaExtent(kernel_shape.x, kernel_shape.y, kernel_shape.z);
    cudaChannelFormatDesc kchannelDesc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&a_kernel, &kchannelDesc, kextent);
    cudaMemcpy3DParms kcopyParams = {0};
    kcopyParams.srcPtr   = make_cudaPitchedPtr((void*)h_kernel, kextent.width*sizeof(float), kextent.width, kextent.height);
    kcopyParams.dstArray = a_kernel;
    kcopyParams.extent   = kextent;
    kcopyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&kcopyParams);

    texK.normalized = false;
    texK.filterMode = cudaFilterModePoint;
    texK.addressMode[0] = cudaAddressModeClamp;
    texK.addressMode[1] = cudaAddressModeClamp;
    texK.addressMode[2] = cudaAddressModeClamp;

    cudaBindTextureToArray(texK, a_kernel, kchannelDesc);
    cudaCheckErrors("Kernel to texture");

    cudaMalloc((float **) &d_dest, r_mem_size);
    cudaCheckErrors("SRC & KERNEL & DST");

    // bdim and gdim
    dim3 threads;
    threads.x = total.x < max_threads_dim? total.x : max_threads_dim;
    threads.y = total.y < max_threads_dim? total.y : max_threads_dim;
    threads.z = total.z < max_threads_dim? total.z : max_threads_dim;
    dim3 grid((total.x + threads.x - 1) / threads.x, \
              (total.y + threads.y - 1) / threads.y, \
              (total.z + threads.z - 1) / threads.z);

    convolution_raw<<<grid, threads>>>(d_dest, im_shape, kernel_shape, total);
    cudaCheckErrors("Convolution");

    cudaMemcpy(h_dest, d_dest, r_mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy back");

    cudaUnbindTexture(texSrc);
    cudaUnbindTexture(texK);

    cudaFreeArray(a_src);
    cudaFreeArray(a_kernel);
    cudaFree(d_dest);

    cudaCheckErrors("Free everything");

    cudaDeviceReset();
}
