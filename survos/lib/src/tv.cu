// http://gpu4vision.icg.tugraz.at/papers/2010/knoll.pdf#pub47

#include "preprocess.cuh"


__device__
float divergence(const float* pz, const float* py, const float* px,
                 long idx, long z, long y, long x,
                 long size2d, int3 shape, float3 spacing)
{
    float _div = 0.0f;
    long _idx;
    if ( z - 1 >= 0 ) {
        _idx = (z - 1) * size2d + y * shape.x + x;
        _div += (pz[idx] - pz[_idx]);
    } else {
        _div += pz[idx];
    }

    if ( y - 1 >= 0 ) {
        _idx = z * size2d + (y - 1) * shape.x + x;
        _div += (py[idx] - py[_idx]);
    } else {
        _div += py[idx];
    }

    if ( x - 1 >= 0 ) {
        _idx = z * size2d + y * shape.x + (x - 1);
        _div += (px[idx] - px[_idx]);
    } else {
        _div += px[idx];
    }

    return _div;
}

__device__
void gradient(const float* u, float* grad,
              long idx, long z, long y, long x,
              long size2d, int3 shape, float3 spacing)
{
    float uidx = u[idx];

    if ( z + 1 < shape.z ) {
        grad[0] = (u[(z+1)*size2d + y*shape.x + x] - uidx);
    }

    if ( y + 1 < shape.y ) {
        grad[1] = (u[z*size2d + (y+1)*shape.x + x] - uidx);
    }

    if ( x + 1 < shape.x ) {
        grad[2] = (u[z*size2d + y*shape.x + (x+1)] - uidx);
    }
}


__global__
void update_u(const float* f, const float* pz, const float* py, const float* px, float* u,
              float tau, float lambda, int3 shape, float3 spacing)
{
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    long t = idx % plane;
    long z = idx / plane;
    long y = t / shape.x;
    long x = t % shape.x;

    float _div = divergence(pz, py, px, idx, z, y, x, plane, shape, spacing);

    float r = u[idx] * (1.0f - tau) + tau * (f[idx] + lambda * _div);

    u[idx] = r;
}


__global__
void update_p(const float* u, float* pz, float* py, float* px,
              float tau, int3 shape, float3 spacing)
{
    long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    long t = idx % plane;
    long z = idx / plane;
    long y = t / shape.x;
    long x = t % shape.x;

    float grad[3], q[3];
    grad[0] = 0; grad[1] = 0; grad[2] = 0;
    gradient(u, grad, idx, z, y, x, plane, shape, spacing);

    q[0] = pz[idx] + tau * grad[0];
    q[1] = py[idx] + tau * grad[1];
    q[2] = px[idx] + tau * grad[2];

    float n = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];

    float norm = fmaxf(1.0f, sqrtf(fmaxf(0, n)));

    pz[idx] = q[0] / norm;
    py[idx] = q[1] / norm;
    px[idx] = q[2] / norm;
}


// Main function
void tvdenoising_gpu(const float* src, float* dst, float lambda,
                 float3 spacing, int3 shape, int maxIter, float eps,
                 int gpu)
{
    // Init params
    long total = shape.x * shape.y * shape.z;
    long mem_size = sizeof(float) * total;

    // Init cuda memory
    int max_threads = initCuda(gpu);

    float *d_src, *d_u, *d_px, *d_py, *d_pz;

    // F
    cudaMalloc(&d_src, mem_size);
    cudaMemset(d_src, 0, mem_size);
    cudaMemcpy(d_src, src, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory Malloc and Memset: SRC");
    // U
    cudaMalloc(&d_u, mem_size);
    cudaMemset(d_u, 0, mem_size);
    cudaMemcpy(d_u, src, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory Malloc and Memset: U");
    // PX
    cudaMalloc(&d_px, mem_size);
    cudaMemset(d_px, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: PX");
    // PY
    cudaMalloc(&d_py, mem_size);
    cudaMemset(d_py, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: PY");
    // PZ
    cudaMalloc(&d_pz, mem_size);
    cudaMemset(d_pz, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: PZ");

    // bdim and gdim
    dim3 block(max_threads, 1, 1);
    dim3 grid((total+max_threads-1)/max_threads, 1, 1);

    float tau2, tau1;
    for ( int i = 0; i < maxIter; i++ )
    {
        tau2 = 0.3f + 0.02f * i;
        tau1 = (1.f/tau2) * ((1.f/6.f) - (5.f/(15.f+i)));

        update_u<<<grid, block>>>(d_src, d_pz, d_py, d_px, d_u, tau1, 1.0f/lambda,
                                  shape, spacing);

        update_p<<<grid, block>>>(d_u, d_pz, d_py, d_px, tau2, shape, spacing);
    }

    cudaCheckErrors("TV minimization");

    cudaMemcpy(dst, d_u, mem_size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("Copy result back");

    cudaFree(d_src);
    cudaFree(d_u);
    cudaFree(d_pz);
    cudaFree(d_py);
    cudaFree(d_px);
    cudaDeviceReset();
}
