// http://www.ipol.im/pub/art/2013/61/article_lr.pdf

#include "preprocess.cuh"


__global__
void update_u(const float* f, const float* pz, const float* py, const float* px,
              float* u, float* err, float lambda, int3 shape)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = z * shape.y * shape.x + y * shape.x + x;

    if ( x >= shape.x || y >= shape.y || z >= shape.z )
        return;

#define UIDX(zz, yy, xx) ((zz) * shape.y * shape.x + (yy) * shape.x + (xx))

    float d = 0.0f;

    if ( z > 0 )                d -= pz[UIDX(z-1, y, x)];
    if ( y > 0 )                d -= py[UIDX(z, y-1, x)];
    if ( x > 0 )                d -= px[UIDX(z, y, x-1)];

    if ( z < shape.z - 1 )      d += pz[UIDX(z, y, x)];
    if ( y < shape.y - 1 )      d += py[UIDX(z, y, x)];
    if ( x < shape.x - 1 )      d += px[UIDX(z, y, x)];

#undef UIDX

    float fidx  = f[idx];
    float unext = fidx - 1.0f / lambda * d;
    float udiff = fidx - unext;

    u[idx] = unext;
    err[idx] = udiff * udiff;
}


__global__
void update_v(const float* f, const float* pz, const float* py, const float* px,
              float* v, float lambda, int3 shape)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = z * shape.y * shape.x + y * shape.x + x;

    if ( x >= shape.x || y >= shape.y || z >= shape.z )
        return;

#define UIDX(zz, yy, xx) ((zz) * shape.y * shape.x + (yy) * shape.x + (xx))

    float d = 0.0f;

    if ( z > 0 )                d -= pz[UIDX(z-1, y, x)];
    if ( y > 0 )                d -= py[UIDX(z, y-1, x)];
    if ( x > 0 )                d -= px[UIDX(z, y, x-1)];

    if ( z < shape.z - 1 )      d += pz[UIDX(z, y, x)];
    if ( y < shape.y - 1 )      d += py[UIDX(z, y, x)];
    if ( x < shape.x - 1 )      d += px[UIDX(z, y, x)];

#undef UIDX

    v[idx] = d - f[idx] * lambda;
}


__global__
void update_p(const float* v, float* pz, float* py, float* px,
              float* err, float rho, int3 shape)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = z * shape.y * shape.x + y * shape.x + x;

    if ( x >= shape.x || y >= shape.y || z >= shape.z )
        return;

    float gz = 0.0f, gy = 0.0f, gx = 0.0f;
    float vidx = v[idx];

#define VIDX(zz, yy, xx) ((zz) * shape.y * shape.x + (yy) * shape.x + (xx))

    if ( z < shape.z - 1 )      gz = v[VIDX(z+1, y, x)] - vidx;
    if ( y < shape.y - 1 )      gy = v[VIDX(z, y+1, x)] - vidx;
    if ( x < shape.x - 1 )      gx = v[VIDX(z, y, x+1)] - vidx;

#undef VIDX

    float tverr = sqrtf(gz * gz + gy * gy + gx * gx);
    float denom = 1.0f + rho * tverr;

    pz[idx] = (pz[idx] + rho * gz) / denom;
    py[idx] = (py[idx] + rho * gy) / denom;
    px[idx] = (px[idx] + rho * gx) / denom;

    err[idx] = tverr;
}


// Main function
void tvchambolle(const float* src, float* dst, float lambda, float rho,
                 int3 shape, int maxIter, float eps)
{
    // Init params
    size_t total = shape.x * shape.y * shape.z;
    size_t mem_size = sizeof(float) * total;

    // Init cuda memory
    initCuda();

    float *d_src, *d_errtv, *d_errl2, *d_px, *d_py, *d_pz, *d_u, *d_v;

    // F
    cudaMalloc(&d_src, mem_size);
    cudaMemcpy(d_src, src, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory Malloc and Memset: SRC");
    // U
    cudaMalloc(&d_u, mem_size);
    cudaMemcpy(d_u, src, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory Malloc and Memset: U");
    // U
    cudaMalloc(&d_v, mem_size);
    cudaMemset(d_v, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: V");
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
    // ERR TV
    cudaMalloc(&d_errtv, mem_size);
    cudaMemset(d_errtv, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: ERR TV");
    // ERR L2
    cudaMalloc(&d_errl2, mem_size);
    cudaMemset(d_errl2, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: ERR L2");

    // bdim and gdim
    dim3 block(10, 10, 10);
    dim3 grid((shape.x+block.x-1)/block.x,
              (shape.y+block.y-1)/block.y,
              (shape.z+block.z-1)/block.z);

    float error_tv, error_l2, eprev;

    printf("=======================\n");
    printf("Chambolle TV denoising\n");
    printf("=======================\n");
    printf("Image shape: (%d, %d, %d)\n", shape.z, shape.y, shape.x);
    printf("Lambda: %.5f, delta: %.5f\n", lambda, rho);
    printf("Block: (%d, %d, %d), Grid: (%d, %d, %d)\n\n", \
           block.z, block.y, block.x, grid.z, grid.y, grid.x);

    for ( int i = 0; i < maxIter; i++ )
    {
        update_u<<<grid, block>>>(d_src, d_pz, d_py, d_px, d_u, d_errl2, lambda, shape);
        update_v<<<grid, block>>>(d_src, d_pz, d_py, d_px, d_v, lambda, shape);
        update_p<<<grid, block>>>(d_v, d_pz, d_py, d_px, d_errtv, rho, shape);

        error_tv = reduce<float>(d_errtv, total) / total;
        error_l2 = reduce<float>(d_errl2, total) / total;

        printf("[%d] TVerr: %.10f # L2err: %.10f\n", i+1, error_tv, error_l2);

        if ( i == 0 ) {
            eprev = error_tv;
        } else {
            if ( eprev < error_tv ) {
                printf("Gradient diverged.\n");
                break;
            }

            if ( error_l2 > eps ) {
                printf("Converged.\n");
                break;
            }

            eprev = error_tv;
        }
    }

    cudaCheckErrors("TV Chambolle Minimization");

    cudaMemcpy(dst, d_u, mem_size, cudaMemcpyDeviceToHost);

    cudaCheckErrors("Copy result back");

    cudaFree(d_src);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_pz);
    cudaFree(d_py);
    cudaFree(d_px);
    cudaFree(d_errtv);
    cudaFree(d_errl2);
    cudaDeviceReset();
}