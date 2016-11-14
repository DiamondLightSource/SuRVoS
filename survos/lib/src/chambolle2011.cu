// A first-order primal-dual algorithm for convex problems with applications to imaging
// Chambolle, A. and Pock, T.

#include "preprocess.cuh"


__global__
void update_ul2(const float* f, float* u, float* up,
              const float* pz, const float* py, const float* px, float *err,
              float rho, float lambda, float theta, int3 shape)
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

    float uprev = u[idx];
    float fidx = f[idx];
    float beta = rho * lambda;
    float utmp = uprev + rho * d;

    float unext = (utmp + beta * fidx) / (1.0f + beta);

    u[idx] = unext;
    up[idx] = unext + theta * (unext - uprev);

    float diff = unext - fidx;
    err[idx] = diff * diff;
}

__global__
void update_ul1(const float* f, float* u, float* up,
              const float* pz, const float* py, const float* px, float *err,
              float rho, float lambda, float theta, int3 shape)
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

    float uprev = u[idx];
    float fidx = f[idx];
    float beta = rho * lambda;
    float utmp = uprev + rho * d;
    float unext = utmp - fidx;

    if ( unext > beta ) {
        unext = utmp - beta;
    } else if ( unext < -beta ) {
        unext = utmp + beta;
    } else {
        unext = fidx;
    }

    u[idx] = unext;
    up[idx] = unext + theta * (unext - uprev);

    float diff = unext - fidx;
    err[idx] = diff * diff;
}


__global__
void update_pv(const float* up, float* pz, float* py, float* px, float* err,
              float sigma, int3 shape)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = z * shape.y * shape.x + y * shape.x + x;

    if ( x >= shape.x || y >= shape.y || z >= shape.z )
        return;

    float gz = 0.0f, gy = 0.0f, gx = 0.0f;
    float uidx = up[idx];

#define UIDX(zz, yy, xx) up[(zz) * shape.y * shape.x + (yy) * shape.x + (xx)]

    if ( z < shape.z - 1 )      gz = UIDX(z+1, y, x) - uidx;
    if ( y < shape.y - 1 )      gy = UIDX(z, y+1, x) - uidx;
    if ( x < shape.x - 1 )      gx = UIDX(z, y, x+1) - uidx;

#undef UIDX

    err[idx] = sqrtf(gz * gz + gy * gy + gx * gx);

    float _pz = pz[idx] + sigma * gz;
    float _py = py[idx] + sigma * gy;
    float _px = px[idx] + sigma * gx;
    float norm = fmaxf(1.0f, sqrtf(_pz*_pz + _py*_py + _px*_px));

    pz[idx] = _pz / norm;
    py[idx] = _py / norm;
    px[idx] = _px / norm;
}


// Main function
void tvchambolle1st(const float* src, float* dst,
                    float lambda, float rho, float theta, float sigma, float gamma,
                    int3 shape, int maxIter, float eps, bool l2)
{
    // Init params
    size_t total = shape.x * shape.y * shape.z;
    size_t mem_size = sizeof(float) * total;

    // Init cuda memory
    initCuda();

    float *d_src, *d_px, *d_py, *d_pz, *d_u, *d_up, *d_errtv, *d_errl2;

    // F
    cudaMalloc(&d_src, mem_size);
    cudaMemcpy(d_src, src, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory Malloc and Memset: SRC");
    // U
    cudaMalloc(&d_u, mem_size);
    cudaMemcpy(d_u, d_src, mem_size, cudaMemcpyDeviceToDevice);
    cudaCheckErrors("Memory Malloc and Memset: U");
    // UP
    cudaMalloc(&d_up, mem_size);
    cudaMemcpy(d_up, d_src, mem_size, cudaMemcpyDeviceToDevice);
    cudaCheckErrors("Memory Malloc and Memset: UP");
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
    printf("Image shape: (%d, %d, %d), l2: %d\n", shape.z, shape.y, shape.x, l2);
    printf("Lambda: %.5f, rho: %.5f, sigma: %.5f, theta: %.5f\n", lambda, rho, sigma, theta);
    printf("Block: (%d, %d, %d), Grid: (%d, %d, %d)\n\n", \
           block.z, block.y, block.x, grid.z, grid.y, grid.x);

    for ( int i = 0; i < maxIter; i++ )
    {
        update_pv<<<grid, block>>>(d_up, d_pz, d_py, d_px, d_errtv, sigma, shape);

        theta = 1.0f / (sqrtf(1.0f + 2.0f * gamma * rho));

        if ( l2 ) {
            update_ul2<<<grid, block>>>(d_src, d_u, d_up, d_pz, d_py, d_px, d_errl2,
                                        rho, lambda, theta, shape);
        } else {
            update_ul1<<<grid, block>>>(d_src, d_u, d_up, d_pz, d_py, d_px, d_errl2,
                                        rho, lambda, theta, shape);
        }

        rho = theta * rho;
        sigma = sigma / theta;

        error_tv = reduce<float>(d_errtv, total) / total;
        error_l2 = reduce<float>(d_errl2, total) / total;

        printf("[%d] TVerr: %.10f # L2err: %.10f\n", i+1, error_tv, error_l2);

        if ( i == 0 ) {
            eprev = error_tv;
        } else {
            if ( eprev < error_tv && l2 ) {
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
    cudaFree(d_up);
    cudaFree(d_pz);
    cudaFree(d_py);
    cudaFree(d_px);
    cudaFree(d_errl2);
    cudaFree(d_errtv);
    cudaDeviceReset();
}
