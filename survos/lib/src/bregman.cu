// ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf
// http://tag7.web.rice.edu/Split_Bregman.html
// Imp: http://www.ipol.im/pub/art/2012/g-tvd/article.pdf

#include "preprocess.cuh"


__global__ void
update_u(const float *f, float *u, float *err,
         const float* v, float lambda, float mu,
         const float *dz, const float *dy, const float *dx,
         const float *bz, const float *by, const float *bx,
         const float *b2, int3 shape)
{

    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    size_t t = idx % plane;
    size_t z = idx / plane;
    size_t y = t / shape.x;
    size_t x = t % shape.x;

    float uterm = 0.0f, vterm = 0.0f;
    int num = 0;

#define UIDX(zz, yy, xx) ((zz) * shape.y * shape.x + (yy) * shape.x + (xx))

    vterm = v[idx] - b2[idx];

    num++;
    if ( z == 0 ) {
        uterm += u[UIDX(z+1, y, x)];
        uterm -= dz[UIDX(z, y, x)];
        uterm += bz[UIDX(z, y, x)];
    } else if ( z == shape.z - 1 ) {
        uterm += u[UIDX(z-1, y, x)];
    } else {
        uterm += u[UIDX(z+1, y, x)];
        uterm += u[UIDX(z-1, y, x)];
        uterm += dz[UIDX(z-1, y, x)] - dz[UIDX(z, y, x)];
        uterm += bz[UIDX(z, y, x)] - bz[UIDX(z-1, y, x)];
        num++;
    }

    num++;
    if ( y == 0 ) {
        uterm += u[UIDX(z, y+1, x)];
        uterm -= dy[UIDX(z, y, x)];
        uterm += by[UIDX(z, y, x)];
    } else if ( y == shape.y - 1 ) {
        uterm += u[UIDX(z, y-1, x)];
    } else {
        uterm += u[UIDX(z, y+1, x)];
        uterm += u[UIDX(z, y-1, x)];
        uterm += dy[UIDX(z, y-1, x)] - dy[UIDX(z, y, x)];
        uterm += by[UIDX(z, y, x)] - by[UIDX(z, y-1, x)];
        num++;
    }

    num++;
    if ( x == 0 ) {
        uterm += u[UIDX(z, y, x+1)];
        uterm -= dx[UIDX(z, y, x)];
        uterm += bx[UIDX(z, y, x)];
    } else if ( x == shape.x - 1 ) {
        uterm += u[UIDX(z, y, x-1)];
    } else {
        uterm += u[UIDX(z, y, x+1)];
        uterm += u[UIDX(z, y, x-1)];
        uterm += dx[UIDX(z, y, x-1)] - dx[UIDX(z, y, x)];
        uterm += bx[UIDX(z, y, x)] - bx[UIDX(z, y, x-1)];
        num++;
    }

#undef UIDX

    // Update U
    float norm = mu + num * lambda;
    float uprev = u[idx];
    float uidx = (lambda * uterm + mu * vterm) / norm;
    u[idx] = uidx;

    float diff = uidx - f[idx];
    err[idx] = diff * diff;
}


__global__ void
update_d_ani(const float *u, float lambda,
             float *dz, float *dy, float *dx,
             float *bz, float *by, float *bx, float *err,
             int3 shape)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    size_t t = idx % plane;
    size_t z = idx / plane;
    size_t y = t / shape.x;
    size_t x = t % shape.x;

    float gz = 0.0f, gy = 0.0f, gx = 0.0f;

#define UIDX(zz, yy, xx) u[(zz) * shape.y * shape.x + (yy) * shape.x + (xx)]

    float uidx = UIDX(z, y, x);

    if ( z+1 < shape.z )
        gz = UIDX(z+1, y, x) - uidx;

    if ( y+1 < shape.y )
        gy = UIDX(z, y+1, x) - uidx;

    if ( x+1 < shape.x )
        gx = UIDX(z, y, x+1) - uidx;

#undef UIDX

    err[idx] = sqrtf(gz * gz + gy * gy + gx * gx);

    float flux = 1.0f/lambda;
    float _bz = bz[idx], _by = by[idx], _bx = bx[idx];
    float _dz = 0.0f, _dy = 0.0f, _dx = 0.0f;
    float sz, sy, sx;

    sz = gz + _bz;
    sy = gy + _by;
    sx = gx + _bx;

    _dz = copysignf(fmaxf(fabsf(sz) - flux, 0.0f), sz);
    _dy = copysignf(fmaxf(fabsf(sy) - flux, 0.0f), sy);
    _dx = copysignf(fmaxf(fabsf(sx) - flux, 0.0f), sx);

    dz[idx] = _dz;
    dy[idx] = _dy;
    dx[idx] = _dx;

    bz[idx] = _bz + gz - _dz;
    by[idx] = _by + gy - _dy;
    bx[idx] = _bx + gx - _dx;
}


__global__ void
update_v_l1(const float* f, const float* u, float* v, float* b2,
            float lambda, float mu, int3 shape)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    size_t t = idx % plane;
    size_t z = idx / plane;
    size_t y = t / shape.x;
    size_t x = t % shape.x;

    float fidx = f[idx];
    float uidx = u[idx];
    float bidx = b2[idx];
    float beta = lambda / mu;

    float s = uidx - fidx + b2[idx];
    float vnew = fidx + copysignf(fmaxf(fabsf(s) - beta, 0.0f), s);
    v[idx] = vnew;
    b2[idx] = bidx + uidx - vnew;
}

__global__ void
update_v_l2(const float* f, const float* u, float* v, float* b2,
            float lambda, float mu, int3 shape)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    size_t t = idx % plane;
    size_t z = idx / plane;
    size_t y = t / shape.x;
    size_t x = t % shape.x;

    v[idx] = f[idx];
    b2[idx] = 0;
}

__global__ void
update_v_poisson(const float* f, const float* u, float* v, float* b2,
                 float lambda, float mu, int3 shape)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t plane = shape.y * shape.x;

    if ( idx >= plane * shape.z )
        return;

    size_t t = idx % plane;
    size_t z = idx / plane;
    size_t y = t / shape.x;
    size_t x = t % shape.x;

    float fidx = f[idx];
    float uidx = u[idx];
    float bidx = b2[idx];
    float beta = lambda / mu;

    float s = (uidx - beta + b2[idx]) / 2.0f;
    float vnew = s + sqrtf(s*s + beta * fidx);
    v[idx] = vnew;
    b2[idx] = bidx + uidx - vnew;
}


// Main function
void tvbregman(const float* src, float* dst, float lambda, float mu,
               int3 shape, int maxIter, float eps, bool isotropic, int method,
               int gpu)
{
    // Init params
    size_t total = shape.x * shape.y * shape.z;
    size_t mem_size = sizeof(float) * total;

    // Init cuda memory
    int max_threads = initCuda(gpu);

    float *d_src, *d_u, *d_v, *d_errtv, *d_errl2, *d_dx, *d_dy, *d_dz, *d_bx, *d_by, *d_bz, *d_b2;

    // F
    cudaMalloc(&d_src, mem_size);
    cudaMemcpy(d_src, src, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory Malloc and Memset: SRC");
    // U
    cudaMalloc(&d_u, mem_size);
    cudaMemcpy(d_u, src, mem_size, cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory Malloc and Memset: U");
    // ERR TV
    cudaMalloc(&d_errtv, mem_size);
    cudaMemset(d_errtv, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: ERR TV");
    // ERR L2
    cudaMalloc(&d_errl2, mem_size);
    cudaMemset(d_errl2, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: ERR L2");
    // DX
    cudaMalloc(&d_dx, mem_size);
    cudaMemset(d_dx, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: DX");
    // DY
    cudaMalloc(&d_dy, mem_size);
    cudaMemset(d_dy, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: DY");
    // DZ
    cudaMalloc(&d_dz, mem_size);
    cudaMemset(d_dz, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: DZ");
    // BX
    cudaMalloc(&d_bx, mem_size);
    cudaMemset(d_bx, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: BX");
    // BY
    cudaMalloc(&d_by, mem_size);
    cudaMemset(d_by, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: BY");
    // BZ
    cudaMalloc(&d_bz, mem_size);
    cudaMemset(d_bz, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: BZ");
    // B2
    cudaMalloc(&d_b2, mem_size);
    cudaMemset(d_b2, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: B2");
    // V
    cudaMalloc(&d_v, mem_size);
    cudaMemset(d_v, 0, mem_size);
    cudaCheckErrors("Memory Malloc and Memset: V");

    // bdim and gdim
    dim3 block(max_threads, 1, 1);
    dim3 grid((total+max_threads-1)/max_threads, 1, 1);

    printf("=======================\n");
    printf("Bregman TV denoising\n");
    printf("=======================\n");
    printf("Image shape: (%d, %d, %d)\n", shape.z, shape.y, shape.x);
    printf("Lambda: %.5f, mu: %.5f, isotropic: %d\n", lambda, mu, isotropic);
    printf("Block: (%d, %d, %d), Grid: (%d, %d, %d), Method: %d\n\n", \
           block.z, block.y, block.x, grid.z, grid.y, grid.x, method);

    float error_tv, error_l2, eprev;

    int i = 0;

    for ( i = 0; i < maxIter; i++ )
    {
        update_d_ani<<<grid, block>>>(d_u, lambda, d_dz, d_dy, d_dx,
                                      d_bz, d_by, d_bx, d_errtv, shape);
        if ( method == 1 )
            update_v_l1<<<grid, block>>>(d_src, d_u, d_v, d_b2, lambda, mu, shape);
        else if ( method == 2 )
            update_v_l2<<<grid, block>>>(d_src, d_u, d_v, d_b2, lambda, mu, shape);
        else
            update_v_poisson<<<grid, block>>>(d_src, d_u, d_v, d_b2, lambda, mu, shape);

        update_u<<<grid, block>>>(d_src, d_u, d_errl2, d_v, lambda, mu,
                                  d_dz, d_dy, d_dx,
                                  d_bz, d_by, d_bx, d_b2, shape);

        error_tv = reduce<float>(d_errtv, total, gpu) / total;
        error_l2 = reduce<float>(d_errl2, total, gpu) / total;

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

    cudaCheckErrors("TV Bregman Minimization");

    cudaMemcpy(dst, d_u, mem_size, cudaMemcpyDeviceToHost);

    cudaCheckErrors("Copy result back");

    cudaFree(d_src);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_errtv);
    cudaFree(d_errl2);
    cudaFree(d_dz);
    cudaFree(d_dy);
    cudaFree(d_dx);
    cudaFree(d_bz);
    cudaFree(d_by);
    cudaFree(d_bx);
    cudaFree(d_b2);
    cudaDeviceReset();
}
