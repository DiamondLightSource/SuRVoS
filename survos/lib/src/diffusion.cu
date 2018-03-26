
#include "preprocess.cuh"


__global__
void anisodiff3d(const float* data, float* result, int type,
                 float gamma, float lambda,
                 long cols, long rows, long depth)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t size2d = rows * cols;

    if ( idx >= size2d * depth )
        return;

    size_t t = idx % size2d;
    size_t z = idx / size2d;
    size_t y = t / cols;
    size_t x = t % cols;

    long zs[] = {-1, 1, 0, 0, 0, 0};
    long ys[] = { 0, 0,-1, 1, 0, 0};
    long xs[] = { 0, 0, 0, 0,-1, 1};

    long idx2;
    float diff = 0.0f, tmp, tmp2;
    long z2, y2, x2;

    int count = 0;

    for ( int i = 0; i < sizeof(ys) / sizeof(long); i++ ) {
        z2 = z + zs[i];
        y2 = y + ys[i];
        x2 = x + xs[i];
        idx2 = z2 * size2d + y2 * cols + x2;

        if ( z2 < 0 || y2 < 0 || x2 < 0 || z2 >= depth || y2 >= rows || x2 >= cols )
            continue;

        tmp = data[idx2] - data[idx];
        if ( type == 1 ) {
            diff += tmp * expf(-(tmp*tmp) / (gamma*gamma));
        }
        else if ( type == 2 ) {
            diff += tmp * (1.f / (1.f + (tmp*tmp) / (gamma*gamma)));
        }
        else if ( type == 3 ) {
            if ( abs(tmp) <= gamma ) {
                tmp2 = (1.f - (tmp*tmp) / (gamma*gamma));
                diff += tmp * 0.5f * (tmp2*tmp2);
            }
        }
        else if ( type == 4 ) {
            if ( abs(tmp) <= gamma ) {
                diff += tmp / gamma;
            }
            else if ( abs(tmp) > gamma ) {
                diff += tmp * abs(1 / tmp);
            }
        }

        count += 1;
    }

    if ( count > 1 ) {
        result[idx] = data[idx] + lambda * diff / count;
    }
}


// Main function
void anidiffusion_gpu(const float* src, float* dst, const float lambda,
                  const int3 shape, const float gamma, const int mode,
                  const int maxIter, const float eps, int gpu)
{
    // Init params
    size_t total = shape.x * shape.y * shape.z;
    size_t mem_size = sizeof(float) * total;

    // Init cuda memory
    int max_threads = initCuda(gpu);

    float *d_1, *d_2;

    // F
    cudaCustomMalloc((void**)&d_1, mem_size, "d_1");
    cudaMemcpy(d_1, src, mem_size, cudaMemcpyHostToDevice);
    // U
    cudaCustomMalloc((void**)&d_2, mem_size, "d_2");
    cudaMemcpy(d_2, src, mem_size, cudaMemcpyHostToDevice);

    // bdim and gdim
    dim3 block(max_threads, 1, 1);
    dim3 grid((total+max_threads-1)/max_threads, 1, 1);

    int i;
    for ( i = 0; i < maxIter; i++ )
    {
        if ( i%2==0 ) {
            anisodiff3d<<<grid, block>>>(d_1, d_2, mode, gamma, lambda,
                                         shape.x, shape.y, shape.z);
        }
        else {
            anisodiff3d<<<grid, block>>>(d_2, d_1, mode, gamma, lambda,
                                         shape.x, shape.y, shape.z);
        }
    }

    cudaError_t error;
    if ( i%2==0 ) {
        error = cudaMemcpy(dst, d_2, mem_size, cudaMemcpyDeviceToHost);
    }
    else {
        error = cudaMemcpy(dst, d_1, mem_size, cudaMemcpyDeviceToHost);
    }

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_dest,d_dest) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaFree(d_1);
    cudaFree(d_2);
    cudaDeviceReset();
}
