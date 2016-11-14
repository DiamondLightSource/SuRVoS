
#include "cuda.cuh"

#include <cub/cub.cuh>

using namespace cub;


bool CUDA_STARTED = false;


__host__ void initCuda()
{
    if ( CUDA_STARTED == true ) { return; }

    int devID = 0;
    cudaError_t error;
    cudaDeviceProp deviceProp;

    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    CUDA_STARTED = true;
}


template<typename T> __host__
T reduce(const T* h_in, size_t num_items)
{
    initCuda();

    CachingDeviceAllocator  g_allocator(true);

    T sum;

    // Allocate problem device arrays
    T *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));
    // Allocate device output array
    T *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * 1));

    // Request and allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    // Run
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

    // Copy the result to host
    CubDebugExit(cudaMemcpy(&sum, d_out, sizeof(T) * 1, cudaMemcpyDeviceToHost));

    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    return sum;
}

template long reduce(const long *h_in, size_t num_items);
template int reduce(const int *h_in, size_t num_items);
template float reduce(const float *h_in, size_t num_items);
template double reduce(const double *h_in, size_t num_items);


template<typename T> __host__
T reduceMax(const T* h_in, size_t num_items)
{
    initCuda();

    CachingDeviceAllocator  g_allocator(true);

    T tmax;

    // Allocate problem device arrays
    T *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));
    // Allocate device output array
    T *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * 1));

    // Request and allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    // Run
    CubDebugExit(DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

    // Copy the result to host
    CubDebugExit(cudaMemcpy(&tmax, d_out, sizeof(T) * 1, cudaMemcpyDeviceToHost));

    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    return tmax;
}

template long reduceMax(const long *h_in, size_t num_items);
template int reduceMax(const int *h_in, size_t num_items);
template float reduceMax(const float *h_in, size_t num_items);
template double reduceMax(const double *h_in, size_t num_items);