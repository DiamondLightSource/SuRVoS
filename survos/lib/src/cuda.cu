
#include "cuda.cuh"

#include <cub/cub.cuh>

using namespace cub;


bool CUDA_STARTED = false;
int CUDA_DEVICE = -1;
int CUDA_THREADS = -1;


__host__
int initCuda(int devID)
{
    cudaError_t error;
    if ( CUDA_STARTED == true ) {
        printf("[GPU] Reusing device (%d) with %d threads\n.", CUDA_DEVICE, CUDA_THREADS);
        error = cudaSetDevice(CUDA_DEVICE);
        if ( error != cudaSuccess ) {
            printf("[GPU] cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        } else {
            return CUDA_THREADS;
        }
    }

    int num_devices, max_multiprocessors = 0;
    cudaDeviceProp properties;

    if ( devID < 0 ) {
        cudaGetDeviceCount(&num_devices);
        if ( num_devices > 1 ) {
            for ( int device = 0; device < num_devices; device++ ) {
                cudaGetDeviceProperties(&properties, device);
                if ( max_multiprocessors < properties.multiProcessorCount ) {
                    max_multiprocessors = properties.multiProcessorCount;
                    devID = device;
                }
            }
        } else if ( num_devices == 1 ) {
            devID = 0;
        }
    }

    error = cudaSetDevice(devID);

    if ( error != cudaSuccess ) {
        printf("[GPU] cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    } else {
        printf("[GPU] Selected CUDA device: %d\n", devID);
    }

    error = cudaGetDeviceProperties(&properties, devID);

    if ( properties.computeMode == cudaComputeModeProhibited )
    {
        printf("[GPU] Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_FAILURE);
    }

    if ( error != cudaSuccess )
    {
        printf("[GPU] cudaGetDeviceProperties returned error code %d, line(%d)\n",
               error, __LINE__);
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("[GPU] Device %d: \"%s\" with compute capability %d.%d, %d threads pb.\n",
               devID, properties.name, properties.major, properties.minor,
               properties.maxThreadsPerBlock);
    }

    CUDA_STARTED = true;
    CUDA_DEVICE = devID;
    CUDA_THREADS = properties.maxThreadsPerBlock;
    return CUDA_THREADS;
}


template<typename T> __host__
T reduce(const T* h_in, size_t num_items, int gpu)
{
    initCuda(gpu);

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

template long reduce(const long *h_in, size_t num_items, int gpu);
template int reduce(const int *h_in, size_t num_items, int gpu);
template float reduce(const float *h_in, size_t num_items, int gpu);
template double reduce(const double *h_in, size_t num_items, int gpu);


template<typename T> __host__
T reduceMax(const T* h_in, size_t num_items, int gpu)
{
    initCuda(gpu);

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

template long reduceMax(const long *h_in, size_t num_items, int gpu);
template int reduceMax(const int *h_in, size_t num_items, int gpu);
template float reduceMax(const float *h_in, size_t num_items, int gpu);
template double reduceMax(const double *h_in, size_t num_items, int gpu);
