
#ifndef __INIT_CUDA__
#define __INIT_CUDA__

#include <stdio.h>
#include <cuda_runtime.h>

int initCuda(int devID=-1);
#ifdef _WIN32
#define SURVOS_EXPORT __declspec(dllexport)
#else
#define SURVOS_EXPORT
#endif

template<typename T> T SURVOS_EXPORT reduce(const T *h_in, size_t num_items, int gpu=-1);
template<typename T> T SURVOS_EXPORT reduceMax(const T *h_in, size_t num_items, int gpu=-1);

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            printf("[CUDA ERROR] %s: %s\n",msg,cudaGetErrorString(__err));\
            exit(EXIT_FAILURE);\
        } \
    } while (0)

void inline cudaCustomMalloc(void** d_src, size_t mem_size, const char *error_str)
{
    cudaError_t error;
    error = cudaMalloc(d_src, mem_size);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc returned error code %d, line(%d)\nError Message: %s", error, __LINE__, error_str);
        exit(EXIT_FAILURE);
    }
}
#ifdef _WIN32
template int SURVOS_EXPORT reduce<int>(const int *h_in, size_t num_items, int gpu=-1);
template int SURVOS_EXPORT reduceMax<int>(const int *h_in, size_t num_items, int gpu=-1);
template float SURVOS_EXPORT reduce<float>(const float *h_in, size_t num_items, int gpu=-1);
template float SURVOS_EXPORT reduceMax<float>(const float *h_in, size_t num_items, int gpu=-1);
template long SURVOS_EXPORT reduce<long>(const long *h_in, size_t num_items, int gpu=-1);
template long SURVOS_EXPORT reduceMax<long>(const long *h_in, size_t num_items, int gpu=-1);
template double SURVOS_EXPORT reduce<double>(const double *h_in, size_t num_items, int gpu=-1);
template double SURVOS_EXPORT reduceMax<double>(const double *h_in, size_t num_items, int gpu=-1);
#endif
#endif
