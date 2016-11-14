
#ifndef __INIT_CUDA__
#define __INIT_CUDA__

#include <stdio.h>
#include <cuda_runtime.h>

void initCuda();

template<typename T> T reduce(const T *h_in, size_t num_items);
template<typename T> T reduceMax(const T *h_in, size_t num_items);


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

#endif