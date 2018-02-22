struct int3{
	int x;
	int y;
	int z;
};
void __declspec(dllimport) convolution(const float *h_src, const float *h_kernel, float *h_dest,
                 const int3 im_shape, const int3 kernel_shape, int gpu=-1);

void __declspec(dllimport) convolution_separable(const float *h_src, const float *h_kernelz,
                           const float *h_kernely, const float *h_kernelx,
                           float *h_dest, const int3 ishape, const int3 kshape,
                           int gpu=-1);

void __declspec(dllimport) convolution_separable_shared(const float *h_src, const float *h_kernelz,
                                  const float *h_kernely, const float *h_kernelx,
                                  float *h_dest, const int3 ishape, const int3 kshape,
                                  int gpu=-1);

void __declspec(dllimport) n_convolution_separable_shared(const float *h_src, const float *h_kernels, float *h_dest,
                                    const int3 ishape, const int3 kshape, const int n_kernels,
                                    int gpu=-1);