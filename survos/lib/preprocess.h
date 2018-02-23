#ifdef WIN32
#define SURVOS_EXPORT __declspec(dllimport)
#else
#define SURVOS_EXPORT
#endif


struct int3{
	int x;
	int y;
	int z;
};
struct float3{
	float x;
	float y;
	float z;
};
template<typename T> T SURVOS_EXPORT reduce(const T *h_in, size_t num_items, int gpu=-1);
template<typename T> T SURVOS_EXPORT reduceMax(const T *h_in, size_t num_items, int gpu=-1);

void anidiffusion_gpu(const float* src, float* dst, const float lambda,
                  const int3 shape, const float gamma=1.0f/8.0f, const int mode=3,
                  const int maxIter=100, const float eps=1e-6, int gpu=-1);

void tvdenoising_gpu(const float* src, float* dst, float lambda,
                 float3 spacing, int3 shape, int maxIter=100, float eps=1e-6,
                 int gpu=-1);

void tvchambolle_gpu(const float* src, float* dst, float lambda, float rho,
                 int3 shape, int maxIter=100, float eps=1e-6, int gpu=-1);

void tvbregman_gpu(const float* src, float* dst, float lambda, float mu,
               int3 shape, int maxIter=100, float eps=1e-6, bool isotropic=true,
               int method=2, int gpu=-1);

void tvchambolle1st_gpu(const float* src, float* dst,
                    float lambda, float rho, float theta, float sigma, float gamma,
                    int3 shape, int maxIter=100, float eps=1e-6, bool l2=true,
int gpu=-1);

