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
void SURVOS_EXPORT slicSupervoxels(const float *h_src, int *h_dest, const float compactness, \
                     const int3 im_shape, const int3 sp_shape, \
                     const int3 window, const float3 spacing, \
                     float min_size_ratio=0.5, float max_size_ratio=3, \
                     unsigned short max_iter=5, bool enforce_connectivity=true,
                     int gpu=-1);
