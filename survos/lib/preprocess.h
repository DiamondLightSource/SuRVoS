
template<typename T> T reduce(const T* h_in, size_t num_items, int gpu);
template<typename T> T reduceMax(const T* h_in, size_t num_items, int gpu);


void __declspec(dllimport) symmetric3_eigenvalues(const float *Hzz, const float *Hzy, const float *Hzx,
                            const float *Hyy, const float *Hyx, const float *Hxx,
                            float* out, size_t total_size, bool doabs=false,
                            int gpu=-1);
							
