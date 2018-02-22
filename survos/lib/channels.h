

void __declspec(dllimport) symmetric3_eigenvalues(const float *Hzz, const float *Hzy, const float *Hzx,
                            const float *Hyy, const float *Hyx, const float *Hxx,
                            float* out, size_t total_size, bool doabs=false,
                            int gpu=-1);
							
