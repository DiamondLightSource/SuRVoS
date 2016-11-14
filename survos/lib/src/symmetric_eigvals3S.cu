

#include "symmetric_eigvals3S.cuh"


__device__
float determinant3S(const float *A)
{
	return A[0*3+0] * (A[1*3+1]*A[2*3+2] - A[1*3+2]*A[2*3+1]) \
		 - A[0*3+1] * (A[1*3+0]*A[2*3+2] - A[1*3+2]*A[2*3+0]) \
		 + A[0*3+2] * (A[1*3+0]*A[2*3+1] - A[1*3+1]*A[2*3+0]);
}


// TODO: Improve by generating hessian matrix also in the kernel to reduce memory
__global__
void k_symmetric_eigvalues(const float *Hxx, const float *Hxy, const float *Hxz,
						   const float *Hyy, const float *Hyz, const float *Hzz,
						   float* out, size_t total)
{
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( idx >= total ) return;

	float A[9];
	A[0*3+0] = Hxx[idx]; A[0*3+1] = Hxy[idx]; A[0*3+2] = Hxz[idx];
	A[1*3+0] = Hxy[idx]; A[1*3+1] = Hyy[idx]; A[1*3+2] = Hyz[idx];
	A[2*3+0] = Hxz[idx]; A[2*3+1] = Hyz[idx]; A[2*3+2] = Hzz[idx];

	float B[9];
	float q, p, p2, r, phi, p1, invp;

	p1 = (A[0*3+1]*A[0*3+1]) + (A[0*3+2]*A[0*3+2]) + (A[1*3+2]*A[1*3+2]);

	if ( p1 == 0 ) {
		out[idx*3+0] = A[0*3+0];
		out[idx*3+1] = A[1*3+1];
		out[idx*3+2] = A[2*3+2];
	} else {
		q  = (A[0*3+0] + A[1*3+1] + A[2*3+2]) / 3.;
		p2 = (A[0*3+0] - q) * (A[0*3+0] - q) \
			 + (A[1*3+1] - q) * (A[1*3+1] - q) \
			 + (A[2*3+2] - q) * (A[2*3+2] - q) \
			 + 2 * p1;
		p  = sqrt(p2 / 6.);
		invp = 1. / p;

		for ( int i = 0; i < 3; i++ )
		{
			for ( int j = 0; j < 3; j++ )
			{
				if ( i == j ) {
					B[i*3+j] = invp * (A[i*3+j] - q);
				} else {
					B[i*3+j] = invp * A[i*3+j];
				}
			}
		}

		r = determinant3S(B) / 2.0f;

		if ( r <= -1.0f )
			phi = M_PI / 3.0f;
		else if ( r >= 1.0f )
			phi = 0.0f;
		else
			phi = acos(r) / 3.0f;

		float eigv[3];
		eigv[0] = q + 2.0f * p * cos(phi);
		eigv[2] = q + 2.0f * p * cos(phi + (2.0f * M_PI / 3.0f));
		eigv[1] = 3.0f * q - eigv[0] - eigv[2];

		out[idx*3+0] = eigv[0];
		out[idx*3+1] = eigv[1];
		out[idx*3+2] = eigv[2];
	}
}

__global__
void k_abs_symmetric_eigvalues(const float *Hxx, const float *Hxy, const float *Hxz,
							   const float *Hyy, const float *Hyz, const float *Hzz,
							   float* out, size_t total)
{
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( idx >= total ) return;

	float A[9];
	A[0*3+0] = Hxx[idx]; A[0*3+1] = Hxy[idx]; A[0*3+2] = Hxz[idx];
	A[1*3+0] = Hxy[idx]; A[1*3+1] = Hyy[idx]; A[1*3+2] = Hyz[idx];
	A[2*3+0] = Hxz[idx]; A[2*3+1] = Hyz[idx]; A[2*3+2] = Hzz[idx];

	float B[9];
	float q, p, p2, r, phi, p1, invp;

	p1 = (A[0*3+1]*A[0*3+1]) + (A[0*3+2]*A[0*3+2]) + (A[1*3+2]*A[1*3+2]);

	if ( p1 == 0 ) {
		out[idx*3+0] = A[0*3+0];
		out[idx*3+1] = A[1*3+1];
		out[idx*3+2] = A[2*3+2];
		return;
	}

	q  = (A[0*3+0] + A[1*3+1] + A[2*3+2]) / 3.;
	p2 = (A[0*3+0] - q) * (A[0*3+0] - q) \
		 + (A[1*3+1] - q) * (A[1*3+1] - q) \
		 + (A[2*3+2] - q) * (A[2*3+2] - q) \
		 + 2 * p1;
	p  = sqrt(p2 / 6.);
	invp = 1. / p;

	for ( int i = 0; i < 3; i++ )
	{
		for ( int j = 0; j < 3; j++ )
		{
			if ( i == j ) {
				B[i*3+j] = invp * (A[i*3+j] - q);
			} else {
				B[i*3+j] = invp * A[i*3+j];
			}
		}
	}

	r = determinant3S(B) / 2.0f;

	if ( r <= -1.0f )
		phi = M_PI / 3.0f;
	else if ( r >= 1.0f )
		phi = 0.0f;
	else
		phi = acos(r) / 3.0f;

	float eigv[3];
	eigv[0] = q + 2.0f * p * cos(phi);
	eigv[2] = q + 2.0f * p * cos(phi + (2.0f * M_PI / 3.0f));
	eigv[1] = 3.0f * q - eigv[0] - eigv[2];

	float aeigv[3], tmp;
	aeigv[0] = fabsf(eigv[0]);
	aeigv[1] = fabsf(eigv[1]);
	aeigv[2] = fabsf(eigv[2]);

	if ( aeigv[0] >= aeigv[1] && aeigv[0] > aeigv[2] ) {
		tmp = eigv[2]; eigv[2] = eigv[0]; eigv[0] = tmp;
		tmp = aeigv[2]; aeigv[2] = aeigv[0]; aeigv[0] = tmp;
	} else if ( aeigv[1] >= aeigv[0] && aeigv[1] >= aeigv[2] ) {
		tmp = eigv[2]; eigv[2] = eigv[1]; eigv[1] = tmp;
		tmp = aeigv[2]; aeigv[2] = aeigv[1]; aeigv[1] = tmp;
	}
	if ( aeigv[0] > aeigv[1] ) {
		tmp = eigv[1]; eigv[1] = eigv[0]; eigv[0] = tmp;
		tmp = aeigv[1]; aeigv[1] = aeigv[0]; aeigv[0] = tmp;
	}

	out[idx*3+0] = eigv[0];
	out[idx*3+1] = eigv[1];
	out[idx*3+2] = eigv[2];
}


void symmetric3_eigenvalues(const float *Hzz, const float *Hzy, const float *Hzx,
							const float *Hyy, const float *Hyx, const float *Hxx,
							float* out, size_t total_size, bool doabs)
{
	// Init cuda memory
	initCuda();

	float *d_hzz, *d_hzy, *d_hzx, *d_hyy, *d_hyx, *d_hxx, *d_out;
	size_t d_mem_size = total_size * sizeof(float);

	cudaMalloc((float **) &d_hzz, d_mem_size);
	cudaMemcpy(d_hzz, Hzz, d_mem_size, cudaMemcpyHostToDevice);
	cudaMalloc((float **) &d_hzy, d_mem_size);
	cudaMemcpy(d_hzy, Hzy, d_mem_size, cudaMemcpyHostToDevice);
	cudaMalloc((float **) &d_hzx, d_mem_size);
	cudaMemcpy(d_hzx, Hzx, d_mem_size, cudaMemcpyHostToDevice);
	cudaMalloc((float **) &d_hyy, d_mem_size);
	cudaMemcpy(d_hyy, Hyy, d_mem_size, cudaMemcpyHostToDevice);
	cudaMalloc((float **) &d_hyx, d_mem_size);
	cudaMemcpy(d_hyx, Hyx, d_mem_size, cudaMemcpyHostToDevice);
	cudaMalloc((float **) &d_hxx, d_mem_size);
	cudaMemcpy(d_hxx, Hxx, d_mem_size, cudaMemcpyHostToDevice);

	cudaMalloc((float **) &d_out, d_mem_size * 3);
	cudaCheckErrors("SRC & KERNEL & DST");

	// bdim and gdim
	dim3 threads(1024, 1, 1);
	dim3 grids((total_size + threads.x - 1) / threads.x, 1, 1);

	if ( doabs ) {
		k_abs_symmetric_eigvalues<<<grids, threads>>>(d_hzz, d_hzy, d_hzx,
													  d_hyy, d_hyx, d_hxx,
													  d_out, total_size);
	} else {
		k_symmetric_eigvalues<<<grids, threads>>>(d_hzz, d_hzy, d_hzx,
												  d_hyy, d_hyx, d_hxx,
												  d_out, total_size);
	}

	cudaCheckErrors("Eigenvalues");

	cudaMemcpy(out, d_out, d_mem_size * 3, cudaMemcpyDeviceToHost);
	cudaCheckErrors("Memcpy back");

	cudaFree(d_hzz);
	cudaFree(d_hzy);
	cudaFree(d_hzx);
	cudaFree(d_hyy);
	cudaFree(d_hyx);
	cudaFree(d_hxx);
	cudaFree(d_out);

	cudaCheckErrors("Free everything");

	cudaDeviceReset();
}
