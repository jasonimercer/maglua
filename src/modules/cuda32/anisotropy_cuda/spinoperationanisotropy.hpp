#include "spinsystem.hpp"

void cuda_anisotropy32(const float global_scale,
	const float* d_sx, const float* d_sy, const float* d_sz,
	const float* d_nx, const float* d_ny, const float* d_nz, const float* d_k,
	float* d_hx, float* d_hy, float* d_hz,
	const int nx, const int ny, const int nz
					);

void cuda_anisotropy_compressed32(const float global_scale,
	const float* d_sx, const float* d_sy, const float* d_sz,
	const float* d_LUT, const char* d_idx,
	float* d_hx, float* d_hy, float* d_hz,
	const int nxyz);
