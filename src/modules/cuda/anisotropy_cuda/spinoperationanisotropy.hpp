#include "spinsystem.hpp"

void cuda_anisotropy(const double global_scale,
	const double* d_sx, const double* d_sy, const double* d_sz,
	const double* d_nx, const double* d_ny, const double* d_nz, const double* d_k,
	double* d_hx, double* d_hy, double* d_hz,
	const int nx, const int ny, const int nz
					);

void cuda_anisotropy_compressed(const double global_scale,
	const double* d_sx, const double* d_sy, const double* d_sz,
	const double* d_LUT, const char* d_idx,
	double* d_hx, double* d_hy, double* d_hz,
	const int nxyz);
