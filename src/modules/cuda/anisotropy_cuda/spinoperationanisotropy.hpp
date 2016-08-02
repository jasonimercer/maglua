void cuda_anisotropy_N(const double global_scale,
	const double** d_sx_N, const double** d_sy_N, const double** d_sz_N,
	const double* d_nx, const double* d_ny, const double* d_nz, const double* d_k,
	double** d_hx_N, double** d_hy_N, double** d_hz_N,
	const int nxyz, const int n);

void cuda_anisotropy_compressed_N(const double global_scale,
	const double** d_sx_N, const double** d_sy_N, const double** d_sz_N,
	const double* d_LUT, const char* d_idx,
	double** d_hx_N, double** d_hy_N, double** d_hz_N,
	const int nxyz, const int n);
