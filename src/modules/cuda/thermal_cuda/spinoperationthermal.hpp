// i = 0: take 1st 3 number of 6
// i = 1: take 2nd 3 number of 6
void cuda_thermal(const float* d_rng6, const int i, 
	double temperature,
	double* d_hx, double* d_hy, double* d_hz, double* d_ms,
	double* d_scale,
	const int n,
	const double dt, const double alpha, const double* d_alpha, const double gamma, const double* d_gamma);
