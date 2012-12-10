void cuda_llg_cart_apply(const int nx, const int ny, const int nz,
	double* dsx, double* dsy, double* dsz, double* dms, //dest (spinto)
	double* ssx, double* ssy, double* ssz, double* sms, // src (spinfrom)
	double* ddx, double* ddy, double* ddz, double* dds, // dm/dt spins
	double* htx, double* hty, double* htz,              // dm/dt thermal fields
	double* dhx, double* dhy, double* dhz,              // dm/dt fields
	const double dt, const double alpha, const double* d_alpha, const double gamma, const double* d_gamma);
