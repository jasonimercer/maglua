void cuda_llg_cart_apply(const int nx, const int ny, const int nz,
	double* dsx, double* dsy, double* dsz, double* dms, //dest (spinto)
	double* ssx, double* ssy, double* ssz, double* sms, // src (spinfrom)
	double* ddx, double* ddy, double* ddz, double* dds, // dm/dt spins
	double* dhx, double* dhy, double* dhz,              // dm/dt fields
	const double alpha, const double dt, const double gamma);
