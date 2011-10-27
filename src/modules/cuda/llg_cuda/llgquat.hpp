void cuda_llg_quat_apply(const int nx, const int ny, const int nz,
	double* dsx, double* dsy, double* dsz, double* dms, //dest
	double* ssx, double* ssy, double* ssz, double* sms, // src
	double* hx, double* hy, double* hz,
	double* ws1, double* ws2, double* ws3, double* ws4,
	const double alpha, const double dt, const double gamma);
