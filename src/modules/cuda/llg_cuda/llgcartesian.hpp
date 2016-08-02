// void cuda_llg_cart_apply(const int nx, const int ny, const int nz,
// 	double* dsx, double* dsy, double* dsz, double* dms, //dest (spinto)
// 	double* ssx, double* ssy, double* ssz, double* sms, // src (spinfrom)
// 	double* ddx, double* ddy, double* ddz, double* dds, // dm/dt spins
// 	double* htx, double* hty, double* htz,              // dm/dt thermal fields
// 	double* dhx, double* dhy, double* dhz,              // dm/dt fields
// 	const double dt, const double alpha, const double* d_alpha, const double gamma, const double* d_gamma,
// 	int thermalOnlyFirstTerm, int disableRenormalization
// );
// 
// 
// void cuda_llg_cart_apply_N(const int nx, const int ny, const int nz,
// 	double** dsx, double** dsy, double** dsz, double** dms, //dest (spinto)
// 	double** ssx, double** ssy, double** ssz, double** sms, // src (spinfrom)
// 	double** ddx, double** ddy, double** ddz, double** dds, // dm/dt spins
// 	double** htx, double** hty, double* htz,              // dm/dt thermal fields
// 	double** dhx, double** dhy, double** dhz,              // dm/dt fields
// 	double* dt, double** d_alpha_N, double* d_alpha, double** d_gamma_N, double* d_gamma,
// 	int thermalOnlyFirstTerm, int disableRenormalization, const int n
// );


// 	cuda_llg_cart_apply_N(nx, ny, nz,
// 			    d_spinto_x_N,   d_spinto_y_N,   d_spinto_z_N,   d_spinto_m_N,
// 			  d_spinfrom_x_N, d_spinfrom_y_N, d_spinfrom_z_N, d_spinfrom_m_N,
// 			      d_dmdt_x_N,     d_dmdt_y_N,     d_dmdt_z_N,     d_dmdt_m_N,
// 			      d_dmdt_hT_x_N,     d_dmdt_hT_y_N,     d_dmdt_hT_z_N,
// 			      d_dmdt_hS_x_N,     d_dmdt_hS_y_N,     d_dmdt_hS_z_N,
// 				d_dt, d_alpha_N, d_alpha, d_gamma, d_gamma_N, 
// 				thermalOnlyFirstTerm, disableRenormalization, n);


void cuda_llg_cart_apply_N(int nx, int ny, int nz,
	double** dsx, double** dsy, double** dsz, double** dms, //dest (spinto)
	double** ssx, double** ssy, double** ssz, double** sms, // src (spinfrom)
	double** ddx, double** ddy, double** ddz, double** dds, // dm/dt spins
	double** htx, double** hty, double** htz,              // dm/dt thermal fields
	double** dhx, double** dhy, double** dhz,              // dm/dt fields
	double* dt, double** d_alpha_N, double* d_alpha, double** d_gamma_N, double* d_gamma,
	int thermalOnlyFirstTerm, int disableRenormalization, const int n);

