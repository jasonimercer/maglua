// 
// This header is the interface to long range field
// cuda calculations. 
// 

typedef struct JM_LONGRANGE_PLAN JM_LONGRANGE_PLAN;

JM_LONGRANGE_PLAN* make_JM_LONGRANGE_PLAN(int N_x, int N_y, int N_z, 
	double* GammaXX, double* GammaXY, double* GammaXZ,
	                 double* GammaYY, double* GammaYZ,
	                                  double* GammaZZ, void* ws_d_A, void* ws_d_B);

void free_JM_LONGRANGE_PLAN(JM_LONGRANGE_PLAN* p);

void JM_LONGRANGE(JM_LONGRANGE_PLAN* p, 
				  const double* sx, const double* sy, const double* sz,
				  double* hx, double* hy, double* hz, void* ws_d_A, void* ws_d_B);

int JM_LONGRANGE_PLAN_ws_size(int nx, int ny, int nz);
