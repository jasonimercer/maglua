// 
// This header is the interface to long range field
// cuda calculations. 
// 

typedef struct JM_LONGRANGE_PLAN JM_LONGRANGE_PLAN;

JM_LONGRANGE_PLAN* make_JM_LONGRANGE_PLAN(int N_x, int N_y, int N_z, 
	double* GammaXX, double* GammaXY, double* GammaXZ,
	                 double* GammaYY, double* GammaYZ,
	                                  double* GammaZZ);

void free_JM_LONGRANGE_PLAN(JM_LONGRANGE_PLAN* p);

void JM_LONGRANGE(JM_LONGRANGE_PLAN* p, 
				  double* sx, double* sy, double* sz,
				  double* hx, double* hy, double* hz);
