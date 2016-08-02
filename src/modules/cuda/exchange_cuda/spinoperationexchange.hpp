#ifndef SO_CUDA_EX
#define SO_CUDA_EX
typedef struct ex_compressed_struct
{
	int offset;
	double strength;
} ex_compressed_struct;

#if 0
void cuda_exchange(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz,
	const int nx, const int ny, const int nz);

void cuda_exchange_compressed(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const ex_compressed_struct* d_LUT, const unsigned char* d_idx, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz, 
	const int nxyz);
#endif

//  these functions operate a single exchange object on multiple spin systems
void cuda_exchange_N(
	const double** d_sx, const double** d_sy, const double** d_sz, const double** d_sm,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double** d_hx, double** d_hy, double** d_hz, const double global_scale,
	const int nx, const int ny, const int nz,
	const int n);

void cuda_exchange_compressed_N(
	const double** d_sx, const double** d_sy, const double** d_sz, const double** d_sm,
	const ex_compressed_struct* d_LUT, const unsigned char* d_idx, const int max_neighbours,
	double** d_hx, double** d_hy, double** d_hz, const double global_scale, 
	const int nxyz,
	const int n);


#endif
