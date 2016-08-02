#ifndef SO_CUDA_EX
#define SO_CUDA_EX
typedef struct ex_compressed_struct
{
	int offset;
	float strength;
} ex_compressed_struct;

void cuda_exchange32(
	const float* d_sx, const float* d_sy, const float* d_sz,
	const float* d_strength, const int* d_neighbour, const int max_neighbours,
	float* d_hx, float* d_hy, float* d_hz,
	const int nx, const int ny, const int nz);

void cuda_exchange_compressed32(
	const float* d_sx, const float* d_sy, const float* d_sz,
	const ex_compressed_struct* d_LUT, const unsigned char* d_idx, const int max_neighbours,
	float* d_hx, float* d_hy, float* d_hz, 
	const int nxyz);
#endif
