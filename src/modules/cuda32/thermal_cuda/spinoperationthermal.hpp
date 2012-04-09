#include "spinsystem.hpp"

// i = 0: take 1st 3 number of 6
// i = 1: take 2nd 3 number of 6
void cuda_thermal32(const float* d_rng6, const int i, 
	float alpha, float gamma, float dt, float temperature,
	float* d_hx, float* d_hy, float* d_hz, float* d_ms,
	float* d_scale,
	const int nx, const int ny, const int nz);
