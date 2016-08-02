// 
// This header is the interface to short range field
// cuda calculations. 
// 

	
void JM_SHORTRANGE(const int nx, const int ny, const int nz, 
				   const float global_scale,
				   int* ABCount, int** d_ABoffset, float** d_ABvalue,
				   const float* sx, const float* sy, const float* sz,
				   float* hx, float* hy, float* hz);
