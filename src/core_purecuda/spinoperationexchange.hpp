void ex_d_makeStrengthArray(double** d_v, int nx, int ny, int nz, int max_neighbours);
void ex_d_freeStrengthArray(double* d_v);

void ex_d_makeNeighbourArray(int** h_v, int nx, int ny, int nz, int max_neighbours);
void ex_d_freeNeighbourArray(int* h_v);

void ex_h_makeStrengthArray(double** d_v, int nx, int ny, int nz, int max_neighbours);
void ex_h_freeStrengthArray(double* d_v);

void ex_h_makeNeighbourArray(int** h_v, int nx, int ny, int nz, int max_neighbours);
void ex_h_freeNeighbourArray(int* h_v);

void ex_hd_syncStrengthArray(double* d_v, double* h_v, int nx, int ny, int nz, int max_neighbours);
void ex_hd_syncNeighbourArray(int* d_v, int* h_v, int nx, int ny, int nz, int max_neighbours);


void cuda_exchange(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz,
	const int nx, const int ny, const int nz
					);
