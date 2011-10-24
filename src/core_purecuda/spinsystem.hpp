/* ss = spin system
 *  d = device
 *  h = host
 */

/* v is a address of the pointer to the array */
void ss_d_make3DArray(double** d_v, int nx, int ny, int nz);
void ss_d_free3DArray(double* d_v);

void ss_h_make3DArray(double** h_v, int nx, int ny, int nz);
void ss_h_free3DArray(double* h_v);

void ss_copyDeviceToHost_(double* dest, double* src, int nxyz, const char* file, unsigned int line);
#define ss_copyDeviceToHost(d,s,n) ss_copyDeviceToHost_(d, s, n, __FILE__, __LINE__)

void ss_copyHostToDevice_(double* dest, double* src, int nxyz, const char* file, unsigned int line);
#define ss_copyHostToDevice(d,s,n) ss_copyHostToDevice_(d, s, n, __FILE__, __LINE__)

void ss_d_set3DArray_(double* d_v, int nx, int ny, int nz, double value, const char* file, const unsigned int line);
#define ss_d_set3DArray(d,x,y,z,v) ss_d_set3DArray_(d,x,y,z,v,__FILE__, __LINE__)

void ss_d_add3DArray(double* d_dest, int nx, int ny, int nz, double* d_src1, double* d_src2);


double ss_reduce3DArray_sum(double* d_v, double* d_ws1, double* h_ws1, int nx, int ny, int nz);
