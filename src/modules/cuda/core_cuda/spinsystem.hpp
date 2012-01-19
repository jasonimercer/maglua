/* ss = spin system
 *  d = device
 *  h = host
 */

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef CORECUDA_EXPORTS
  #define CORECUDA_API __declspec(dllexport)
 #else
  #define CORECUDA_API __declspec(dllimport)
 #endif
#else
 #define CORECUDA_API 
#endif

#include <cuda.h>
#include <cuda_runtime.h>

/* v is a address of the pointer to the array */

/* v is a address of the pointer to the array */
cudaError_t CORECUDA_API malloc_device_(void** d_v, size_t n, const char* file, const unsigned int line);
cudaError_t CORECUDA_API malloc_host_(void** d_v, size_t n, const char* file, const unsigned int line);
#define malloc_device(v,n) malloc_device_((void**)v,n,__FILE__, __LINE__)
#define malloc_host(v,n) malloc_host_((void**)v,n,__FILE__, __LINE__)

void CORECUDA_API free_device_(void* d_v, const char* file, unsigned int line);
void CORECUDA_API free_host_(void* d_v, const char* file, unsigned int line);
#define free_device(v) free_device_((void*)v,__FILE__,__LINE__)
#define free_host(v) free_host_((void*)v,__FILE__,__LINE__)




void CORECUDA_API ss_copyDeviceToHost_(double* dest, double* src, int nxyz, const char* file, const unsigned int line);
#define ss_copyDeviceToHost(d,s,n) ss_copyDeviceToHost_(d, s, n, __FILE__, __LINE__)

void CORECUDA_API ss_copyHostToDevice_(double* dest, double* src, int nxyz, const char* file, const unsigned int line);
#define ss_copyHostToDevice(d,s,n) ss_copyHostToDevice_(d, s, n, __FILE__, __LINE__)

void CORECUDA_API ss_d_set3DArray_(double* d_v, int nx, int ny, int nz, double value, const char* file, const unsigned int line);
#define ss_d_set3DArray(d,x,y,z,v) ss_d_set3DArray_(d,x,y,z,v,__FILE__, __LINE__)

void CORECUDA_API ss_d_add3DArray(double* d_dest, int nx, int ny, int nz, double* d_src1, double* d_src2);


void CORECUDA_API ss_d_scaleadd3DArray(double* d_dest, int n, double s1, double* d_src1, double s2, double* d_src2);

void CORECUDA_API ss_d_scale3DArray(double* d_dest, int n, double s1);


double CORECUDA_API ss_reduce3DArray_sum(double* d_v, double* d_ws1, double* h_ws1, int nx, int ny, int nz);

void CORECUDA_API ss_d_absDiffArrays_(double* d_dest, double* d_src1, double* d_src2, int nxyz, const char* file, const unsigned int line);
#define ss_d_absDiffArrays(d,s1,s2,nxyz) ss_d_absDiffArrays_(d,s1,s2,nxyz, __FILE__, __LINE__)


void CORECUDA_API ss_d_copyArray(double* d_dest, double* d_src, int nxyz);

void CORECUDA_API memcpy_d2d_(void* d_dest, void* d_src, size_t n,const char* file, unsigned int line);
void CORECUDA_API memcpy_d2h_(void* h_dest, void* d_src, size_t n,const char* file, unsigned int line);
void CORECUDA_API memcpy_h2d_(void* d_dest, void* h_src, size_t n,const char* file, unsigned int line);

#define memcpy_d2d(a,b,n) memcpy_d2d_((void*)a, (void*)b, n,__FILE__, __LINE__)
#define memcpy_h2d(a,b,n) memcpy_h2d_((void*)a, (void*)b, n,__FILE__, __LINE__)
#define memcpy_d2h(a,b,n) memcpy_d2h_((void*)a, (void*)b, n,__FILE__, __LINE__)



void CORECUDA_API ss_d_add3DArray(double* d_dest, int nx, int ny, int nz, double* d_src1, double* d_src2);
void CORECUDA_API cuda_addArrays(double* d_dest, int n, const double* d_src1, const double* d_src2);
void CORECUDA_API cuda_scaledAddArrays(double* d_dest, int n, const double s1, const double* d_src1, const double s2, const double* d_src2);
