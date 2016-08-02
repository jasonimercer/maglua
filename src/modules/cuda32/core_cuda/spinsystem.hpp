/* ss = spin system
 *  d = device
 *  h = host
 */

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning32(disable: 4251)

 #ifdef CORECUDA_EXPORTS
  #define CORECUDA_API __declspec32(dllexport)
 #else
  #define CORECUDA_API __declspec32(dllimport)
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




void CORECUDA_API ss_copyDeviceToHost32_(float* dest, float* src, int nxyz, const char* file, const unsigned int line);
#define ss_copyDeviceToHost32(d,s,n) ss_copyDeviceToHost32_(d, s, n, __FILE__, __LINE__)

void CORECUDA_API ss_copyHostToDevice32_(float* dest, float* src, int nxyz, const char* file, const unsigned int line);
#define ss_copyHostToDevice32(d,s,n) ss_copyHostToDevice32_(d, s, n, __FILE__, __LINE__)

void CORECUDA_API ss_d_set3DArray32_(float* d_v, int nx, int ny, int nz, float value, const char* file, const unsigned int line);
#define ss_d_set3DArray32(d,x,y,z,v) ss_d_set3DArray32_(d,x,y,z,v,__FILE__, __LINE__)

void CORECUDA_API ss_d_add3DArray32(float* d_dest, int nx, int ny, int nz, float* d_src1, float* d_src2);


void CORECUDA_API ss_d_scaleadd3DArray32(float* d_dest, int n, float s1, float* d_src1, float s2, float* d_src2);

void CORECUDA_API ss_d_scale3DArray32(float* d_dest, int n, float s1);


float CORECUDA_API ss_reduce3DArray_sum32(float* d_v, float* d_ws1, float* h_ws1, int nx, int ny, int nz);

void CORECUDA_API ss_d_absDiffArrays32_(float* d_dest, float* d_src1, float* d_src2, int nxyz, const char* file, const unsigned int line);
#define ss_d_absDiffArrays32(d,s1,s2,nxyz) ss_d_absDiffArrays32_(d,s1,s2,nxyz, __FILE__, __LINE__)


void CORECUDA_API ss_d_copyArray32(float* d_dest, float* d_src, int nxyz);

void CORECUDA_API memcpy_d2d_(void* d_dest, void* d_src, size_t n,const char* file, unsigned int line);
void CORECUDA_API memcpy_d2h_(void* h_dest, void* d_src, size_t n,const char* file, unsigned int line);
void CORECUDA_API memcpy_h2d_(void* d_dest, void* h_src, size_t n,const char* file, unsigned int line);

#define memcpy_d2d(a,b,n) memcpy_d2d_((void*)a, (void*)b, n,__FILE__, __LINE__)
#define memcpy_h2d(a,b,n) memcpy_h2d_((void*)a, (void*)b, n,__FILE__, __LINE__)
#define memcpy_d2h(a,b,n) memcpy_d2h_((void*)a, (void*)b, n,__FILE__, __LINE__)

void CORECUDA_API ss_d_add3DArray32(float* d_dest, int nx, int ny, int nz, float* d_src1, float* d_src2);
void CORECUDA_API cuda_addArrays32(float* d_dest, int n, const float* d_src1, const float* d_src2);
void CORECUDA_API cuda_scaledAddArrays32(float* d_dest, int n, const float s1, const float* d_src1, const float s2, const float* d_src2);
