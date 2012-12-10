#include <cuda.h>
#include <cuda_runtime.h>

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ARRAYCUDA_EXPORTS
  #define ARRAYCUDA_API __declspec(dllexport)
 #else
  #define ARRAYCUDA_API __declspec(dllimport)
 #endif
#else
 #define ARRAYCUDA_API 
#endif

/* v is a address of the pointer to the array */
cudaError_t ARRAYCUDA_API malloc_device_(void** d_v, size_t n, const char* file, const unsigned int line);
cudaError_t ARRAYCUDA_API malloc_host_(void** d_v, size_t n, const char* file, const unsigned int line);
#define malloc_device(v,n) malloc_device_((void**)v,n,__FILE__, __LINE__)
#define malloc_host(v,n) malloc_host_((void**)v,n,__FILE__, __LINE__)


cudaError_t ARRAYCUDA_API malloc_dh_(void** d, void** h, size_t n, const char* file, const unsigned int line);
#define malloc_dh(d,h,n) malloc_dh_((void**)d,(void**)h,n,__FILE__, __LINE__)


void ARRAYCUDA_API free_device_(void* d_v, const char* file, unsigned int line);
void ARRAYCUDA_API free_host_(void* d_v, const char* file, unsigned int line);
#define free_device(v) free_device_((void*)v,__FILE__,__LINE__)
#define free_host(v) free_host_((void*)v,__FILE__,__LINE__)


void ARRAYCUDA_API free_dh_(void* d, void* h, const char* file, unsigned int line);
#define free_dh(d,h) free_dh_((void*)d,(void*)h,__FILE__,__LINE__)



void ARRAYCUDA_API memcpy_d2d_(void* d_dest, void* d_src, size_t n,const char* file, unsigned int line);
void ARRAYCUDA_API memcpy_d2h_(void* h_dest, void* d_src, size_t n,const char* file, unsigned int line);
void ARRAYCUDA_API memcpy_h2d_(void* d_dest, void* h_src, size_t n,const char* file, unsigned int line);

#define memcpy_d2d(a,b,n) memcpy_d2d_((void*)a, (void*)b, n,__FILE__, __LINE__)
#define memcpy_h2d(a,b,n) memcpy_h2d_((void*)a, (void*)b, n,__FILE__, __LINE__)
#define memcpy_d2h(a,b,n) memcpy_d2h_((void*)a, (void*)b, n,__FILE__, __LINE__)





ARRAYCUDA_API void  registerWS();
ARRAYCUDA_API void  unregisterWS();

ARRAYCUDA_API void  getWSMem5_(
			   void** ptr1, size_t size1, 
			   void** ptr2, size_t size2, 
			   void** ptr3, size_t size3,
			   void** ptr4, size_t size4,
			   void** ptr5, size_t size5);
ARRAYCUDA_API void  getWSMem4_(
			   void** ptr1, size_t size1, 
			   void** ptr2, size_t size2, 
			   void** ptr3, size_t size3,
			   void** ptr4, size_t size4);
ARRAYCUDA_API void  getWSMem3_(
			   void** ptr1, size_t size1, 
			   void** ptr2, size_t size2, 
			   void** ptr3, size_t size3);
ARRAYCUDA_API void  getWSMem2_(
			   void** ptr1, size_t size1, 
			   void** ptr2, size_t size2);
ARRAYCUDA_API void  getWSMem1_(
			   void** ptr1, size_t size1);

#define getWSMem1(p1,s1) getWSMem1_((void**)p1, s1)
#define getWSMem2(p1,s1, p2,s2) getWSMem2_((void**)p1, s1, (void**)p2, s2)
#define getWSMem3(p1,s1, p2,s2, p3,s3) getWSMem3_((void**)p1, s1, (void**)p2, s2, (void**)p3, s3)
#define getWSMem4(p1,s1, p2,s2, p3,s3, p4,s4) getWSMem4_((void**)p1, s1, (void**)p2, s2, (void**)p3, s3, (void**)p4, s4)
#define getWSMem5(p1,s1, p2,s2, p3,s3, p4,s4, p5,s5) getWSMem5_((void**)p1, s1, (void**)p2, s2, (void**)p3, s3, (void**)p4, s4, (void**)p5, s5)
