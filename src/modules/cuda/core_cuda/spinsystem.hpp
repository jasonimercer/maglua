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

/* v is a address of the pointer to the array */
void CORECUDA_API ss_d_make3DArray(double** d_v, int nx, int ny, int nz);
void CORECUDA_API ss_d_free3DArray(double* d_v);

void CORECUDA_API ss_h_make3DArray(double** h_v, int nx, int ny, int nz);
void CORECUDA_API ss_h_free3DArray(double* h_v);

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

