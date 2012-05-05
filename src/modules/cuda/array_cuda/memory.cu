#include <cuda.h>
#include <cuda_runtime.h>

#include "memory.hpp"
#include <stdio.h>

#define KCHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}

#define KCHECK_FL(f,l) \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  f, l, cudaGetErrorString(i));\
}


#define SEGFAULT \
{ \
	long* i = 0; \
	*i = 5; \
}

static size_t memTotal()
{
	size_t free, total;
	//cuMemGetInfo(&free, &total);
	cudaMemGetInfo(&free, &total);       

	return total;
}

static size_t memLeft()
{
// 	CUresult res;
	size_t free, total;
	//cuMemGetInfo(&free, &total);
	cudaMemGetInfo(&free, &total); 

	return free;
}

#define CHECKCALL_FL(expression, file, line) \
{ \
	const cudaError_t err = (expression); \
	if(err != cudaSuccess) \
	{ \
		printf("(%s:%i) %s => (%i)%s\n", file, line, #expression, err, cudaGetErrorString(err)); \
		fprintf(logfile,"(%s:%i) %s => (%i)%s\n", file, line, #expression, err, cudaGetErrorString(err)); \
	} \
}

#define CHECKCALL_FLe(lval,expression, file, line)	\
{ \
	lval = (expression); \
	if(lval != cudaSuccess) \
	{ \
		printf("(%s:%i) %s => (%i)%s\n", file, line, #expression, lval, cudaGetErrorString(lval)); \
		fprintf(logfile,"(%s:%i) %s => (%i)%s\n", file, line, #expression, lval, cudaGetErrorString(lval)); \
	} \
}

#define CHECKCALL(expression)  CHECKCALL_FL(expression, __FILE__, __LINE__)
#define CHECKCALLe(lval,expression)  CHECKCALL_FLe(lval,expression, __FILE__, __LINE__)

static FILE* logfile = 0;
cudaError_t malloc_device_(void** d_v, size_t n, const char* file, unsigned int line)
{
    if(!logfile)
    {
	logfile = fopen("malloc.log", "w");
    }

    cudaError_t err;
// 	printf("malloc_device %i bytes\n", n);
    CHECKCALL_FLe(err,cudaMalloc(d_v, n), file, line);

	
	fprintf(logfile, "[%10lu/%10lu] (%s:%i) %8li %p\n", memTotal()-memLeft(), memTotal(), file, line, n, *d_v);
// 	fprintf(logfile, "free %i bytes\n",  memLeft());
	fflush(logfile);

	//TODO here we check for fail and compress if needed
	
	return err; //eventually this will reflect succesfulness of malloc
}

void free_device_(void* d_v, const char* file, unsigned int line)
{
	CHECKCALL_FL(cudaFree(d_v), file,line);
}

cudaError_t malloc_host_(void** h_v, size_t n, const char* file, unsigned int line)
{
    cudaError_t err;
    CHECKCALL_FLe(err,cudaMallocHost(h_v, n),file,line);
    return err; //to mirror malloc_device
}

void free_host_(void* h_v, const char* file, unsigned int line)
{
	CHECKCALL_FL(cudaFreeHost(h_v),file,line);
}





void memcpy_d2d_(void* d_dest, void* d_src, size_t n, const char* file, const unsigned int line)
{
    CHECKCALL_FL(cudaMemcpy(d_dest, d_src, n, cudaMemcpyDeviceToDevice),file,line);
}

void memcpy_d2h_(void* h_dest, void* d_src, size_t n, const char* file, const unsigned int line)
{
    CHECKCALL_FL(cudaMemcpy(h_dest, d_src, n, cudaMemcpyDeviceToHost),file,line);
}

void memcpy_h2d_(void* d_dest, void* h_src, size_t n, const char* file, const unsigned int line)
{
    CHECKCALL_FL(cudaMemcpy(d_dest, h_src, n, cudaMemcpyHostToDevice),file,line);
}
