#include <cuda.h>
#include <cuda_runtime.h>

#include "memory.hpp"
#include <stdio.h>

#define KCHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
	{ \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
		fprintf(logfile,"(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
		fflush(logfile); \
	}\
}

#define KCHECK_FL(f,l) \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
	{ \
		printf("(%s:%i) %s\n",  f, l, cudaGetErrorString(i));\
		fprintf(logfile,"(%s:%i) %s\n",  f, l, cudaGetErrorString(i));\
		fflush(logfile); \
	} \
}


#define SEGFAULT \
{ \
	long* i = 0; \
	*i = 5; \
}

static FILE* logfile = 0;
void log_print(const char* msg)
{
	if(!logfile)
    {
		logfile = fopen("malloc.log", "w");
    }
    
    fprintf(stderr, msg);
    fprintf(logfile, msg);
	fflush(logfile);
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
		fflush(logfile); \
		fclose(logfile); \
		int* i = 0;\
		*i = 4; \
	} \
}

#define CHECKCALL_FLe(lval,expression, file, line)	\
{ \
	lval = (expression); \
	if(lval != cudaSuccess) \
	{ \
		printf("(%s:%i) %s => (%i)%s\n", file, line, #expression, lval, cudaGetErrorString(lval)); \
		fprintf(logfile,"(%s:%i) %s => (%i)%s\n", file, line, #expression, lval, cudaGetErrorString(lval)); \
		fflush(logfile); \
		fclose(logfile); \
		int* i = 0;\
		*i = 4; \
	} \
}

#define CHECKCALL(expression)  CHECKCALL_FL(expression, __FILE__, __LINE__)
#define CHECKCALLe(lval,expression)  CHECKCALL_FLe(lval,expression, __FILE__, __LINE__)

cudaError_t malloc_device_(void** d_v, size_t n, const char* file, unsigned int line)
{
    if(!logfile)
    {
		logfile = fopen("malloc.log", "w");
    }

    cudaError_t err;
// 	printf("malloc_device %i bytes\n", n);
// 	fprintf(logfile, "malloc_device [%10lu/%10lu] (%s:%i) %8li %p\n", memTotal()-memLeft(), memTotal(), file, line, n, *d_v);

	
	CHECKCALL_FLe(err,cudaMalloc(d_v, n), file, line);
	fprintf(logfile, "malloc_device (%s:%i) %8li %p\n", file, line, n, *d_v);
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
	if(!logfile)
    {
		logfile = fopen("malloc.log", "w");
    }
    
    cudaError_t err;
    CHECKCALL_FLe(err,cudaMallocHost(h_v, n),file,line);

	fprintf(logfile, "malloc_host (%s:%i) %8li %p\n", file, line, n, *h_v);
	fflush(logfile);
	return err; //to mirror malloc_device
}

void free_host_(void* h_v, const char* file, unsigned int line)
{
	CHECKCALL_FL(cudaFreeHost(h_v),file,line);
}



cudaError_t ARRAYCUDA_API malloc_dh_(void** d, void** h, size_t n, const char* file, const unsigned int line)
{
	        malloc_host_(h, n, file, line);
	return  malloc_device_(d, n, file, line);
}


void ARRAYCUDA_API free_dh_(void* d, void* h, const char* file, unsigned int line)
{
	free_device_(d, file, line);
	free_host_(h, file, line);
}



void memcpy_d2d_(void* d_dest, void* d_src, size_t n, const char* file, const unsigned int line)
{
    CHECKCALL_FL(cudaMemcpy(d_dest, d_src, n, cudaMemcpyDeviceToDevice),file,line);
}

void memcpy_d2h_(void* h_dest, void* d_src, size_t n, const char* file, const unsigned int line)
{
	if(!logfile)
    {
		logfile = fopen("malloc.log", "w");
    }
	fprintf(logfile, "memcpy_d2h (%s:%i) %p <- %p (%li)\n", file, line, h_dest, d_src, n);
	fflush(logfile);
	
    CHECKCALL_FL(cudaMemcpy(h_dest, d_src, n, cudaMemcpyDeviceToHost),file,line);
}

void memcpy_h2d_(void* d_dest, void* h_src, size_t n, const char* file, const unsigned int line)
{
	if(!logfile)
    {
		logfile = fopen("malloc.log", "w");
    }
	fprintf(logfile, "memcpy_h2d (%s:%i) %p <- %p (%li)\n", file, line, d_dest, h_src, n);
	fflush(logfile);

	
	// 	printf("h2d: %p %p\n", d_dest, h_src);
    CHECKCALL_FL(cudaMemcpy(d_dest, h_src, n, cudaMemcpyHostToDevice),file,line);
}













#define MEMORY_SLOTS 256



typedef struct work_space_device_memory
{
	int refcount;
	void* d_memory[MEMORY_SLOTS];
	size_t size[MEMORY_SLOTS];
	int slot_refcount[MEMORY_SLOTS];
	long level[MEMORY_SLOTS];
	
} work_space_device_memory;

static work_space_device_memory WS_MEM_D = {0};
static work_space_device_memory WS_MEM_H = {0};

ARRAYCUDA_API void  registerWS()
{
	if(WS_MEM_D.refcount == 0) //initialize
	{
		for(int i=0; i<MEMORY_SLOTS; i++)
		{
			WS_MEM_D.d_memory[i] = 0;
			WS_MEM_D.size[i] = 0;
			WS_MEM_D.slot_refcount[i] = 0;
			WS_MEM_D.level[i] = -1000; //dumb
		}
	}
	
	if(WS_MEM_H.refcount == 0) //initialize
	{
		for(int i=0; i<MEMORY_SLOTS; i++)
		{
			WS_MEM_H.d_memory[i] = 0;
			WS_MEM_H.size[i] = 0;
			WS_MEM_H.slot_refcount[i] = 0;
			WS_MEM_H.level[i] = -1000; //dumb
		}
	}
	
	WS_MEM_D.refcount++;
	WS_MEM_H.refcount++;
}

ARRAYCUDA_API void unregisterWS()
{
	WS_MEM_D.refcount--;
	if(WS_MEM_D.refcount == 0)
	{
		for(int i=0; i<MEMORY_SLOTS; i++)
		{
			if(WS_MEM_D.d_memory[i])
				free_device(WS_MEM_D.d_memory[i]);
			WS_MEM_D.d_memory[i] = 0;
			WS_MEM_D.size[i] = 0;
			WS_MEM_D.level[i] = -1000; //dumb
		}
	}

	WS_MEM_H.refcount--;
	if(WS_MEM_H.refcount == 0)
	{
		for(int i=0; i<MEMORY_SLOTS; i++)
		{
			if(WS_MEM_H.d_memory[i])
				free_host(WS_MEM_H.d_memory[i]);
			WS_MEM_H.d_memory[i] = 0;
			WS_MEM_H.size[i] = 0;
			WS_MEM_H.level[i] = -1000; //dumb
		}
	}
}

// multiple operations can use the same workspace so you had better make sure 
// the workspace isn't storing something important: Keep ws usage contained
ARRAYCUDA_API bool getWSMemD_(void** ptr, size_t size, long level)
{
	for(int i=0; i<MEMORY_SLOTS; i++)
	{
		if(WS_MEM_D.level[i] == -1000 || WS_MEM_D.level[i] == level)
		{
			if(WS_MEM_D.size[i] == 0)
			{
				malloc_device(&(WS_MEM_D.d_memory[i]), size);
				WS_MEM_D.size[i] = size;
			}
			
			if(WS_MEM_D.size[i] >= size)
			{
				WS_MEM_D.slot_refcount[i]++;
				*ptr = WS_MEM_D.d_memory[i];
				WS_MEM_D.level[i] = level;
				return true;
			}
		}
	}
	
	char buffer[1024];
	
	snprintf(buffer, 1024, "Failed to allocate device memory for data of size %li (level = %li)\n", size, level);
	log_print(buffer);
	snprintf(buffer, 1024, "Here is the current list:\n");
	log_print(buffer);
	for(int i=0; i<MEMORY_SLOTS; i++)
	{
		snprintf(buffer, 1024, "% 3i  size: %8li  level: %8li\n", i, WS_MEM_D.size[i], WS_MEM_D.level[i]);
		log_print(buffer);
	}
	
	*ptr = 0;
	return false;
}


// multiple operations can use the same workspace so you had better make sure 
// the workspace isn't storing something important: Keep ws usage contained
ARRAYCUDA_API bool getWSMemH_(void** ptr, size_t size, long level)
{
	for(int i=0; i<MEMORY_SLOTS; i++)
	{
		if(WS_MEM_H.level[i] == -1000 || WS_MEM_H.level[i] == level)
		{
			if(WS_MEM_H.size[i] == 0)
			{
				malloc_host(&(WS_MEM_H.d_memory[i]), size);
				WS_MEM_H.size[i] = size;
			}
			
			if(WS_MEM_H.size[i] >= size)
			{
				WS_MEM_H.slot_refcount[i]++;
				*ptr = WS_MEM_H.d_memory[i];
				WS_MEM_H.level[i] = level;
				return true;
			}
		}
	}
	
	char buffer[1024];
	
	snprintf(buffer, 1024, "Failed to allocate host memory for data of size %li (level = %li)\n", size, level);
	log_print(buffer);
	snprintf(buffer, 1024, "Here is the current list:\n");
	log_print(buffer);
	for(int i=0; i<MEMORY_SLOTS; i++)
	{
		snprintf(buffer, 1024, "% 3i  size: %8li  level: %8li\n", i, WS_MEM_H.size[i], WS_MEM_H.level[i]);
		log_print(buffer);
	}

	
	*ptr = 0;
	return false;
}

