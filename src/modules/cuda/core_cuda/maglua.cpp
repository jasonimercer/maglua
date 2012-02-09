/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "luacommon.h"
#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
CORECUDA_API int lib_register(lua_State* L);
CORECUDA_API int lib_deps(lua_State* L);
CORECUDA_API int lib_version(lua_State* L);
CORECUDA_API const char* lib_name(lua_State* L);
CORECUDA_API int lib_main(lua_State* L);
}

#include "info.h"
#include "spinsystem.h"
#include "spinoperation.h"
#include <string.h>

CORECUDA_API int lib_register(lua_State* L)
{
	registerSpinSystem(L);

	return 0;
}

CORECUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

CORECUDA_API const char* lib_name(lua_State* L)
{
	return "Core-Cuda";
}

#ifndef WIN32
#if CUDART_VERSION >= 2000
static int ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
			int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
			int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 },
		{ 0x11,  8 },
		{ 0x12,  8 },
		{ 0x13,  8 },
		{ 0x20, 32 },
		{ 0x21, 48 },
		{   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) )
		{
			return nGpuArchCoresPerSM[index].Cores;
		}
                index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}
#endif

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error =    cuDeviceGetAttribute( attribute, device_attribute, device );

    if( CUDA_SUCCESS != error) {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
                error, __FILE__, __LINE__);
        exit(-1);
    }
}
#endif

const char* get_gpu_from_arg = 
"for k,v in pairs(arg) do if v == \"-gpu\" then local a = arg[k+1] table.remove(arg, k+1) table.remove(arg,k) return a end end";

CORECUDA_API int lib_main(lua_State* L)
{
	int deviceCount;
	int gpu = -1;
	cudaGetDeviceCount(&deviceCount);

	int a = lua_gettop(L);
	
	if(luaL_dostring(L, get_gpu_from_arg))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return 0;
	}
	
	int b = lua_gettop(L);
	
	if(b == a + 1)
	{
		gpu = lua_tointeger(L, -1);
		while(lua_gettop(L) > a)
			lua_pop(L, 1);
	}

	if(deviceCount == 1)
		gpu = 0;

	if(gpu >= deviceCount || gpu < 0)
	{
		cudaSetDevice(0);
		fprintf(stderr, "WARNING: Using default GPU 0. Select a GPU with -gpu N\n");
		fprintf(stderr, "         N = [0,%i]\n", deviceCount-1);
		return 0;
	}
	
	cudaSetDevice(gpu);
	return 0;
#if 0
	cudaSetDevice(1);

	
	int driverVersion;
	int runtimeVersion;
	int deviceCount;
	int device;
	
	cudaGetDeviceCount(&deviceCount);
	cudaGetDevice(&device);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	for(int i=0; i<argc; i++)
	{
		if(strcasecmp(argv[i], "-cuda_info") == 0)
		{


			
			printf("CUDA Info\n");
			printf("-------------------------------\n");
			printf("Device                 %d/%d\n", device, deviceCount);
        	printf("Driver Version         %d.%d\n", driverVersion/1000, driverVersion%100);
			printf("Runtime Version        %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
			printf("Global Memory     %6.0f MB\n",  (float)deviceProp.totalGlobalMem/1048576.0f);
			#if CUDART_VERSION >= 2000
			printf("Multiprocessors      %3d\n",  deviceProp.multiProcessorCount);
			printf("CUDA Cores/MP        %3d\n",  ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
			printf("CUDA Cores           %3d\n",  ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
			#endif
			#if CUDART_VERSION >= 4000
			int memoryClock;
			getCudaAttribute<int>( &memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device );
			printf("Memory Clock rate:  %.2f Mhz\n", memoryClock * 1e-3f);
			int memBusWidth;
			getCudaAttribute<int>( &memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device );
			printf("Memory Bus Width:  %5d bits\n", memBusWidth);
			int L2CacheSize;
			getCudaAttribute<int>( &L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device );
        	if(L2CacheSize)
			{
				printf("L2 Cache Size:%10d bytes\n", L2CacheSize);
			}

			#endif
			
			printf("Constant memory:%8d bytes\n", (int)deviceProp.totalConstMem); 
			printf("Shared mem/block:%7d bytes\n", (int)deviceProp.sharedMemPerBlock);
			printf("Registers/block: %7d\n", (int)deviceProp.regsPerBlock);
			printf("Warp size:       %7d\n", (int)deviceProp.warpSize);
			printf("Max threads/block:%6d\n", (int)deviceProp.maxThreadsPerBlock);
// 			printf("Maximum sizes of each dimension of a block:    %d x %d x %d\n",
//                deviceProp.maxThreadsDim[0],
//                deviceProp.maxThreadsDim[1],
//                deviceProp.maxThreadsDim[2]);
// 			printf("Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
//                deviceProp.maxGridSize[0],
//                deviceProp.maxGridSize[1],
//                deviceProp.maxGridSize[2]);
// 			printf("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
// 			printf("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);

			
			printf("-------------------------------\n");
			
			
			return;
		}
	}
#endif
}
