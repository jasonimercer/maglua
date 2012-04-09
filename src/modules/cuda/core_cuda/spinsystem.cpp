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

#include "spinsystem.h"
#include "spinoperation.h"
#include <iostream>
#include <math.h>
#ifndef WIN32
#include <strings.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "luamigrate.h"
#include "spinsystem.hpp"

using namespace std;
#define CLAMP(x, m) ((x<0)?0:(x>m?m:x))

void  getWSMem(double** ptr0, size_t size0, 
			   double** ptr1, size_t size1, 
			   double** ptr2, size_t size2,
			   double** ptr3, size_t size3,
			   double** ptr4, size_t size4)
{
	getWSMem( (void**) ptr0, size0, 
			  (void**) ptr1, size1, 
			  (void**) ptr2, size2, 
			  (void**) ptr3, size3, 
			  (void**) ptr4, size4 );
}

void  getWSMem(void** ptr0, size_t size0, 
			   void** ptr1, size_t size1, 
			   void** ptr2, size_t size2,
			   void** ptr3, size_t size3,
			   void** ptr4, size_t size4)
{
	void** ptr[5]  = {ptr0, ptr1, ptr2, ptr3, ptr4};
	size_t size[5] = {size0, size1, size2, size3, size4};
	
	for(int i=0; i<5; i++)
	{
		if(!ptr[i]) return;
		if(size[i] > WS_MEM.size[i]) 
		{
			free_device(WS_MEM.d_memory[i]);
			malloc_device(&(WS_MEM.d_memory[i]), size[i]);
			WS_MEM.size[i] = size[i];
		}
		*ptr[i] = WS_MEM.d_memory[i];
	}
}



void registerWS()
{
	if(!WS_MEM.refcount)
	{
		for(int i=0; i<5; i++)
		{
				WS_MEM.d_memory[i] = 0;
				WS_MEM.size[i] = 0;
		}
	}
	WS_MEM.refcount++;
}

void  unregisterWS()
{
	WS_MEM.refcount--;
	
	if(!WS_MEM.refcount)
	{
		for(int i=0; i<5; i++)
		{
			free_device(WS_MEM.d_memory[i]);
			WS_MEM.d_memory[i] = 0;
			WS_MEM.size[i] = 0;
		}
	}
}


SpinSystem::SpinSystem(const int NX, const int NY, const int NZ)
	: LuaBaseObject(ENCODE_SPINSYSTEM), d_x(0), d_y(0), d_z(0), 
		d_ms(0), gamma(1.0), alpha(1.0), dt(1.0),
		nx(NX), ny(NY), nz(NZ),
		time(0)
{
	d_x = 0;
	init();
	L = 0;
}

void SpinSystem::push(lua_State* L)
{
	luaT_push<SpinSystem>(L, this);
}

int SpinSystem::luaInit(lua_State* L)
{
	deinit();
	int n[3];
	if(luaT_is<SpinSystem>(L, 1))
	{
		SpinSystem* ss = luaT_to<SpinSystem>(L, 1);
		n[0] = ss->nx;
		n[1] = ss->ny;
		n[2] = ss->nz;
	}
	else
	{
		lua_getNint(L, 3, n, 1, 1);
	}
	nx = n[0];
	ny = n[1];
	nz = n[2];
	init();
	return 0;
}


void SpinSystem::sync_spins_dh(bool force)
{
	if(new_device_spins || force)
	{
		if(new_host_spins)
		{
			printf("(%s:%i) overwriting new host spins\n", __FILE__, __LINE__);
		}
		
		const size_t dxyz = sizeof(double) * nxyz;

		memcpy_d2h(h_x, d_x, dxyz);
		memcpy_d2h(h_y, d_y, dxyz);
		memcpy_d2h(h_z, d_z, dxyz);
		memcpy_d2h(h_ms,d_ms,dxyz);

		new_device_spins = false;
		new_host_spins = false;
	}
}

void SpinSystem::sync_fields_dh(int field, bool force)
{
	ensureSlotExists(field);

	if(new_device_fields[field] || force)
	{
		const size_t dxyz = sizeof(double) * nxyz;
// 	  printf("(%s:%i) sync_fields_dh(%i)\n", __FILE__, __LINE__, field);
		memcpy_d2h(h_hx[field], d_hx[field], dxyz);
		memcpy_d2h(h_hy[field], d_hy[field], dxyz);
		memcpy_d2h(h_hz[field], d_hz[field], dxyz);
		
		new_device_fields[field] = false;
		new_host_fields[field] = false;
	}
}

void SpinSystem::sync_spins_hd(bool force)
{
	if(new_host_spins || force)
	{
		const size_t dxyz = sizeof(double) * nxyz;

// 	  printf("(%s:%i) sync_spins_hd\n", __FILE__, __LINE__);
		memcpy_h2d(d_x, h_x, dxyz);
		memcpy_h2d(d_y, h_y, dxyz);
		memcpy_h2d(d_z, h_z, dxyz);
		memcpy_h2d(d_ms,h_ms,dxyz);
		
		new_host_spins = false;
		new_device_spins = false;
	}
}

void SpinSystem::sync_fields_hd(int field, bool force)
{
	ensureSlotExists(field);
	if(new_host_fields[field] || force)
	{
		const size_t dxyz = sizeof(double) * nxyz;

// 	  printf("(%s:%i) sync_fields_hd(%i)\n", __FILE__, __LINE__, field);
		memcpy_h2d(d_hx[field], h_hx[field], dxyz);
		memcpy_h2d(d_hy[field], h_hy[field], dxyz);
		memcpy_h2d(d_hz[field], h_hz[field], dxyz);

		new_host_fields[field] = false;
		new_device_fields[field] = false;
	}
}



SpinSystem::~SpinSystem()
{
	deinit();
}

SpinSystem* SpinSystem::copy(lua_State* L)
{
	SpinSystem* c = new SpinSystem(nx, ny, nz);
	
	c->copyFrom(L, this);
	
	return c;
}

void SpinSystem::diff(SpinSystem* other, double* v4)
{
	if(	nx != other->nx ||
		ny != other->ny ||
		nz != other->nz)
	{
		v4[0] = 1e8;
		v4[1] = 1e8;
		v4[2] = 1e8;
		v4[3] = 1e8;
		return;
	}
	
	v4[0] = 0;
	v4[1] = 0;
	v4[2] = 0;
	
	//ensure spins are on the device
	sync_spins_hd();
	other->sync_spins_hd();

	double* d_ws1;
	double* d_ws2;
	
	const int sz = sizeof(double)*nxyz;
	getWSMem(&d_ws1, sz, &d_ws2, sz);//, &d_ws3, sz, &d_ws4, sz);
	
	ss_d_absDiffArrays(d_ws1, d_x, other->d_x, nxyz);
	v4[0] = ss_reduce3DArray_sum(d_ws1, d_ws2, h_ws1, nx, ny, nz);

	ss_d_absDiffArrays(d_ws1, d_y, other->d_y, nxyz);
	v4[1] = ss_reduce3DArray_sum(d_ws1, d_ws2, h_ws1, nx, ny, nz);

	ss_d_absDiffArrays(d_ws1, d_z, other->d_z, nxyz);
	v4[2] = ss_reduce3DArray_sum(d_ws1, d_ws2, h_ws1, nx, ny, nz);
	
	v4[3] = sqrt(v4[0]*v4[0] + v4[1]*v4[1] + v4[2]*v4[2]);
}


bool SpinSystem::copyFrom(lua_State* L, SpinSystem* src)
{
	if(nx != src->nx) return false;
	if(ny != src->ny) return false;
	if(nz != src->nz) return false;
	
	copySpinsFrom(L, src);
	copyFieldsFrom(L, src);
	
	alpha = src->alpha;
	gamma = src->gamma;
	dt = src->dt;
	time = src->time;
	
	// unref data - if exists
	for(int i=0; i<nxyz; i++)
	{
		if(extra_data[i] != LUA_REFNIL)
			luaL_unref(L, LUA_REGISTRYINDEX, extra_data[i]);
		extra_data[i] = LUA_REFNIL;
	}
	
	// make copies of references
	for(int i=0; i<nxyz; i++)
	{
		if(src->extra_data[i] != LUA_REFNIL)
		{
			lua_rawgeti(L, LUA_REGISTRYINDEX, src->extra_data[i]);
			extra_data[i] = luaL_ref(L, LUA_REGISTRYINDEX);
		}
	}
	
	return true;
}

bool SpinSystem::copySpinsFrom(lua_State* L, SpinSystem* src)
{
	if(nx != src->nx) return false;
	if(ny != src->ny) return false;
	if(nz != src->nz) return false;
	
	src->sync_spins_hd();
	
	ss_d_copyArray(d_x,  src->d_x,  nxyz);
	ss_d_copyArray(d_y,  src->d_y,  nxyz);
	ss_d_copyArray(d_z,  src->d_z,  nxyz);
	ss_d_copyArray(d_ms, src->d_ms, nxyz);
	
	new_host_spins = false;
	new_device_spins = true;

	return true;
}

bool SpinSystem::copyFieldFrom(lua_State* L, SpinSystem* src, int slot)
{
	if(nx != src->nx) return false;
	if(ny != src->ny) return false;
	if(nz != src->nz) return false;

	if(slot < 0 || slot >= NSLOTS)
		return false;
	
	src->ensureSlotExists(slot);
	ensureSlotExists(slot);
	
	src->sync_fields_hd(slot);
	ss_d_copyArray(d_hx[slot], src->d_hx[slot], nxyz);
	ss_d_copyArray(d_hy[slot], src->d_hy[slot], nxyz);
	ss_d_copyArray(d_hz[slot], src->d_hz[slot], nxyz);

	slot_used[slot] = true;
	new_host_fields[slot] = false;
	new_device_fields[slot] = true;
	
	return true;
}

bool SpinSystem::copyFieldsFrom(lua_State* L, SpinSystem* src)
{
	return copyFieldFrom(L, src, SUM_SLOT);
}


void SpinSystem::deinit()
{
	if(d_x)
	{
		if(L)
		{
			for(int i=0; i<nxyz; i++)
			{
				if(extra_data[i] != LUA_REFNIL)
				{
					lua_unref(L, extra_data[i]);
				}
			}
		}
		
		free_device(d_x);
		free_device(d_y);
		free_device(d_z);
		free_device(d_ms);
		
		unregisterWS();
		
		free_host(h_ws1);
			
		free_host(h_x);
		free_host(h_y);
		free_host(h_z);
		free_host(h_ms);
		
		delete [] slot_used;
		
		for(int i=0; i<NSLOTS; i++)
		{
			if(d_hx[i])
			{
				free_device(d_hx[i]);
				free_device(d_hy[i]);
				free_device(d_hz[i]);
			}

			if(d_hx[i])
			{
				free_host(h_hx[i]);
				free_host(h_hy[i]);
				free_host(h_hz[i]);
			}
		}

		delete [] d_hx;
		delete [] d_hy;
		delete [] d_hz;
		
		delete [] h_hx;
		delete [] h_hy;
		delete [] h_hz;
		
		delete [] extra_data;
		
		delete [] new_host_fields;
		delete [] new_device_fields;
		d_x = 0;
	}
}


void SpinSystem::init()
{
	nxyz = nx * ny * nz;
	if(d_x)
		deinit();

	new_host_spins = false;
	new_device_spins = false;

	
	new_host_fields = new bool[NSLOTS]; 
	new_device_fields = new bool[NSLOTS]; 

	for(int i=0; i<NSLOTS; i++)
	{
		new_host_fields[i] = false;
		new_device_fields[i] = false;
	}


	malloc_device(&d_x,  nx*ny*nz*sizeof(double));
	malloc_device(&d_y,  nx*ny*nz*sizeof(double));
	malloc_device(&d_z,  nx*ny*nz*sizeof(double));
	malloc_device(&d_ms, nx*ny*nz*sizeof(double));

	malloc_host(&h_x,  nx*ny*nz*sizeof(double));
	malloc_host(&h_y,  nx*ny*nz*sizeof(double));
	malloc_host(&h_z,  nx*ny*nz*sizeof(double));
	malloc_host(&h_ms, nx*ny*nz*sizeof(double));

	for(int i=0; i<nx*ny*nz; i++)
	{
		h_x[i] = 0;
		h_y[i] = 0;
		h_z[i] = 0;
		h_ms[i] = 0;
	}

	//set spins to (0,0,0)
	// setting them on the device
	ss_d_set3DArray(d_x, nx, ny, nz, 0);
	ss_d_set3DArray(d_y, nx, ny, nz, 0);
	ss_d_set3DArray(d_z, nx, ny, nz, 0);
	ss_d_set3DArray(d_ms, nx, ny, nz, 0);
	new_device_spins = true;
	sync_spins_dh();
	
	
	registerWS();
	
	const int min_size = (nx*ny*nz) / 64 + 1;
	
	if(nx*ny*nz < min_size)
	{
		malloc_host(&h_ws1, sizeof(double)*min_size);
	}
	else
		malloc_host(&h_ws1, sizeof(double)*nx*ny*nz);
	
	
	d_hx = new double* [NSLOTS];
	d_hy = new double* [NSLOTS];
	d_hz = new double* [NSLOTS];
	
	h_hx = new double* [NSLOTS];
	h_hy = new double* [NSLOTS];
	h_hz = new double* [NSLOTS];
	
	slot_used = new bool[NSLOTS];

	extra_data = new int[nxyz];
	for(int i=0; i<nxyz; i++)
		extra_data[i] = LUA_REFNIL;

	for(int i=0; i<NSLOTS; i++)
	{
		d_hx[i] = 0;
		d_hy[i] = 0;
		d_hz[i] = 0;
		
		h_hx[i] = 0;
		h_hy[i] = 0;
		h_hz[i] = 0;

		slot_used[i] = false;
	}
}

void SpinSystem::ensureSlotExists(int slot)
{
	if(d_hx[slot]) return;

	const int nn = sizeof(double) * nx*ny*nz;
	malloc_device(&(d_hx[slot]), nn);
	malloc_device(&(d_hy[slot]), nn);
	malloc_device(&(d_hz[slot]), nn);

	ss_d_set3DArray(d_hx[slot], nx, ny, nz, 0);
	ss_d_set3DArray(d_hy[slot], nx, ny, nz, 0);
	ss_d_set3DArray(d_hz[slot], nx, ny, nz, 0);

	malloc_host(&(h_hx[slot]), nn);
	malloc_host(&(h_hy[slot]), nn);
	malloc_host(&(h_hz[slot]), nn);
		
	memcpy_d2h(h_hx[slot], d_hx[slot], nn);
	memcpy_d2h(h_hy[slot], d_hy[slot], nn);
	memcpy_d2h(h_hz[slot], d_hz[slot], nn);
}


void SpinSystem::encode(buffer* b)
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);

	encodeDouble(alpha, b);
	encodeDouble(   dt, b);
	encodeDouble(gamma, b);

	encodeDouble(time, b);
	
	sync_spins_dh();
	for(int i=0; i<nxyz; i++)
	{
		encodeDouble(h_x[i], b);
		encodeDouble(h_y[i], b);
		encodeDouble(h_z[i], b);
	}
	
	
	int numExtraData = 0;
	
	for(int i=0; i<nxyz; i++)
	{
		if(extra_data[i] != LUA_REFNIL)
			numExtraData++;
	}
	
	if(numExtraData > nxyz / 2) //then we'll write all explicitly
	{
		encodeInteger(-1, b); //flag for "all data"
		for(int i=0; i<nxyz; i++)
		{
			if(extra_data[i] != LUA_REFNIL)
			{
				lua_rawgeti(L, LUA_REGISTRYINDEX, extra_data[i]);
			}
			else
			{
				lua_pushnil(L);
			}
			
			_exportLuaVariable(L, -1, b);
			lua_pop(L, 1);
		}
	}
	else
	{
		encodeInteger(numExtraData, b); //flag for "partial data" and number of partial data
		for(int i=0; i<nxyz; i++)
		{
			if(extra_data[i] != LUA_REFNIL)
			{
				encodeInteger(i, b);
				lua_rawgeti(L, LUA_REGISTRYINDEX, extra_data[i]);
				_exportLuaVariable(L, -1, b);
				lua_pop(L, 1);
			}
		}
	}
}

int  SpinSystem::decode(buffer* b)
{
	deinit();
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	nxyz = nx*ny*nz;
	
	alpha = decodeDouble(b);
	dt = decodeDouble(b);
	gamma = decodeDouble(b);

	time = decodeDouble(b);
	init();

	for(int j=0; j<nxyz; j++)
	{
		h_x[j] = decodeDouble(b);
		h_y[j] = decodeDouble(b);
		h_z[j] = decodeDouble(b);
		h_ms[j] = sqrt(h_x[j]*h_x[j]+h_y[j]*h_y[j]+h_z[j]*h_z[j]);
	}
	new_host_spins = true;
	new_device_spins = false;


	int numPartialData = decodeInteger(b);
	if(numPartialData < 0) //then all, implicitly
	{
		for(int i=0; i<nxyz; i++)
		{
			_importLuaVariable(L, b);
			extra_data[i] = luaL_ref(L, LUA_REGISTRYINDEX);
		}
	}
	else
	{
		for(int i=0; i<numPartialData; i++)
		{
			int idx = decodeInteger(b);
			_importLuaVariable(L, b);
			extra_data[idx] = luaL_ref(L, LUA_REGISTRYINDEX);
		}
	}

	return 0;
}



void SpinSystem::sumFields()
{
	ensureSlotExists(SUM_SLOT);
	sync_fields_hd(SUM_SLOT);
	//ss_d_set3DArray(d_hx[SUM_SLOT], nx, ny, nz, 0);
	//ss_d_set3DArray(d_hy[SUM_SLOT], nx, ny, nz, 0);
	//ss_d_set3DArray(d_hz[SUM_SLOT], nx, ny, nz, 0);

	for(int i=0; i<NSLOTS; i++)
	{
		if(slot_used[i])
		{
			sync_fields_hd(i);
			ss_d_add3DArray(d_hx[SUM_SLOT], nx, ny, nz, d_hx[SUM_SLOT], d_hx[i]);
			ss_d_add3DArray(d_hy[SUM_SLOT], nx, ny, nz, d_hy[SUM_SLOT], d_hy[i]);
			ss_d_add3DArray(d_hz[SUM_SLOT], nx, ny, nz, d_hz[SUM_SLOT], d_hz[i]);
		}
	}

	slot_used[SUM_SLOT] = true;
	new_device_fields[SUM_SLOT] = true;
}

bool SpinSystem::addFields(double mult, SpinSystem* addThis)
{
	if(nx != addThis->nx) return false;
	if(ny != addThis->ny) return false;
	if(nz != addThis->nz) return false;
	
	sync_fields_hd(SUM_SLOT);
	addThis->sync_fields_hd(SUM_SLOT);

	ss_d_scaleadd3DArray(d_hx[SUM_SLOT], nxyz, 1.0, d_hx[SUM_SLOT], mult, addThis->d_hx[SUM_SLOT]);
	ss_d_scaleadd3DArray(d_hy[SUM_SLOT], nxyz, 1.0, d_hy[SUM_SLOT], mult, addThis->d_hy[SUM_SLOT]);
	ss_d_scaleadd3DArray(d_hz[SUM_SLOT], nxyz, 1.0, d_hz[SUM_SLOT], mult, addThis->d_hz[SUM_SLOT]);

	
// 	for(int j=0; j<nxyz; j++)
// 	{
// 		hx[SUM_SLOT][j] += mult * addThis->hx[SUM_SLOT][j];
// 		hy[SUM_SLOT][j] += mult * addThis->hy[SUM_SLOT][j];
// 		hz[SUM_SLOT][j] += mult * addThis->hz[SUM_SLOT][j];
// 	}
	return true;
}

const char* SpinSystem::slotName(int index)
{
	if(index == EXCHANGE_SLOT)
		return "Exchange";
	if(index == ANISOTROPY_SLOT)
		return "Anisotropy";
	if(index == THERMAL_SLOT)
		return "Thermal";
	if(index == DIPOLE_SLOT)
		return "Dipole";
	if(index == APPLIEDFIELD_SLOT)
		return "Applied";
	if(index == SUM_SLOT)
		return "Total";
	if(index == SHORTRANGE_SLOT)
		return "ShortRange";
	return 0;
}

int SpinSystem::getSlot(const char* name)
{
	if(strcasecmp(name, "exchange") == 0)
		return EXCHANGE_SLOT;
	if(strcasecmp(name, "anisotropy") == 0)
		return ANISOTROPY_SLOT;
	if(strcasecmp(name, "thermal") == 0 || strcasecmp(name, "stochastic") == 0 || strcasecmp(name, "temperature") == 0)
		return THERMAL_SLOT;
	if(strcasecmp(name, "dipole") == 0 || strcasecmp(name, "magnetostatic") == 0  || strcasecmp(name, "magnetostatics") == 0)
		return DIPOLE_SLOT;
	if(strcasecmp(name, "short") == 0 || strcasecmp(name, "shortrange") == 0)
		return SHORTRANGE_SLOT;
	if(strcasecmp(name, "applied") == 0 || strcasecmp(name, "zeeman") == 0)
		return APPLIEDFIELD_SLOT;
	if(strcasecmp(name, "total") == 0 || strcasecmp(name, "sum") == 0)
		return SUM_SLOT;
	return -1;
}

// void SpinSystem::fft()
// {
// 	for(int i=0; i<nxyz; i++) rx[i] = x[i];
// 	for(int i=0; i<nxyz; i++) ry[i] = y[i];
// 	for(int i=0; i<nxyz; i++) rz[i] = z[i];
// 	
// 	for(int k=0; k<nz; k++)
// 	{
// 		fftw_execute_dft(r2q, 
// 			reinterpret_cast<fftw_complex*>(&rx[k*nx*ny]),
// 			reinterpret_cast<fftw_complex*>(&qx[k*nx*ny]));
// 		fftw_execute_dft(r2q, 
// 			reinterpret_cast<fftw_complex*>(&ry[k*nx*ny]),
// 			reinterpret_cast<fftw_complex*>(&qy[k*nx*ny]));
// 		fftw_execute_dft(r2q, 
// 			reinterpret_cast<fftw_complex*>(&rz[k*nx*ny]),
// 			reinterpret_cast<fftw_complex*>(&qz[k*nx*ny]));
// 	}
// 
// 	fft_time = time;
// }

void SpinSystem::zeroField(int i)
{
	slot_used[i] = false;
	
	ss_d_set3DArray(d_hx[i], nx, ny, nz, 0);
	ss_d_set3DArray(d_hy[i], nx, ny, nz, 0);
	ss_d_set3DArray(d_hz[i], nx, ny, nz, 0);

	//will need to fetch fields from device before we can use them here
	new_device_fields[i] = true;

}

void SpinSystem::zeroFields()
{
	for(int i=0; i<NSLOTS; i++)
	{
		if(slot_used[i])
			zeroField(i);
	}
}

bool SpinSystem::member(const int px, const int py, const int pz) const
{
	if(px < 0 || py < 0 || pz < 0)
		return false;

	if(px >= nx || py >= ny || pz >= nz)
		return false;
	
	return true;
}

void  SpinSystem::set(const int idx, double sx, double sy, double sz)
{
	sync_spins_dh();
	h_x[idx] = sx;
	h_y[idx] = sy;
	h_z[idx] = sz;

	h_ms[idx] = sqrt(sx*sx+sy*sy+sz*sz);
	
	new_host_spins = true;
}


void SpinSystem::set(const int px, const int py, const int pz, const double sx, const double sy, const double sz)
{
	if(!member(px, py, pz))
	{
		return; //out of bounds
	}
	
	const int idx = px + py*nx + pz*nx*ny;
	
	set(idx, sx, sy, sz);
}

int  SpinSystem::getidx(const int px, const int py, const int pz) const
{
	int x = px;
	int y = py;
	int z = pz;
	
	while(x >= nx) x-=nx;
	while(y >= ny) y-=ny;
	while(z >= nz) z-=nz;
	
	while(x < 0) x+=nx;
	while(y < 0) y+=ny;
	while(z < 0) z+=nz;
	
// 	const int x = CLAMP(px, nx-1);
// 	const int y = CLAMP(py, ny-1);
// 	const int z = CLAMP(pz, nz-1);
	
	return x + y*nx + z*nx*ny;
}


#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>

// return numspins * {<x>, <y>, <z>, <M>, <x^2>, <y^2>, <z^2>, <M^2>}
void SpinSystem::getNetMag(double* v8)
{

#if 1

	double* d_ws1;// = (double*)getWSMem(sizeof(double)*nxyz);

	const int sz = sizeof(double)*nxyz;
	getWSMem(&d_ws1, sz);

	
	sync_spins_hd();
	v8[0] = ss_reduce3DArray_sum(d_x, d_ws1, h_ws1, nx, ny, nz);
	v8[1] = ss_reduce3DArray_sum(d_y, d_ws1, h_ws1, nx, ny, nz);
	v8[2] = ss_reduce3DArray_sum(d_z, d_ws1, h_ws1, nx, ny, nz);

	v8[3] = sqrt(v8[0]*v8[0] + v8[1]*v8[1] + v8[2]*v8[2]);

	v8[4] = v8[0];
	v8[5] = v8[1];
	v8[6] = v8[2];
	v8[7] = v8[3];
	
	return;
#else
	sync_spins_dh();

	for(int i=0; i<8; i++)
		v8[i] = 0;
	
	for(int i=0; i<nxyz; i++)
	{
		v8[0] += h_x[i];
		v8[4] += h_x[i]*h_x[i];
	}
	for(int i=0; i<nxyz; i++)
	{
		v8[1] += h_y[i];
		v8[5] += h_y[i]*h_y[i];
	}
	for(int i=0; i<nxyz; i++)
	{
		v8[2] += h_z[i];
		v8[6] += h_z[i]*h_z[i];
	}

	v8[3] = sqrt(v8[0]*v8[0] + v8[1]*v8[1] + v8[2]*v8[2]);
	v8[7] = sqrt(v8[4]*v8[4] + v8[5]*v8[5] + v8[6]*v8[6]);
#endif
}













static int l_settimestep(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	ss->dt = lua_tonumber(L, 2);
	return 0;
}
static int l_gettimestep(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	lua_pushnumber(L, ss->dt);
	return 1;
}

static int l_setalpha(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	ss->alpha = lua_tonumber(L, 2);
	return 0;
}
static int l_getalpha(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	lua_pushnumber(L, ss->alpha);
	return 1;
}

static int l_setgamma(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	ss->gamma = lua_tonumber(L, 2);
	return 0;
}
static int l_getgamma(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	lua_pushnumber(L, ss->gamma);
	return 1;
}


static int l_netmag(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	
	double m = 1;
	if(lua_isnumber(L, 2))
		m = lua_tonumber(L, 2);

	double v8[8];
	
	ss->getNetMag(v8);

	for(int i=0; i<8; i++)
	{
		lua_pushnumber(L, v8[i]*m);
	}
	
	return 8;
}

static int l_setspin(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	
	int r1, r2;
	int site[3];
	double spin[3];
	
	r1 = lua_getNint(L, 3, site, 2, 1);
	if(r1 < 0)
		return luaL_error(L, "invalid site");
	
	r2 = lua_getNdouble(L, 3, spin, 2+r1, 0);
	if(r2 < 0)
		return luaL_error(L, "invalid spin");
	
	bool n = lua_isnumber(L, 2+r1+r2);
	
	
	int px = site[0] - 1;
	int py = site[1] - 1;
	int pz = site[2] - 1;
	
	double sx = spin[0];
	double sy = spin[1];
	double sz = spin[2];

	if(n)
	{
		double len = fabs(lua_tonumber(L, 2+r1+r2));
		double rr = sx*sx + sy*sy + sz*sz;
		if(rr > 0)
		{
			rr = len / sqrt(rr);
			sx *= rr;
			sy *= rr;
			sz *= rr;
		}
		else
		{
			sx = 0;
			sy = 0;
			sz = len;
		}
	}
	
	ss->set(px, py, pz, sx, sy, sz);
	
	return 0;
}

static int l_getspin(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	int site[3];
	
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
	{
		//try again
		site[0] = 1;
		site[1] = 1;
		site[2] = 1;
		
		for(int i=0; i<3; i++)
		{
			if(lua_isnumber(L, i+2))
				site[i] = lua_tointeger(L, i+2);
			else
				break;
		}
		
	}
	
	int px = site[0] - 1;
	int py = site[1] - 1;
	int pz = site[2] - 1;
	
	if(!ss->member(px, py, pz))
		return 0;
	
	int idx = ss->getidx(px, py, pz);
	
	ss->sync_spins_dh();
	lua_pushnumber(L, ss->h_x[idx]);
	lua_pushnumber(L, ss->h_y[idx]);
	lua_pushnumber(L, ss->h_z[idx]);
	
	double len2 = ss->h_x[idx]*ss->h_x[idx] 
				+ ss->h_y[idx]*ss->h_y[idx]
				+ ss->h_z[idx]*ss->h_z[idx];
	
	lua_pushnumber(L, sqrt(len2));
	
	return 4;
}

static int l_getunitspin(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	
	int site[3];
	
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	int px = site[0] - 1;
	int py = site[1] - 1;
	int pz = site[2] - 1;
	
	
	if(!ss->member(px, py, pz))
		return 0;
	

	int idx = ss->getidx(px, py, pz);

	ss->sync_spins_dh();
	if(ss->h_ms[idx] == 0)
	{
		lua_pushnumber(L, 1);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
		return 3;
	}
	
	double im = 1.0 / ss->h_ms[idx];

	lua_pushnumber(L, ss->h_x[idx]*im);
	lua_pushnumber(L, ss->h_y[idx]*im);
	lua_pushnumber(L, ss->h_z[idx]*im);
	
	return 3;
}

static int l_nx(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	lua_pushnumber(L, ss->nx);
	return 1;
}
static int l_ny(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	lua_pushnumber(L, ss->ny);
	return 1;
}
static int l_nz(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	lua_pushnumber(L, ss->nz);
	return 1;
}

static int l_sumfields(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	
	ss->sumFields();
	
	return 0;
}

static int l_zerofields(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	ss->zeroFields();

	return 0;
}

static int l_settime(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	ss->time = lua_tonumber(L, 2);

	return 0;
}
static int l_gettime(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	lua_pushnumber(L, ss->time);

	return 1;
}

static int l_tostring(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	
	lua_pushfstring(L, "SpinSystem (%dx%dx%d)", ss->nx, ss->ny, ss->nz);
	return 1;
}


static int l_netfield(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	
	const char* name = lua_tostring(L, 2);
	
	int slot = ss->getSlot(name);
	
	if(slot < 0)
		return luaL_error(L, "Unknown field type`%s'", name);
	

	double xyz[3] = {0,0,0};
	const int nxyz = ss->nxyz;
	
	ss->sync_fields_dh(slot); //from dev
	for(int i=0; i<nxyz; i++)
	{
		xyz[0] += ss->h_hx[slot][i];
		xyz[1] += ss->h_hy[slot][i];
		xyz[2] += ss->h_hz[slot][i];
	}
	
	
	lua_pushnumber(L, xyz[0] / ((double)nxyz));
	lua_pushnumber(L, xyz[1] / ((double)nxyz));
	lua_pushnumber(L, xyz[2] / ((double)nxyz));
	
	return 3;
}

static int l_getfield(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	const char* name = lua_tostring(L, 2);

	if(!name)
		return luaL_error(L, "First argument must a string");
	
	int site[3];
	int r = lua_getNint(L, 3, site, 3, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int x = site[0] - 1;
	const int y = site[1] - 1;
	const int z = site[2] - 1;
	
	int idx = ss->getidx(x, y, z);
	int slot = ss->getSlot(name);

	if(slot < 0)
		return luaL_error(L, "Unknown field type`%s'", name);


	ss->sync_fields_dh(slot);
	lua_pushnumber(L, ss->h_hx[slot][idx]);
	lua_pushnumber(L, ss->h_hy[slot][idx]);
	lua_pushnumber(L, ss->h_hz[slot][idx]);

	return 3;
}

static int l_addfields(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, dest, 1);
	
	SpinSystem* src = 0;
	double mult;
	
	if(lua_isnumber(L, 2))
	{
		mult = lua_tonumber(L, 2);
		src  = luaT_to<SpinSystem>(L, 3);
	}
	else
	{
		mult = 1.0;
		src  = luaT_to<SpinSystem>(L, 2);
	}
	if(!src) return 0;
	
	if(!dest->addFields(mult, src))
		return luaL_error(L, "Failed to sum fields");
	
	return 0;
}

static int l_copy(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	luaT_push<SpinSystem>(L, ss->copy(L));
	return 1;
}

static int l_copyto(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, src,  1);
	LUA_PREAMBLE(SpinSystem, dest, 2);

	if(!dest->copyFrom(L, src))
		return luaL_error(L, "Failed to copyTo");
	
	return 0;
}

static int l_copyfieldsto(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, src,  1);
	LUA_PREAMBLE(SpinSystem, dest, 2);

	if(!dest->copyFieldsFrom(L, src))
		return luaL_error(L, "Failed to copyTo");
	
	return 0;
}

static int l_copyfieldto(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, src,  1);

    const char* slotname = lua_tostring(L, 2);
    int i = src->getSlot(slotname);

	LUA_PREAMBLE(SpinSystem, dest, 3);

    if(i >= 0)
    {
		if(!dest->copyFieldFrom(L, src, i))
			return luaL_error(L, "Failed to copyTo");
    }
    else
    {
		return luaL_error(L, "Unknown field name");
    }
    return 0;
}


static int l_copyspinsto(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, src,  1);
	LUA_PREAMBLE(SpinSystem, dest, 2);

	if(!dest->copySpinsFrom(L, src))
		return luaL_error(L, "Failed to copyTo");
	
	return 0;
}

static int l_getextradata(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	int site[3];
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int px = site[0] - 1;
	const int py = site[1] - 1;
	const int pz = site[2] - 1;
	
	int idx = ss->getidx(px, py, pz);
	
	if(ss->extra_data[idx] < 0)
		lua_pushnil(L);
	else
		lua_rawgeti(L, LUA_REGISTRYINDEX, ss->extra_data[idx]);
	return 1;
}

static int l_setextradata(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	int site[3];
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int px = site[0] - 1;
	const int py = site[1] - 1;
	const int pz = site[2] - 1;
	
	int idx = ss->getidx(px, py, pz);
	
	if(ss->extra_data[idx] != LUA_REFNIL)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, ss->extra_data[idx]);
	}
	
	lua_pushvalue(L, r+2);
	
	ss->extra_data[idx] = luaL_ref(L, LUA_REGISTRYINDEX);
	ss->L = L;
	
	return 0;
}

static int l_getdiff(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, sa,  1);
	LUA_PREAMBLE(SpinSystem, sb,  2);
	
	double v4[4];
	
	sa->diff(sb, v4);
	for(int i=0; i<4; i++)
	{
		lua_pushnumber(L, v4[i]);
	}

	return 4;
}


int SpinSystem::help(lua_State* L)
{
	int i = 0;
	char buf[1024];
	buf[0] = 0;
	const char* field_types;
	do
	{
		field_types = SpinSystem::slotName(i); i++;
		if(field_types)
			sprintf(buf+strlen(buf), "\"%s\", ", field_types);
	}while(field_types);
	
	buf[strlen(buf)-2] = 0;
	
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Represents and contains a lattice of spins including orientation and resulting fields.");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expects zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);

	if(func == l_netmag)
	{
		lua_pushstring(L, "Calculate and return net magnetization of a spin system");
		lua_pushstring(L, "1 Optional Number: The return values will be multiplied by this number, default 1.");
		lua_pushstring(L, "8 numbers: mean(x), mean(y), mean(z), mean(M), mean(xx), mean(yy), mean(zz), mean(MM)");
		return 3;
	}
	
	if(func == l_netfield)
	{
		lua_pushstring(L, "Return average field due to an interaction. This field must be calculated with the appropriate operator.");
		lua_pushfstring(L, "1 String: The name of the field type to return. One of: %s", buf);
		lua_pushstring(L, "3 numbers: Vector representing the average field due to an interaction type.");
		return 3;
	}
	
	if(func == l_setspin)
	{
		lua_pushstring(L, "Set the orientation and magnitude of a spin at a site.");
		lua_pushstring(L, "2 *3Vector*s, 1 optional number: The first argument represents a lattice site. The second represents the spin vector. If the third argument is a number, the spin vector will be scaled to this length.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_getspin)
	{
		lua_pushstring(L, "Get the orientation and magnitude of a spin at a site.");
		lua_pushstring(L, "1 *3Vector*: The lattice site.");
		lua_pushstring(L, "1 *3Vector*: The spin vector at the lattice site.");
		return 3;
	}
	
	if(func == l_getunitspin)
	{
		lua_pushstring(L, "Get the orientation of a spin at a site.");
		lua_pushstring(L, "1 *3Vector*: The lattice site.");
		lua_pushstring(L, "1 *3Vector*: The spin normalized vector at the lattice site.");
		return 3;
	}
	
	if(func == l_nx)
	{
		lua_pushstring(L, "Get the first dimensions of the lattice.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size of the first dimension.");
		return 3;
	}
	
	if(func == l_ny)
	{
		lua_pushstring(L, "Get the second dimensions of the lattice.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size of the second dimension.");
		return 3;
	}
	
	if(func == l_nz)
	{
		lua_pushstring(L, "Get the third dimensions of the lattice.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size of the third dimension.");
		return 3;
	}
		
	if(func == l_sumfields)
	{
		lua_pushstring(L, "Sum all the fields into a single effective field.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_zerofields)
	{
		lua_pushstring(L, "Zero all the fields.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_settime)
	{
		lua_pushstring(L, "Set the time of the simulation.");
		lua_pushstring(L, "1 Number: New time for the simulation (default: 0).");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_gettime)
	{
		lua_pushstring(L, "Get the time of the simulation.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Time of the simulation.");
		return 3;
	}
	
	
	if(func == l_getfield)
	{
		lua_pushstring(L, "Get the field at a site due to an interaction");
		lua_pushfstring(L, "1 String, 1 *3Vector*: The first argument identifies the field interaction type, one of: %s. The second argument selects the lattice site.", buf);
		lua_pushstring(L, "3 Numbers: The field vector at the site.");
		return 3;
	}
	
	/*
	if(func == l_getinversespin)
	{
		lua_pushstring(L, "Return the an element of the Fourier Transform of the lattice.");
		lua_pushstring(L, "1 *3Vector*: The lattice site");
		lua_pushstring(L, "Table of Pairs: The s(q) value represented as a table of pairs. The table has 3 components representing x, y and z. Each component has 2 values representing the real and imaginary value. For example, the imaginary part of the x component would be at [1][2].");
		return 3;
	}*/
		
	if(func == l_addfields)
	{
		lua_pushstring(L, "Add fields from one *SpinSystem* to the current one, optionally scaling the field.");
		lua_pushstring(L, "1 Optional Number, 1 *SpinSystem*: The fields in the spin system are added to the calling spin system multiplied by the optional scaling value. This is useful when implementing higher order integrators.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_copy)
	{
		lua_pushstring(L, "Create a new copy of the spinsystem.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 *SpinSystem*");
// 		lua_pushstring(L, "Copy all aspects of the given *SpinSystem* to the calling system.");
// 		lua_pushstring(L, "1 *SpinSystem*: Source spin system.");
// 		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_copyto)
	{
		lua_pushstring(L, "Copy all aspects of the calling *SpinSystem* to the given system.");
		lua_pushstring(L, "1 *SpinSystem*: Destination spin system.");
		lua_pushstring(L, "");
		return 3;
	}

	
	if(func == l_copyspinsto)
	{
		lua_pushstring(L, "Copy spins of the calling *SpinSystem* to the given system.");
		lua_pushstring(L, "1 *SpinSystem*: Destination spin system.");
		lua_pushstring(L, "");
		return 3;
	}

	
	if(func == l_copyfieldsto)
	{
		lua_pushstring(L, "Copy fields of the calling *SpinSystem* to the given system.");
		lua_pushstring(L, "1 *SpinSystem*: Destination spin system.");
		lua_pushstring(L, "");
		return 3;
	}

        if(func == l_copyfieldto)
        {
	    lua_pushstring(L, "Copy a field type of the calling *SpinSystem* to the given system.");
	    lua_pushstring(L, "1 string, 1 *SpinSystem*: Field name, destination spin system.");
	    lua_pushstring(L, "");
	    return 3;
        }


	if(func == l_setalpha)
	{
		lua_pushstring(L, "Set the damping value for the spin system. This is used in *LLG* routines as well as *Thermal* calculations.");
		lua_pushstring(L, "1 Number: The damping value (default 1).");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_getalpha)
	{
		lua_pushstring(L, "Get the damping value for the spin system. This is used in *LLG* routines as well as *Thermal* calculations.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The damping value.");
		return 3;
	}

	if(func == l_settimestep)
	{
		lua_pushstring(L, "Set the time step for the spin system. This is used in *LLG* routines as well as *Thermal* calculations.");
		lua_pushstring(L, "1 Number: The time step.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_gettimestep)
	{
		lua_pushstring(L, "Get the time step for the spin system. This is used in *LLG* routines as well as *Thermal* calculations.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The time step.");
		return 3;
	}

	if(func == l_setgamma)
	{
		lua_pushstring(L, "Set the gamma value for the spin system. This is used in *LLG* routines.");
		lua_pushstring(L, "1 Number: The gamma value.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_getgamma)
	{
		lua_pushstring(L, "Get the gamma value for the spin system. This is used in *LLG* routines.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The gamma value.");
		return 3;
	}

	if(func == l_setextradata)
	{
		lua_pushstring(L, "Set site specific extra data. This may be used for book keeping during initialization.");
		lua_pushstring(L, "1 *3Vector*, 1 Value: Stores the value at the site specified. Implicit PBC.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_getextradata)
	{
		lua_pushstring(L, "Get site specific extra data. This may be used for book keeping during initialization.");
		lua_pushstring(L, "1 *3Vector*: Site position. Implicit PBC.");
		lua_pushstring(L, "1 Value: The value stored at this site position.");
		return 3;
	}

	if(func == l_getdiff)
	{
		lua_pushstring(L, "Compute the absolute difference between the current *SpinSystem* and a given *SpinSystem*. dx = Sum( |x[i] - other:x[i]|)");
		lua_pushstring(L, "1 *SpinSystem*: to compare against.");
		lua_pushstring(L, "4 Numbers: The differences in the x, y and z components and the length of the difference vector.");
		return 3;
	}

	return LuaBaseObject::help(L);
}



const luaL_Reg* SpinSystem::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		{"__tostring",   l_tostring},
		{"netMoment",    l_netmag},
		{"netField",     l_netfield},
		{"setSpin",      l_setspin},
		{"spin"   ,      l_getspin},
		{"unitSpin",     l_getunitspin},
		{"nx",           l_nx},
		{"ny",           l_ny},
		{"nz",           l_nz},
		{"sumFields",    l_sumfields},
		{"resetFields",  l_zerofields},
		{"setTime",      l_settime},
		{"time",         l_gettime},
		{"field",        l_getfield},
		//{"inverseSpin",  l_getinversespin},
		{"addFields",    l_addfields},
		{"copy",         l_copy},
		{"copyTo",       l_copyto},
		{"copySpinsTo",  l_copyspinsto},
		{"copyFieldsTo", l_copyfieldsto},
		{"copyFieldTo",  l_copyfieldto},
		{"setAlpha",     l_setalpha},
		{"alpha",        l_getalpha},
		{"setTimeStep",  l_settimestep},
		{"timeStep",     l_gettimestep},
		{"setGamma",     l_setgamma},
		{"gamma",        l_getgamma},
		{"setExtraData", l_setextradata},
		{"extraData",    l_getextradata},
		{"diff",         l_getdiff},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
