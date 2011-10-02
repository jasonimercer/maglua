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
#include "jthread.h"
#include "spinoperation.h"
#include <iostream>
#include <math.h>
#include <strings.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "luamigrate.h"
#include "spinsystem.hpp"

using namespace std;
#define CLAMP(x, m) ((x<0)?0:(x>m?m:x))

SpinSystem::SpinSystem(const int NX, const int NY, const int NZ)
	: Encodable(ENCODE_SPINSYSTEM), d_x(0), d_y(0), d_z(0), 
		d_ms(0), gamma(1.0), alpha(1.0), dt(1.0),
		nx(NX), ny(NY), nz(NZ), refcount(0),
		nslots(NSLOTS), time(0)
{
	init();
	L = 0;
}

void SpinSystem::sync_spins_dh(bool force)
{
	if(new_device_spins || force)
	{
		if(new_host_spins)
		{
			printf("(%s:%i) overwriting new host spins\n", __FILE__, __LINE__);
		}
	
		ss_copyDeviceToHost(h_x, d_x, nxyz);
		ss_copyDeviceToHost(h_y, d_y, nxyz);
		ss_copyDeviceToHost(h_z, d_z, nxyz);
		ss_copyDeviceToHost(h_ms,d_ms,nxyz);

		new_device_spins = false;
		new_host_spins = false;
	}
}

void SpinSystem::sync_fields_dh(int field, bool force)
{
	if(new_device_fields[field] || force)
	{
		ss_copyDeviceToHost(h_hx[field], d_hx[field], nxyz);
		ss_copyDeviceToHost(h_hy[field], d_hy[field], nxyz);
		ss_copyDeviceToHost(h_hz[field], d_hz[field], nxyz);
		
		new_device_fields[field] = false;
		new_host_fields[field] = false;
	}
}

void SpinSystem::sync_spins_hd(bool force)
{
	if(new_host_spins || force)
	{
		ss_copyHostToDevice(d_x, h_x, nxyz);
		ss_copyHostToDevice(d_y, h_y, nxyz);
		ss_copyHostToDevice(d_z, h_z, nxyz);
		ss_copyHostToDevice(d_ms,h_ms,nxyz);
		
		new_host_spins = false;
		new_device_spins = false;
	}	
}

void SpinSystem::sync_fields_hd(int field, bool force)
{
	if(new_host_fields[field] || force)
	{
		ss_copyHostToDevice(d_hx[field], h_hx[field], nxyz);
		ss_copyHostToDevice(d_hy[field], h_hy[field], nxyz);
		ss_copyHostToDevice(d_hz[field], h_hz[field], nxyz);

		new_host_fields[field] = false;
		new_device_fields[field] = false;
	}
}



SpinSystem::~SpinSystem()
{
	deinit();
}

// SpinSystem* SpinSystem::copy(lua_State* L)
// {
// 	SpinSystem* c = new SpinSystem(nx, ny, nz);
// 	
// 	c->copyFrom(L, this);
// 	
// 	return c;
// }

// void SpinSystem::diff(SpinSystem* other, double* v4)
// {
// 	if(	nx != other->nx ||
// 		ny != other->ny ||
// 		nz != other->nz)
// 	{
// 		v4[0] = 1e8;
// 		v4[1] = 1e8;
// 		v4[2] = 1e8;
// 		v4[3] = 1e8;
// 		return;
// 	}
// 	
// 	v4[0] = 0;
// 	v4[1] = 0;
// 	v4[2] = 0;
// 	
// 	const double* txyz[3] = {x,y,z};
// 	const double* oxyz[3] = {other->x,other->y,other->z};
// 	
// 	for(int j=0; j<3; j++)
// 	{
// 		for(int i=0; i<nxyz; i++)
// 		{
// // 			printf("%f\n", txyz[j][i]);
// 			v4[j] += fabs(txyz[j][i] - oxyz[j][i]);
// 		}
// 	}
// 	
// 	v4[3] = sqrt(v4[0]*v4[0] + v4[1]*v4[1] + v4[2]*v4[2]);
// }
// 
// 
// bool SpinSystem::copyFrom(lua_State* L, SpinSystem* src)
// {
// 	if(nx != src->nx) return false;
// 	if(ny != src->ny) return false;
// 	if(nz != src->nz) return false;
// 	
// 	memcpy(hx[SUM_SLOT], src->hx[SUM_SLOT], nxyz * sizeof(double));
// 	memcpy(hy[SUM_SLOT], src->hy[SUM_SLOT], nxyz * sizeof(double));
// 	memcpy(hz[SUM_SLOT], src->hz[SUM_SLOT], nxyz * sizeof(double));
// 	
// 	memcpy( x, src->x,  nxyz * sizeof(double));
// 	memcpy( y, src->y,  nxyz * sizeof(double));
// 	memcpy( z, src->z,  nxyz * sizeof(double));
// 	memcpy(ms, src->ms, nxyz * sizeof(double));
// 	
// 	alpha = src->alpha;
// 	gamma = src->gamma;
// 	dt = src->dt;
// 	
// 	fft_time = time - 1.0;
// 		
// 	// unref data - if exists
// 	for(int i=0; i<nxyz; i++)
// 	{
// 		if(extra_data[i] != LUA_REFNIL)
// 			luaL_unref(L, LUA_REGISTRYINDEX, extra_data[i]);
// 		extra_data[i] = LUA_REFNIL;
// 	}
// 	
// 	// make copies of references
// 	for(int i=0; i<nxyz; i++)
// 	{
// 		if(src->extra_data[i] != LUA_REFNIL)
// 		{
// 			lua_rawgeti(L, LUA_REGISTRYINDEX, src->extra_data[i]);
// 			extra_data[i] = luaL_ref(L, LUA_REGISTRYINDEX);
// 		}
// 	}
// 	
// 	return true;
// }
// 
// bool SpinSystem::copySpinsFrom(lua_State* L, SpinSystem* src)
// {
// 	if(nx != src->nx) return false;
// 	if(ny != src->ny) return false;
// 	if(nz != src->nz) return false;
// 	
// 	memcpy( x, src->x,  nxyz * sizeof(double));
// 	memcpy( y, src->y,  nxyz * sizeof(double));
// 	memcpy( z, src->z,  nxyz * sizeof(double));
// 	memcpy(ms, src->ms, nxyz * sizeof(double));
// 	
// 	fft_time = time - 1.0;
// 	
// 	return true;
// }
// 
// bool SpinSystem::copyFieldsFrom(lua_State* L, SpinSystem* src)
// {
// 	if(nx != src->nx) return false;
// 	if(ny != src->ny) return false;
// 	if(nz != src->nz) return false;
// 	
// 	memcpy(hx[SUM_SLOT], src->hx[SUM_SLOT], nxyz * sizeof(double));
// 	memcpy(hy[SUM_SLOT], src->hy[SUM_SLOT], nxyz * sizeof(double));
// 	memcpy(hz[SUM_SLOT], src->hz[SUM_SLOT], nxyz * sizeof(double));
// 	
// 	fft_time = time - 1.0;
// 	
// 	return true;
// }


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
		
		ss_d_free3DArray(d_x);
		ss_d_free3DArray(d_y);
		ss_d_free3DArray(d_z);
		ss_d_free3DArray(d_ms);
		
		ss_d_free3DArray(d_ws1);
		ss_d_free3DArray(d_ws2);
		ss_d_free3DArray(d_ws3);
		ss_d_free3DArray(d_ws4);
			
		ss_h_free3DArray(h_x);
		ss_h_free3DArray(h_y);
		ss_h_free3DArray(h_z);
		ss_h_free3DArray(h_ms);
		
		delete [] slot_used;
		
		for(int i=0; i<nslots; i++)
		{
			ss_d_free3DArray(d_hx[i]);
			ss_d_free3DArray(d_hy[i]);
			ss_d_free3DArray(d_hz[i]);

			ss_h_free3DArray(h_hx[i]);
			ss_h_free3DArray(h_hy[i]);
			ss_h_free3DArray(h_hz[i]);
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

	new_host_spins = false;
	new_device_spins = false;

	
	new_host_fields = new bool[NSLOTS]; 
	new_device_fields = new bool[NSLOTS]; 

	for(int i=0; i<NSLOTS; i++)
	{
		new_host_fields[i] = false;
		new_device_fields[i] = false;
	}

	ss_d_make3DArray(&d_ws1, nx, ny, nz);
	ss_d_make3DArray(&d_ws2, nx, ny, nz);
	ss_d_make3DArray(&d_ws3, nx, ny, nz);
	ss_d_make3DArray(&d_ws4, nx, ny, nz);

	ss_d_make3DArray(&d_x,  nx, ny, nz);
	ss_d_make3DArray(&d_y,  nx, ny, nz);
	ss_d_make3DArray(&d_z,  nx, ny, nz);
	ss_d_make3DArray(&d_ms, nx, ny, nz);

	ss_h_make3DArray(&h_x,  nx, ny, nz);
	ss_h_make3DArray(&h_y,  nx, ny, nz);
	ss_h_make3DArray(&h_z,  nx, ny, nz);
	ss_h_make3DArray(&h_ms, nx, ny, nz);
	
	//set spins to (0,0,0)
	// setting them on the device
	ss_d_set3DArray(d_x, nx, ny, nz, 0);
	ss_d_set3DArray(d_y, nx, ny, nz, 0);
	ss_d_set3DArray(d_z, nx, ny, nz, 0);
	ss_d_set3DArray(d_ms, nx, ny, nz, 0);
	new_device_spins = true;
	sync_spins_dh();
	
	
	d_hx = new double* [nslots];
	d_hy = new double* [nslots];
	d_hz = new double* [nslots];
	
	h_hx = new double* [nslots];
	h_hy = new double* [nslots];
	h_hz = new double* [nslots];
	
	slot_used = new bool[nslots];

	extra_data = new int[nxyz];
	for(int i=0; i<nxyz; i++)
		extra_data[i] = LUA_REFNIL;
	
	for(int i=0; i<NSLOTS; i++)
	{
		ss_d_make3DArray(&(d_hx[i]), nx, ny, nz);
		ss_d_make3DArray(&(d_hy[i]), nx, ny, nz);
		ss_d_make3DArray(&(d_hz[i]), nx, ny, nz);

		ss_h_make3DArray(&(h_hx[i]), nx, ny, nz);
		ss_h_make3DArray(&(h_hy[i]), nx, ny, nz);
		ss_h_make3DArray(&(h_hz[i]), nx, ny, nz);
		
		new_host_fields[i] = true; //this will also get set in zeroFields but we're
								   // doing it here to remind ourselves of the pattern
		zeroField(i);
		sync_fields_dh(i);
	}
}

//   void encodeBuffer(const void* s, int len, buffer* b);
//   void encodeDouble(const double d, buffer* b);
//   void encodeInteger(const int i, buffer* b);
//    int decodeInteger(const char* buf, int* pos);
// double decodeDouble(const char* buf, int* pos);

// int SpinSystem::start_thread(int idx, void *(*start_routine)(void*), void* arg)
// {
// 	if(!jthreads[idx])
// 	{
// 		jthreads[idx] = new JThread(start_routine, arg);
// 		jthreads[idx]->start();
// 	}
// // 	else
// // 	{
// // 		
// // 	}
// 	
// }

#warning encode/decode are empty
void SpinSystem::encode(buffer* b) const {}
int  SpinSystem::decode(buffer* b) {}

#if 0
void SpinSystem::encode(buffer* b) const
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);

	encodeDouble(alpha, b);
	encodeDouble(   dt, b);
	encodeDouble(gamma, b);

	encodeDouble(time, b);
	
	for(int i=0; i<nxyz; i++)
	{
		encodeDouble(x[i], b);
		encodeDouble(y[i], b);
		encodeDouble(z[i], b);
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
	double r, i;
	
	deinit();
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	
	alpha = decodeDouble(b);
	dt = decodeDouble(b);
	gamma = decodeDouble(b);

	time = decodeDouble(b);
	init();

	for(int j=0; j<nxyz; j++)
	{
		x[j] = decodeDouble(b);
		y[j] = decodeDouble(b);
		z[j] = decodeDouble(b);
		ms[j] = sqrt(x[j]*x[j]+y[j]*y[j]+z[j]*z[j]);
	}
	


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
#endif



void SpinSystem::sumFields()
{
// 	for(int i=0; i<NSLOTS; i++)
// 	{
// 		if(jthreads[i])
// 		{
// 			jthreads[i]->join();
// 			delete jthreads[i];
// 			jthreads[i] = 0;
// 		}
// 	}

	ss_d_set3DArray(d_hx[SUM_SLOT], nx, ny, nz, 0);
	ss_d_set3DArray(d_hy[SUM_SLOT], nx, ny, nz, 0);
	ss_d_set3DArray(d_hz[SUM_SLOT], nx, ny, nz, 0);

	for(int i=1; i<NSLOTS; i++)
	{
		if(slot_used[i])
		{
			ss_d_add3DArray(d_hx[SUM_SLOT], nx, ny, nz, d_hx[SUM_SLOT], d_hx[i]);
			ss_d_add3DArray(d_hy[SUM_SLOT], nx, ny, nz, d_hy[SUM_SLOT], d_hy[i]);
			ss_d_add3DArray(d_hz[SUM_SLOT], nx, ny, nz, d_hz[SUM_SLOT], d_hz[i]);
		}
	}
	
#if 0
#if 0
	vector<int> v;
	for(int i=1; i<NSLOTS; i++)
	{
		if(slot_used[i])
			v.push_back(i);
	}
	unsigned int n = v.size();
	
	double* sumx = hx[SUM_SLOT];
	double* sumy = hy[SUM_SLOT];
	double* sumz = hz[SUM_SLOT];
	
	#pragma omp parallel shared(sumx, sumy, sumz)
	{
		#pragma omp for nowait
		for(int i=0; i<nxyz; i++)
			sumx[i] = 0;
		#pragma omp for nowait
		for(int i=0; i<nxyz; i++)
			sumy[i] = 0;
		#pragma omp for nowait
		for(int i=0; i<nxyz; i++)
			sumz[i] = 0;
		#pragma omp barrier
	
		#pragma omp for
		for(unsigned int i=0; i<n; i++)
		{
			const int k = v[i];
			for(int j=0; j<nxyz; j++)
				sumx[j] += hx[k][j];
			for(int j=0; j<nxyz; j++)
				sumy[j] += hx[k][j];
			for(int j=0; j<nxyz; j++)
				sumz[j] += hx[k][j];
			
		}
	}
	
#else
	for(int j=0; j<nxyz; j++)
	{
		hx[SUM_SLOT][j] = hx[1][j];
		for(int i=2; i<NSLOTS; i++)
			hx[SUM_SLOT][j] += hx[i][j];

		hy[SUM_SLOT][j] = hy[1][j];
		for(int i=2; i<NSLOTS; i++)
			hy[SUM_SLOT][j] += hy[i][j];

		hz[SUM_SLOT][j] = hz[1][j];
		for(int i=2; i<NSLOTS; i++)
			hz[SUM_SLOT][j] += hz[i][j];
	}
#endif	
// 	#pragma omp parallel for
// 	for(int j=0; j<nxyz; j++)
// 	{
// 		hx[SUM_SLOT][j] = hx[1][j];
// 		for(int i=2; i<NSLOTS; i++)
// 			hx[SUM_SLOT][j] += hx[i][j];
// 
// 		hy[SUM_SLOT][j] = hy[1][j];
// 		for(int i=2; i<NSLOTS; i++)
// 			hy[SUM_SLOT][j] += hy[i][j];
// 
// 		hz[SUM_SLOT][j] = hz[1][j];
// 		for(int i=2; i<NSLOTS; i++)
// 			hz[SUM_SLOT][j] += hz[i][j];
// 	}
#endif	
}
/*
bool SpinSystem::addFields(double mult, SpinSystem* addThis)
{
	if(nx != addThis->nx) return false;
	if(ny != addThis->ny) return false;
	if(nz != addThis->nz) return false;
	
	for(int j=0; j<nxyz; j++)
	{
		hx[SUM_SLOT][j] += mult * addThis->hx[SUM_SLOT][j];
		hy[SUM_SLOT][j] += mult * addThis->hy[SUM_SLOT][j];
		hz[SUM_SLOT][j] += mult * addThis->hz[SUM_SLOT][j];
	}
	return true;
}
*/
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
	if(strcasecmp(name, "dipole") == 0)
		return DIPOLE_SLOT;
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

	
// 	for(int j=0; j<nxyz; j++)
// 	{
// 		h_hx[i][j] = 0;
// 		h_hy[i][j] = 0;
// 		h_hz[i][j] = 0;
// 	}

	//will need to fetch fields from device before we can use them here
	new_device_fields[i] = true;
}

void SpinSystem::zeroFields()
{
	for(int i=0; i<NSLOTS; i++)
	{
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
	
	sync_spins_dh(); // get latest spin state, we don't want to clobber it with old local state
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

// return numspins * {<x>, <y>, <z>, <M>, <x^2>, <y^2>, <z^2>, <M^2>}
void SpinSystem::getNetMag(double* v8)
{
	sync_spins_dh();
	
	for(int i=0; i<8; i++)
		v8[i] = 0;
	
	for(int i=0; i<nxyz; i++)
	{
		v8[0] += h_x[i];
		v8[4] += h_x[i]*h_x[i];
		v8[1] += h_y[i];
		v8[5] += h_y[i]*h_y[i];
		v8[2] += h_z[i];
		v8[6] += h_z[i]*h_z[i];
	}

	v8[3] = sqrt(v8[0]*v8[0] + v8[1]*v8[1] + v8[2]*v8[2]);
	v8[7] = sqrt(v8[4]*v8[4] + v8[5]*v8[5] + v8[6]*v8[6]);
}
















SpinSystem* checkSpinSystem(lua_State* L, int idx)
{
	SpinSystem** pp = (SpinSystem**)luaL_checkudata(L, idx, "MERCER.spinsystem");
    luaL_argcheck(L, pp != NULL, 1, "`SpinSystem' expected");
    return *pp;
}

void lua_pushSpinSystem(lua_State* L, SpinSystem* ss)
{
	ss->refcount++;
	ss->L = L;
	
	SpinSystem** pp = (SpinSystem**)lua_newuserdata(L, sizeof(SpinSystem**));
	
	*pp = ss;
	luaL_getmetatable(L, "MERCER.spinsystem");
	lua_setmetatable(L, -2);
}


int l_ss_new(lua_State* L)
{
	int nx, ny, nz;
	int n[3];
	int r = lua_getNint(L, 3, n, 1, 1);
	
	if(r < 0)
	{
		//fill out with ones
		for(int i=0; i<3; i++)
		{
			if(lua_isnumber(L, i+1))
				n[i] = lua_tointeger(L, i+1);
			else
				n[i] = 1;
		}
	}
	
	nx = n[0];
	ny = n[1];
	nz = n[2];

	
	lua_pushSpinSystem(L, new SpinSystem(nx, ny, nz));
	
	return 1;
}

int l_ss_gc(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
	ss->refcount--;
	if(ss->refcount == 0)
	{
		for(int i=0; i<ss->nxyz; i++)
		{
			if(ss->extra_data[i] != -1)
				luaL_unref(L, LUA_REGISTRYINDEX, ss->extra_data[i]);
		}
		delete ss;
	}
	return 0;
}


int l_ss_settimestep(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	ss->dt = lua_tonumber(L, 2);
	return 0;
}
int l_ss_gettimestep(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	lua_pushnumber(L, ss->dt);
	return 1;
}

int l_ss_setalpha(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	ss->alpha = lua_tonumber(L, 2);
	return 0;
}
int l_ss_getalpha(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	lua_pushnumber(L, ss->alpha);
	return 1;
}

int l_ss_setgamma(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	ss->gamma = lua_tonumber(L, 2);
	return 0;
}
int l_ss_getgamma(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	lua_pushnumber(L, ss->gamma);
	return 1;
}


int l_ss_netmag(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
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

int l_ss_setspin(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
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

int l_ss_getspin(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;

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

int l_ss_getunitspin(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
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

int l_ss_nx(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	lua_pushnumber(L, ss->nx);
	return 1;
}
int l_ss_ny(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	lua_pushnumber(L, ss->ny);
	return 1;
}
int l_ss_nz(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	lua_pushnumber(L, ss->nz);
	return 1;
}

int l_ss_sumfields(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
	ss->sumFields();
	
	return 0;
}

int l_ss_zerofields(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;

	ss->zeroFields();

	return 0;
}

int l_ss_settime(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;

	ss->time = lua_tonumber(L, 2);

	return 0;
}
int l_ss_gettime(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;

	lua_pushnumber(L, ss->time);

	return 1;
}

static int l_ss_tostring(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
	lua_pushfstring(L, "SpinSystem (%dx%dx%d)", ss->nx, ss->ny, ss->nz);
	return 1;
}

int l_ss_netfield(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
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

int l_ss_getfield(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;

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

// int l_ss_addfields(lua_State* L)
// {
// 	SpinSystem* dest = checkSpinSystem(L, 1);
// 	if(!dest) return 0;
// 	
// 	SpinSystem* src = 0;
// 	double mult;
// 	
// 	if(lua_isnumber(L, 2))
// 	{
// 		mult = lua_tonumber(L, 2);
// 		src  = checkSpinSystem(L, 3);
// 	}
// 	else
// 	{
// 		mult = 1.0;
// 		src  = checkSpinSystem(L, 2);
// 	}
// 	if(!src) return 0;
// 	
// 	if(!dest->addFields(mult, src))
// 		return luaL_error(L, "Failed to sum fields");
// 	
// 	return 0;
// }

// int l_ss_copy(lua_State* L)
// {
// 	SpinSystem* src = checkSpinSystem(L, 1);
// 	if(!src) return 0;
// 	
// 	lua_pushSpinSystem(L, src->copy(L));
// 	return 1;
// }

// int l_ss_copyto(lua_State* L)
// {
// 	SpinSystem* src = checkSpinSystem(L, 1);
// 	if(!src) return 0;
// 	
// 	SpinSystem* dest = checkSpinSystem(L, 2);
// 	if(!dest) return 0;
// 
// 	if(!dest->copyFrom(L, src))
// 		return luaL_error(L, "Failed to copyTo");
// 	
// 	return 0;
// }

// int l_ss_copyfieldsto(lua_State* L)
// {
// 	SpinSystem* src = checkSpinSystem(L, 1);
// 	if(!src) return 0;
// 	
// 	SpinSystem* dest = checkSpinSystem(L, 2);
// 	if(!dest) return 0;
// 
// 	if(!dest->copyFieldsFrom(L, src))
// 		return luaL_error(L, "Failed to copyTo");
// 	
// 	return 0;
// }

// int l_ss_copyspinsto(lua_State* L)
// {
// 	SpinSystem* src = checkSpinSystem(L, 1);
// 	if(!src) return 0;
// 	
// 	SpinSystem* dest = checkSpinSystem(L, 2);
// 	if(!dest) return 0;
// 
// 	if(!dest->copySpinsFrom(L, src))
// 		return luaL_error(L, "Failed to copyTo");
// 	
// 	return 0;
// }

static int l_ss_getextradata(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;

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

static int l_ss_setextradata(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;

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


// int l_ss_getinversespin(lua_State* L)
// {
// 	SpinSystem* ss = checkSpinSystem(L, 1);
// 	if(!ss) return 0;
// 	
// 	if(ss->time != ss->fft_time)
// 		ss->fft();
// 
// 	int site[3];
// 	int r = lua_getNint(L, 3, site, 2, 1);
// 	if(r < 0)
// 		return luaL_error(L, "invalid site");
// 	
// 	const int px = site[0] - 1;
// 	const int py = site[1] - 1;
// 	const int pz = site[2] - 1;
// 	
// 	if(!ss->member(px, py, pz))
// 		return luaL_error(L, "(%d %d %d) is not a member of the system", px+1, py+1, pz+1);
// 	
// 	int idx = ss->getidx(px, py, pz);
// 	
// 	lua_newtable(L);
// 	lua_pushinteger(L, 1);
// 	lua_pushnumber(L, real(ss->qx[idx]));
// 	lua_settable(L, -3);
// 	lua_pushinteger(L, 2);
// 	lua_pushnumber(L, imag(ss->qx[idx]));
// 	lua_settable(L, -3);
// 	
// 	lua_newtable(L);
// 	lua_pushinteger(L, 1);
// 	lua_pushnumber(L, real(ss->qy[idx]));
// 	lua_settable(L, -3);
// 	lua_pushinteger(L, 2);
// 	lua_pushnumber(L, imag(ss->qy[idx]));
// 	lua_settable(L, -3);
// 	
// 	lua_newtable(L);
// 	lua_pushinteger(L, 1);
// 	lua_pushnumber(L, real(ss->qz[idx]));
// 	lua_settable(L, -3);
// 	lua_pushinteger(L, 2);
// 	lua_pushnumber(L, imag(ss->qz[idx]));
// 	lua_settable(L, -3);
// 	
// 	return 3;
// }

// static int l_ss_getdiff(lua_State* L)
// {
// 	SpinSystem* sa = checkSpinSystem(L, 1);
// 	if(!sa) return 0;
// 	
// 	SpinSystem* sb = checkSpinSystem(L, 2);
// 	if(!sb) return 0;
// 	
// 	double v4[4];
// 	
// 	sa->diff(sb, v4);
// 	for(int i=0; i<4; i++)
// 	{
// 		lua_pushnumber(L, v4[i]);
// 	}
// 
// 	return 4;
// }

static int l_ss_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.spinsystem");
	return 1;
}

static int l_ss_help(lua_State* L)
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
		lua_pushstring(L, ""); //input, empty
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
	
	if(func == l_ss_new)
	{
		lua_pushstring(L, "Create a new Spin System.");
		lua_pushstring(L, "1 *3Vector*: The width, depth and number of layers for a spin system. Omitted parameters are assumed to be 1."); 
		lua_pushstring(L, "1 Spin System");
		return 3;
	}
	
// 	if(func == l_ss_netmag)
// 	{
// 		lua_pushstring(L, "Calculate and return net magnetization of a spin system");
// 		lua_pushstring(L, "1 Optional Number: The return values will be multiplied by this number, default 1.");
// 		lua_pushstring(L, "8 numbers: mean(x), mean(y), mean(z), mean(M), mean(xx), mean(yy), mean(zz), mean(MM)");
// 		return 3;
// 	}
	
	if(func == l_ss_netfield)
	{
		lua_pushstring(L, "Return average field due to an interaction. This field must be calculated with the appropriate operator.");
		lua_pushfstring(L, "1 String: The name of the field type to return. One of: %s", buf);
		lua_pushstring(L, "3 numbers: Vector representing the average field due to an interaction type.");
		return 3;
	}
	
	if(func == l_ss_setspin)
	{
		lua_pushstring(L, "Set the orientation and magnitude of a spin at a site.");
		lua_pushstring(L, "2 *3Vector*s, 1 optional number: The first argument represents a lattice site. The second represents the spin vector. If the third argument is a number, the spin vector will be scaled to this length.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ss_getspin)
	{
		lua_pushstring(L, "Get the orientation and magnitude of a spin at a site.");
		lua_pushstring(L, "1 *3Vector*: The lattice site.");
		lua_pushstring(L, "1 *3Vector*: The spin vector at the lattice site.");
		return 3;
	}
	
	if(func == l_ss_getunitspin)
	{
		lua_pushstring(L, "Get the orientation of a spin at a site.");
		lua_pushstring(L, "1 *3Vector*: The lattice site.");
		lua_pushstring(L, "1 *3Vector*: The spin normalized vector at the lattice site.");
		return 3;
	}
	
	if(func == l_ss_nx)
	{
		lua_pushstring(L, "Get the first dimensions of the lattice.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size of the first dimension.");
		return 3;
	}
	
	if(func == l_ss_ny)
	{
		lua_pushstring(L, "Get the second dimensions of the lattice.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size of the second dimension.");
		return 3;
	}
	
	if(func == l_ss_nz)
	{
		lua_pushstring(L, "Get the third dimensions of the lattice.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size of the third dimension.");
		return 3;
	}
	
	
	if(func == l_ss_sumfields)
	{
		lua_pushstring(L, "Sum all the fields into a single effective field.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ss_zerofields)
	{
		lua_pushstring(L, "Zero all the fields.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ss_settime)
	{
		lua_pushstring(L, "Set the time of the simulation.");
		lua_pushstring(L, "1 Number: New time for the simulation (default: 0).");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ss_gettime)
	{
		lua_pushstring(L, "Get the time of the simulation.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Time of the simulation.");
		return 3;
	}
	
	
	if(func == l_ss_getfield)
	{
		lua_pushstring(L, "Get the field at a site due to an interaction");
		lua_pushfstring(L, "1 String, 1 *3Vector*: The first argument identifies the field interaction type, one of: %s. The second argument selects the lattice site.", buf);
		lua_pushstring(L, "3 Numbers: The field vector at the site.");
		return 3;
	}
	
// 	if(func == l_ss_getinversespin)
// 	{
// 		lua_pushstring(L, "Return the an element of the Fourier Transform of the lattice.");
// 		lua_pushstring(L, "1 *3Vector*: The lattice site");
// 		lua_pushstring(L, "Table of Pairs: The s(q) value represented as a table of pairs. The table has 3 components representing x, y and z. Each component has 2 values representing the real and imaginary value. For example, the imaginary part of the x component would be at [1][2].");
// 		return 3;
// 	}
		
// 	if(func == l_ss_addfields)
// 	{
// 		lua_pushstring(L, "Add fields from one *SpinSystem* to the current one, optionally scaling the field.");
// 		lua_pushstring(L, "1 Optional Number, 1 *SpinSystem*: The fields in the spin system are added to the calling spin system multiplied by the optional scaling value. This is useful when implementing higher order integrators.");
// 		lua_pushstring(L, "");
// 		return 3;
// 	}

// 	if(func == l_ss_copy)
// 	{
// 		lua_pushstring(L, "Create a new copy of the spinsystem.");
// 		lua_pushstring(L, "");
// 		lua_pushstring(L, "1 *SpinSystem*");
// // 		lua_pushstring(L, "Copy all aspects of the given *SpinSystem* to the calling system.");
// // 		lua_pushstring(L, "1 *SpinSystem*: Source spin system.");
// // 		lua_pushstring(L, "");
// 		return 3;
// 	}
// 	
// 	if(func == l_ss_copyto)
// 	{
// 		lua_pushstring(L, "Copy all aspects of the calling *SpinSystem* to the given system.");
// 		lua_pushstring(L, "1 *SpinSystem*: Destination spin system.");
// 		lua_pushstring(L, "");
// 		return 3;
// 	}
// 
// 	
// 	if(func == l_ss_copyspinsto)
// 	{
// 		lua_pushstring(L, "Copy spins of the calling *SpinSystem* to the given system.");
// 		lua_pushstring(L, "1 *SpinSystem*: Destination spin system.");
// 		lua_pushstring(L, "");
// 		return 3;
// 	}
// 
// 	
// 	if(func == l_ss_copyfieldsto)
// 	{
// 		lua_pushstring(L, "Copy fields of the calling *SpinSystem* to the given system.");
// 		lua_pushstring(L, "1 *SpinSystem*: Destination spin system.");
// 		lua_pushstring(L, "");
// 		return 3;
// 	}

	if(func == l_ss_setalpha)
	{
		lua_pushstring(L, "Set the damping value for the spin system. This is used in *LLG* routines as well as *Thermal* calculations.");
		lua_pushstring(L, "1 Number: The damping value (default 1).");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_ss_getalpha)
	{
		lua_pushstring(L, "Get the damping value for the spin system. This is used in *LLG* routines as well as *Thermal* calculations.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The damping value.");
		return 3;
	}

	if(func == l_ss_settimestep)
	{
		lua_pushstring(L, "Set the time step for the spin system. This is used in *LLG* routines as well as *Thermal* calculations.");
		lua_pushstring(L, "1 Number: The time step.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_ss_gettimestep)
	{
		lua_pushstring(L, "Get the time step for the spin system. This is used in *LLG* routines as well as *Thermal* calculations.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The time step.");
		return 3;
	}

	if(func == l_ss_setgamma)
	{
		lua_pushstring(L, "Set the gamma value for the spin system. This is used in *LLG* routines.");
		lua_pushstring(L, "1 Number: The gamma value.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_ss_getgamma)
	{
		lua_pushstring(L, "Get the gamma value for the spin system. This is used in *LLG* routines.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The gamma value.");
		return 3;
	}

	if(func == l_ss_setextradata)
	{
		lua_pushstring(L, "Set site specific extra data. This may be used for book keeping during initialization.");
		lua_pushstring(L, "1 *3Vector*, 1 Value: Stores the value at the site specified. Implicit PBC.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_ss_getextradata)
	{
		lua_pushstring(L, "Get site specific extra data. This may be used for book keeping during initialization.");
		lua_pushstring(L, "1 *3Vector*: Site position. Implicit PBC.");
		lua_pushstring(L, "1 Value: The value stored at this site position.");
		return 3;
	}

// 	if(func == l_ss_getdiff)
// 	{
// 		lua_pushstring(L, "Compute the absolute difference between the current *SpinSystem* and a given *SpinSystem*. dx = Sum( |x[i] - other:x[i]|)");
// 		lua_pushstring(L, "1 *SpinSystem*: to compare against.");
// 		lua_pushstring(L, "4 Numbers: The differences in the x, y and z components and the length of the difference vector.");
// 		return 3;
// 	}




	return 0;
}


void registerSpinSystem(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_ss_gc},
		{"__tostring",   l_ss_tostring},
// 		{"netMoment",    l_ss_netmag},
		{"netField",     l_ss_netfield},
		{"setSpin",      l_ss_setspin},
		{"spin"   ,      l_ss_getspin},
		{"unitSpin",     l_ss_getunitspin},
		{"nx",           l_ss_nx},
		{"ny",           l_ss_ny},
		{"nz",           l_ss_nz},
		{"sumFields",    l_ss_sumfields},
//		{"zeroFields",   l_ss_zerofields},
		{"resetFields",  l_ss_zerofields},
		{"setTime",      l_ss_settime},
		{"time",         l_ss_gettime},
		{"field",        l_ss_getfield},
//		{"getField",     l_ss_getfield},
// 		{"inverseSpin",  l_ss_getinversespin},
// 		{"addFields",    l_ss_addfields},
// 		{"copy",         l_ss_copy},
// 		{"copyTo",       l_ss_copyto},
// 		{"copySpinsTo",  l_ss_copyspinsto},
// 		{"copyFieldsTo", l_ss_copyfieldsto},
		{"setAlpha",     l_ss_setalpha},
		{"alpha",        l_ss_getalpha},
		{"setTimeStep",  l_ss_settimestep},
		{"timeStep",     l_ss_gettimestep},
		{"setGamma",     l_ss_setgamma},
		{"gamma",        l_ss_getgamma},
		{"setExtraData", l_ss_setextradata},
		{"extraData",    l_ss_getextradata},
// 		{"diff",         l_ss_getdiff},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.spinsystem");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_ss_new},
		{"help",                l_ss_help},
		{"metatable",           l_ss_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "SpinSystem", functions);
	lua_pop(L,1);
}
