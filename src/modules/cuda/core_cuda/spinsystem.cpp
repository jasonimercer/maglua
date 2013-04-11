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

using namespace std;
#define CLAMP(x, m) ((x<0)?0:(x>m?m:x))

SpinSystem::SpinSystem(const int NX, const int NY, const int NZ)
	: LuaBaseObject(hash32(slineage(0))), x(0), y(0), z(0),
		ms(0),alpha(1.0),  gamma(1.0), dt(1.0),
		nx(NX), ny(NY), nz(NZ),
		nslots(4), time(0)
{
	registerWS();
	site_alpha = 0;
	site_gamma = 0;
	L = 0;
    init();
	
}

int SpinSystem::luaInit(lua_State* L)
{
	LuaBaseObject::luaInit(L);
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


SpinSystem::~SpinSystem()
{
	deinit();
	unregisterWS();
}


SpinSystem* SpinSystem::copy(lua_State* L)
{
	SpinSystem* c = new SpinSystem(nx, ny, nz);
	
	c->copyFrom(L, this);
	
	return c;
}

bool SpinSystem::sameSize(const SpinSystem* other) const
{
	return 	nx == other->nx && 
			ny == other->ny && 
			nz == other->nz;
}

static void cross3(
	const double ax, const double ay, const double az,
	const double bx, const double by, const double bz,
	      double&cx,       double&cy,       double&cz)
{
	cx = ay*bz - az*by;
	cy = az*bx - ax*bz;
	cz = ax*by - ay*bx;
}

static double dot3(
	const double ax, const double ay, const double az,
	const double bx, const double by, const double bz)
{
	return ax*bx + ay*by + az*bz;
}

static double random_double(const double a=0, const double b=1)
{
	return a + (b-a) * ((double)rand()) / ((double)RAND_MAX);
}

static void rotateSpinToward(double& x, double& y, double& z, const double l1, const double gx, const double gy, const double gz, const double l2, const double max_theta)
{
	const double l12 = l1 * l2; 
	if(l12 == 0)
	{
		x = 0;
		y = 0;
		z = 0;
		return;
	}

	const double dotProduct = dot3(x,y,z,   gx,gy,gz);

	double normalizedDotProduct = dotProduct / (l12);

	if(normalizedDotProduct > 1)
		normalizedDotProduct = 1;
	
	if(acos(normalizedDotProduct) <= max_theta)
	{
		x = gx;
		y = gy;
		z = gz;
		return;
	}

	if(acos(normalizedDotProduct) <= -max_theta) //rotate away
	{
		x = -gx;
		y = -gy;
		z = -gz;
		return;
	}

	// now we need a vector ortho to the inputs to rotate about

	double nx, ny, nz;

	if(normalizedDotProduct == -1) //then colinear, choose a random vector
	{
		int rand(void);

		do
		{
			cross3(x, y, z, 
				  random_double(-1,1), 
				  random_double(-1,1), 
				  random_double(-1,1), nx, ny, nz);
		}while(dot3(nx,nx,ny,ny,nz,nz) == 0);
	}
	else
	{
		cross3(x, y, z, gx, gy, gz, nx, ny, nz);
	}


	const double n2 = dot3(nx,ny,nz,   nx,ny,nz);
	const double in = 1.0/sqrt(n2);
	
	nx *= in;
	ny *= in;
	nz *= in;

	// now we have a unit vector ortho to source and destination
	// we need to rotate about it by max_theta to get our goal

	const double _x = x;
	const double _y = y;
	const double _z = z;

	const double _u = nx;
	const double _v = ny;
	const double _w = nz;

	const double cost = cos(max_theta);
	const double sint = sin(max_theta);
	const double ux_vy_wz =  _u*_x+_v*_y+_w*_z;

	x = (_u*(ux_vy_wz)+(_x*(_v*_v+_w*_w)-_u*(_v*_y+_w*_z))*cost + (-_w*_y+_v*_z)*sint);
	y = (_v*(ux_vy_wz)+(_y*(_u*_u+_w*_w)-_v*(_u*_x+_w*_z))*cost + ( _w*_x-_u*_z)*sint);
    z = (_w*(ux_vy_wz)+(_z*(_u*_u+_v*_v)-_w*(_u*_x+_v*_y))*cost + (-_v*_x+_u*_y)*sint);
}


void SpinSystem::rotateToward(SpinSystem* other, double max_angle, dArray* max_by_site)
{
    if(!sameSize(other))
        return;

    if(max_by_site == 0)
    {
        double* xx = x->data();
        double* yy = y->data();
        double* zz = z->data();
        double* mm = ms->data();

        double* ox = other->x->data();
        double* oy = other->y->data();
        double* oz = other->z->data();
        double* om = other->ms->data();
        for(int idx=0; idx<nxyz; idx++)
        {
            rotateSpinToward(xx[idx], yy[idx], zz[idx], mm[idx],
                             ox[idx], oy[idx], oz[idx], om[idx],
                             max_angle);
        }

    }
    else
    {
        double* xx = x->data();
        double* yy = y->data();
        double* zz = z->data();
        double* mm = ms->data();
        double* max = max_by_site->data();

        double* ox = other->x->data();
        double* oy = other->y->data();
        double* oz = other->z->data();
        double* om = other->ms->data();
        for(int idx=0; idx<nxyz; idx++)
        {
            rotateSpinToward(xx[idx], yy[idx], zz[idx], mm[idx],
                             ox[idx], oy[idx], oz[idx], om[idx],
                             max[idx]);
        }
    }

#ifdef CUDA_VERSION
	x->new_host = true;
	y->new_host = true;
	z->new_host = true;
	ms->new_host = true;
#endif
}


void SpinSystem::moveToward(SpinSystem* other, double r)
{
	if(!sameSize(other))
		return;
	
	for(int idx=0; idx<nxyz; idx++)
	{
		(*x) [idx] += r * ((*other->x)[idx] - (*x)[idx]);
		(*y) [idx] += r * ((*other->y)[idx] - (*y)[idx]);
		(*z) [idx] += r * ((*other->z)[idx] - (*z)[idx]);
		(*ms)[idx] += r * ((*other->ms)[idx] - (*ms)[idx]);
	}
	for(int i=0; i<nxyz; i++)
	{
		const double ll = (*x)[i] * (*x)[i] + (*y)[i] * (*y)[i] + (*z)[i] * (*z)[i];
		double il = 0;
		if(ll > 0)
			il = 1.0 / sqrt(ll);
		const double scale_to_fix = il*(*ms)[i];
		(*x)[i] *= scale_to_fix;
		(*y)[i] *= scale_to_fix;
		(*z)[i] *= scale_to_fix;
	}

#ifdef CUDA_VERSION
	x->new_host = true;
	y->new_host = true;
	z->new_host = true;
	ms->new_host = true;
#endif
}

	
void SpinSystem::diff(SpinSystem* other, double* v4)
{
	v4[0] = x->diffSum(other->x);
	v4[1] = y->diffSum(other->y);
	v4[2] = z->diffSum(other->z);
	v4[3] = sqrt(v4[0]*v4[0] + v4[1]*v4[1] + v4[2]*v4[2]);
}


bool SpinSystem::copyFrom(lua_State* _L, SpinSystem* src)
{
	if(src == this) return true;
	if(nx != src->nx) return false;
	if(ny != src->ny) return false;
	if(nz != src->nz) return false;
	L = _L;
	
	int  here_sum_slot = register_slot_name("Total");
	int there_sum_slot = src->register_slot_name("Total");
	
	ensureSlotExists(here_sum_slot);
	src->ensureSlotExists(there_sum_slot);
	
	hx[here_sum_slot]->copyFrom(src->hx[there_sum_slot]);
	hy[here_sum_slot]->copyFrom(src->hy[there_sum_slot]);
	hz[here_sum_slot]->copyFrom(src->hz[there_sum_slot]);

	x->copyFrom(src->x);
	y->copyFrom(src->y);
	z->copyFrom(src->z);
	ms->copyFrom(src->ms);
	
	if(src->site_alpha)
	{
		luaT_dec<dArray>(site_alpha);
		site_alpha = luaT_inc<dArray>(new dArray(nx,ny,nz));
		site_alpha->copyFrom( src->site_alpha );
	}
	else
	{
		luaT_dec<dArray>(site_alpha);
		site_alpha = 0;
	}
		
	if(src->site_gamma)
	{
		luaT_dec<dArray>(site_gamma);
		site_gamma = luaT_inc<dArray>(new dArray(nx,ny,nz));
		site_gamma->copyFrom( src->site_gamma );
	}
	else
	{
		luaT_dec<dArray>(site_gamma);
		site_gamma = 0;
	}	
	
	alpha = src->alpha;
	gamma = src->gamma;
	dt = src->dt;
	time = src->time;
	
	fft_timeC[0] = time - 1.0;
	fft_timeC[1] = time - 1.0;
	fft_timeC[2] = time - 1.0;
		
	for(int i=0; i<nxyz; i++)
	{
		if(extra_data_size[i] && extra_data[i])
		{
			free(extra_data[i]);
			extra_data[i] = 0;
			extra_data_size[i] = 0;
		}
	}
	
	// make copies
	for(int i=0; i<nxyz; i++)
	{
		if(src->extra_data_size[i])
		{
			extra_data_size[i] = src->extra_data_size[i];
			extra_data[i] = (char*) malloc(src->extra_data_size[i]);
			memcpy(extra_data[i], src->extra_data[i], src->extra_data_size[i]);
		}
	}
	
	return true;
}

bool SpinSystem::copySpinsFrom(lua_State* _L, SpinSystem* src)
{
	if(nx != src->nx) return false;
	if(ny != src->ny) return false;
	if(nz != src->nz) return false;
	L = _L;
	x->copyFrom(src->x);
	y->copyFrom(src->y);
	z->copyFrom(src->z);
	ms->copyFrom(src->ms);
	
	fft_timeC[0] = time - 1.0;
	fft_timeC[1] = time - 1.0;
	fft_timeC[2] = time - 1.0;
	
	return true;
}

bool SpinSystem::copyFieldFrom(lua_State* _L, SpinSystem* src, const char* slot_name)
{
	if(nx != src->nx) return false;
	if(ny != src->ny) return false;
	if(nz != src->nz) return false;
	L = _L;

	int dst_slot = register_slot_name(slot_name);
	int src_slot = src->register_slot_name(slot_name);
	
	if(dst_slot != -1 && src_slot != -1)
	{
		hx[dst_slot]->copyFrom(src->hx[src_slot]);
		hy[dst_slot]->copyFrom(src->hy[src_slot]);
		hz[dst_slot]->copyFrom(src->hz[src_slot]);

		fft_timeC[0] = time - 1.0;
		fft_timeC[1] = time - 1.0;
		fft_timeC[2] = time - 1.0;
		return true;
	}
	return false;
	
}

bool SpinSystem::copyFieldsFrom(lua_State* _L, SpinSystem* src)
{
	L = _L;
	return copyFieldFrom(L, src, "Total");
}


void SpinSystem::deinit()
{
	if(x)
	{
		for(int i=0; i<nxyz; i++)
		{
			if(extra_data_size[i] && extra_data[i])
			{
				free(extra_data[i]);
				extra_data[i] = 0;
				extra_data_size[i] = 0;
			}
		}
	
		luaT_dec<dArray>(x); x = 0;
		luaT_dec<dArray>(y);
		luaT_dec<dArray>(z);
		luaT_dec<dArray>(ms);

		luaT_dec<dArray>(site_alpha); site_alpha = 0;
		luaT_dec<dArray>(site_gamma); site_gamma = 0;

		for(int i=0; i<nslots; i++)
		{
			luaT_dec<dArray>(hx[i]);
			luaT_dec<dArray>(hy[i]);
			luaT_dec<dArray>(hz[i]);
			
			if(registered_slot_names[i])
				free(registered_slot_names[i]);
		}

		free(hx);
		free(hy);
		free(hz);
		free(slot_used);
		free(registered_slot_names);
		registered_slot_names = 0;
		
		ws = 0;
		wsReal = 0;
		
		luaT_dec<dcArray>(qx);
		luaT_dec<dcArray>(qy);
		luaT_dec<dcArray>(qz);
		
		delete [] extra_data;
		delete [] extra_data_size;
	}
}


void SpinSystem::init()
{
    if(x) return;

	nxyz = nx * ny * nz;

	x = luaT_inc<dArray>(new dArray(nx, ny, nz));
	y = luaT_inc<dArray>(new dArray(nx, ny, nz));
	z = luaT_inc<dArray>(new dArray(nx, ny, nz));
	ms= luaT_inc<dArray>(new dArray(nx, ny, nz));
	
	x->setAll(0);
	y->setAll(0);
	z->setAll(0);
	ms->setAll(0);

	// decs are real. Init is empty, clearing old if (by some chance) they exist
	luaT_dec<dArray>(site_alpha); site_alpha = 0;
	luaT_dec<dArray>(site_gamma); site_gamma = 0;
	
	// moving to malloc for realloc
	hx = (dArray**)malloc(sizeof(dArray*) * nslots);
	hy = (dArray**)malloc(sizeof(dArray*) * nslots);
	hz = (dArray**)malloc(sizeof(dArray*) * nslots);
	slot_used = (bool*)malloc(sizeof(bool) * nslots);
	registered_slot_names = (char**)malloc(sizeof(char*) * nslots);
	
	extra_data = new char*[nxyz];
	extra_data_size = new int[nxyz];
	
	for(int i=0; i<nxyz; i++)
	{
		extra_data[i] = 0;
		extra_data_size[i] = 0;
	}
	
	for(int i=0; i<nslots; i++)
	{
		hx[i] = 0; //luaT_inc<dArray>(new dArray(nx,ny,nz));
		hy[i] = 0; //luaT_inc<dArray>(new dArray(nx,ny,nz));
		hz[i] = 0; //luaT_inc<dArray>(new dArray(nx,ny,nz));
		registered_slot_names[i] = 0;
		slot_used[i] = false;
	}
	zeroFields();
	
// 	ws = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	ws     = getWSdcArray(nx,ny,nz,hash32("SpinSystem_FFT_Help"));
	wsReal = getWSdArray(nx,ny,nz,hash32("SpinSystem_Real_WS"));

	
	qx = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qy = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qz = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	
	fft_timeC[0] = -1;
	fft_timeC[1] = -1;
	fft_timeC[2] = -1;
	
	register_slot_name("Total"); //will get slot 0
}


dArray* SpinSystem::getFeildArray(int component, const char* name)
{
	int slot = getSlot(name);
	if(slot < 0)
		return 0;
	
	if(component == 0)
		return hx[slot];
	if(component == 1)
		return hy[slot];
	if(component == 2)
		return hz[slot];
	
	return 0;
}

bool SpinSystem::setFeildArray(int component, const char* name, dArray* a)
{
	if(component < 0 || component > 2)
		return false;
	
	int slot = register_slot_name(name);

	if(slot < 0)
		return false;
		
	dArray** hh[3] = {hx,hy,hz};
	
	luaT_inc<dArray>(a);
	luaT_dec<dArray>(hh[component][slot]);
	hh[component][slot] = a;
	return true;
}


int SpinSystem::register_slot_name(const char* name)
{
	if(name == 0)
		return -1;
	
	// checking to see if it's already registered
	int slot = getSlot(name);
	
	
	if(slot == -1) // then it needs to be registered
	{
		for(slot=0; slot<nslots && registered_slot_names[slot]; slot++)
		{
		};
		
		ensureSlotExists(slot); //grow things if needed
		
		const char ll = strlen(name);
		registered_slot_names[slot] = (char*)malloc(ll+1);
		memcpy(registered_slot_names[slot], name, ll+1);
	}
	return slot;
}

	
void SpinSystem::ensureSlotExists(int slot)
{
	if(slot < 0)
	{
		fprintf(stderr, "(%s:%i) Slot index is negative\n", __FILE__, __LINE__, slot);
		return;
	}

	if(slot >= nslots)
	{
		// increase number of slots
		hx = (dArray**)realloc(hx, sizeof(dArray*) * nslots * 2);
		hy = (dArray**)realloc(hy, sizeof(dArray*) * nslots * 2);
		hz = (dArray**)realloc(hz, sizeof(dArray*) * nslots * 2);
		
		slot_used = (bool*)realloc(slot_used, sizeof(bool) * nslots * 2);
		
		registered_slot_names = (char**)realloc(registered_slot_names, sizeof(char*) * nslots * 2);
		
		for(int i=nslots; i<nslots*2; i++)
		{
			hx[i] = 0;
			hy[i] = 0;
			hz[i] = 0;
			registered_slot_names[i] = 0;
			slot_used[i] = false;
		}
		nslots *= 2;
	}

	if(hx[slot]) return;
	
	hx[slot] = luaT_inc<dArray>(new dArray(nx,ny,nz));
	hy[slot] = luaT_inc<dArray>(new dArray(nx,ny,nz));
	hz[slot] = luaT_inc<dArray>(new dArray(nx,ny,nz));

	hx[slot]->zero();
	hy[slot]->zero();
	hz[slot]->zero();
}


void SpinSystem::encode(buffer* b)
{
	char version = 0;
	encodeChar(version, b);
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);

	encodeDouble(alpha, b);
	encodeDouble(   dt, b);
	encodeDouble(gamma, b);

	encodeDouble(time, b);
	
	x->encode(b);
	y->encode(b);
	z->encode(b);
	ms->encode(b);

	int site_alpha_exists = (site_alpha?1:0);
	int site_gamma_exists = (site_gamma?1:0);
	
	encodeInteger(site_alpha_exists, b);
	if(site_alpha_exists)
		site_alpha->encode(b);
	
	encodeInteger(site_gamma_exists, b);
	if(site_gamma_exists)
		site_gamma->encode(b);
	
	
	int numExtraData = 0;
	
	for(int i=0; i<nxyz; i++)
	{
// 		if(extra_data[i] != LUA_REFNIL)
		if(extra_data[i] != 0)
			numExtraData++;
	}
	
	if(numExtraData > nxyz / 2) //then we'll write all explicitly
	{
		encodeInteger(-1, b); //flag for "all data"
		for(int i=0; i<nxyz; i++)
		{
			if(extra_data[i] != 0)
			{
				// should copy directly to buffer but we're doing this for compatibility
				importLuaVariable(L, extra_data[i], extra_data_size[i]);
// 				lua_rawgeti(L, LUA_REGISTRYINDEX, extra_data[i]);
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
			if(extra_data[i] != 0)
			{
				encodeInteger(i, b);
// 				lua_rawgeti(L, LUA_REGISTRYINDEX, extra_data[i]);
				if(extra_data[i] != 0)
				{
					// should copy directly to buffer but we're doing this for compatibility
					importLuaVariable(L, extra_data[i], extra_data_size[i]);
	// 				lua_rawgeti(L, LUA_REGISTRYINDEX, extra_data[i]);
				}
				else
				{
					lua_pushnil(L);
				}
				_exportLuaVariable(L, -1, b);
				lua_pop(L, 1);
			}
		}
	}
}

int  SpinSystem::decode(buffer* b)
{
//	double r, i;
	deinit();
	char version = decodeChar(b);
	if(version == 0)
	{
		nx = decodeInteger(b);
		ny = decodeInteger(b);
		nz = decodeInteger(b);
		nxyz = nx*ny*nz;

		alpha = decodeDouble(b);
		dt = decodeDouble(b);
		gamma = decodeDouble(b);

		time = decodeDouble(b);
		init();

		x->decode(b);
		y->decode(b);
		z->decode(b);
		ms->decode(b);
		
		const int site_alpha_exists = decodeInteger(b);
		if(site_alpha_exists)
		{
			site_alpha = luaT_inc<dArray>(new dArray(nx,ny,nz));
			site_alpha->decode(b);
		}

		const int site_gamma_exists = decodeInteger(b);
		if(site_gamma_exists)
		{
			site_gamma = luaT_inc<dArray>(new dArray(nx,ny,nz));
			site_gamma->decode(b);
		}
	
		int numPartialData = decodeInteger(b);
		if(numPartialData < 0) //then all, implicitly
		{
			for(int i=0; i<nxyz; i++)
			{
				_importLuaVariable(L, b);
				
				extra_data[i] = exportLuaVariable(L, -1, &(extra_data_size[i]));
				lua_pop(L, 1);
			}
		}
		else
		{
			for(int i=0; i<numPartialData; i++)
			{
				int idx = decodeInteger(b);
				_importLuaVariable(L, b);
				
				if(extra_data[idx])
				{
					free(extra_data[idx]);
					extra_data_size[idx] = 0;
				}
				
				extra_data[idx] = exportLuaVariable(L, -1, &(extra_data_size[idx]));
				lua_pop(L, 1);
			}
		}
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}



void SpinSystem::sumFields()
{
	const int SUM_SLOT = getSlot("Total");
	ensureSlotExists(SUM_SLOT);
	hx[SUM_SLOT]->zero();
	for(int i=1; i<nslots; i++)
		if((i != SUM_SLOT) && slot_used[i] && hx[i])
			dArray::pairwiseScaleAdd(hx[SUM_SLOT], 1.0, hx[SUM_SLOT], 1.0, hx[i]);

	hy[SUM_SLOT]->zero();
	for(int i=1; i<nslots; i++)
		if((i != SUM_SLOT) && slot_used[i] && hy[i])
			dArray::pairwiseScaleAdd(hy[SUM_SLOT], 1.0, hy[SUM_SLOT], 1.0, hy[i]);

	hz[SUM_SLOT]->zero();
	for(int i=1; i<nslots; i++)
		if((i != SUM_SLOT) && slot_used[i] && hz[i])
			dArray::pairwiseScaleAdd(hz[SUM_SLOT], 1.0, hz[SUM_SLOT], 1.0, hz[i]);
}

bool SpinSystem::addFields(double mult, SpinSystem* addThis)
{
	if(nx != addThis->nx) return false;
	if(ny != addThis->ny) return false;
	if(nz != addThis->nz) return false;
	
	const int SUM_SLOT = getSlot("Total");
	const int otherSUM_SLOT = addThis->getSlot("Total");
	
	if(SUM_SLOT >= 0 && otherSUM_SLOT >= 0)
	{
		dArray::pairwiseScaleAdd(hx[SUM_SLOT], 1.0, hx[SUM_SLOT], mult, addThis->hx[otherSUM_SLOT]);
		dArray::pairwiseScaleAdd(hy[SUM_SLOT], 1.0, hy[SUM_SLOT], mult, addThis->hy[otherSUM_SLOT]);
		dArray::pairwiseScaleAdd(hz[SUM_SLOT], 1.0, hz[SUM_SLOT], mult, addThis->hz[otherSUM_SLOT]);

		return true;
	}
	return false;
}


const char* SpinSystem::slotName(int index)
{
	if(registered_slot_names == 0)
		return 0;
	
	if(index >= 0 && index < nslots)
		return registered_slot_names[index];
	
	return 0;
}

int SpinSystem::getSlot(const char* name)
{
	if(registered_slot_names == 0 || name == 0)
		return -1;
	
	for(int i=0; i<nslots; i++)
	{
		if(registered_slot_names[i])
		{
			if(strcasecmp(name, registered_slot_names[i]) == 0)
				return i;
		}	
	}
	
	return -1;
}

void SpinSystem::invalidateFourierData()
{
	// should rethink this masterpiece 
	fft_timeC[0] = -fft_timeC[0] -125978;
	fft_timeC[1] = -fft_timeC[1] -125978;
	fft_timeC[2] = -fft_timeC[2] -125978;
}

	
void SpinSystem::fft()
{
	fft(0);
	fft(1);
	fft(2);
}

void SpinSystem::fft(int component)
{
	if(fft_timeC[component] == time)
		return;
	fft_timeC[component] = time;
	
	ws->zero();
	switch(component)
	{
	case 0:	
		arraySetRealPart(ws->data(), x->data(), x->nxyz);
		ws->fft2DTo(qx); 
		break;
	case 1:	
		arraySetRealPart(ws->data(), y->data(), y->nxyz);
		ws->fft2DTo(qy); 
		break;
	case 2:	
		arraySetRealPart(ws->data(), z->data(), z->nxyz);
		ws->fft2DTo(qz); 
		break;
	}
}


void SpinSystem::zeroFields()
{
	for(int i=0; i<nslots; i++)
	{
		slot_used[i] = false;
		if(hx[i])
		{
			hx[i]->zero();
			hy[i]->zero();
			hz[i]->zero();
		}
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


void SpinSystem::setSiteAlpha(const int px, const int py, const int pz, const double a)
{
	const int i = getidx(px, py, pz);
	if(i < 0 || i >= nxyz) //force crash
	{
		int* i = 0;
		*i = 4;
	}
	setSiteAlpha(i, a);
}
void SpinSystem::setSiteAlpha(const int idx, double a)
{
	if(!site_alpha)
	{
		site_alpha = luaT_inc<dArray>(new dArray(nx,ny,nz));
		site_alpha->setAll(alpha);
	}
	(*site_alpha)[idx] = a;
	site_alpha->new_host = true;
}
void SpinSystem::setAlpha(const double a)
{
	luaT_dec<dArray>(site_alpha);
	site_alpha = 0;
	alpha = a;
}



void SpinSystem::setSiteGamma(const int px, const int py, const int pz, const double g)
{
	const int i = getidx(px, py, pz);
	if(i < 0 || i >= nxyz) //force crash
	{
		int* i = 0;
		*i = 4;
	}
	setSiteGamma(i, g);
}

void SpinSystem::setSiteGamma(const int idx, double g)
{
	if(!site_gamma)
	{
		site_gamma = luaT_inc<dArray>(new dArray(nx,ny,nz));
		site_gamma->setAll(gamma);
	}
	(*site_gamma)[idx] = g;
	site_gamma->new_host = true;
}
void SpinSystem::setGamma(const double g)
{
	luaT_dec<dArray>(site_gamma);
	site_gamma = 0;
	gamma = g;
}



void  SpinSystem::set(const int i, double sx, double sy, double sz)
{
	(*x)[i] = sx;
	(*y)[i] = sy;
	(*z)[i] = sz;

	(*ms)[i]= sqrt(sx*sx+sy*sy+sz*sz);
	
	x->new_host = true;
	y->new_host = true;
	z->new_host = true;
	ms->new_host = true;
	
	invalidateFourierData();
}


void SpinSystem::set(const int px, const int py, const int pz, const double sx, const double sy, const double sz)
{
	const int i = getidx(px, py, pz);
	if(i < 0 || i >= nxyz) //force crash
	{
		int* i = 0;
		*i = 4;
	}
	set(i, sx, sy, sz);
}

void SpinSystem::idx2xyz(int idx, int& x, int& y, int& z) const 
{
	while(idx < 0)
		idx += 10*nxyz;
	idx %= nxyz;
	
	z = idx / (nx*ny);
	idx -= z*nx*ny;
	y = idx / nx;
	x = idx - y*nx;
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
	
	return x + y*nx + z*nx*ny;
}

// return numspins * {<x>, <y>, <z>, <M>, <x^2>, <y^2>, <z^2>, <M^2>}

void SpinSystem::getNetMag(dArray* site_scale1, dArray* site_scale2, dArray* site_scale3, double* v, const double m)
{
	for(int i=0; i<8; i++)
		v[i] = 0;

	if(site_scale1)
		if(site_scale1->nxyz != nxyz)
			return;
	if(site_scale2)
		if(site_scale2->nxyz != nxyz)
			return;
	if(site_scale3)
		if(site_scale3->nxyz != nxyz)
			return;


	dArray* xyz[3];
	xyz[0] = x;
	xyz[1] = y;
	xyz[2] = z;

	for(int i=0; i<3; i++)
	{
		wsReal->copyFrom(xyz[i]);

		if(site_scale1)
			dArray::pairwiseMult(wsReal, wsReal, site_scale1);
		if(site_scale2)
			dArray::pairwiseMult(wsReal, wsReal, site_scale2);
		if(site_scale3)
			dArray::pairwiseMult(wsReal, wsReal, site_scale3);

		v[i  ] = wsReal->sum(1.0) * m;
		v[i+4] = wsReal->sum(2.0) * m;
	}

	v[3] = sqrt(v[0]*v[0] + v[1]*v[1] * v[2]*v[2]);
	v[7] = sqrt(v[4]*v[4] + v[5]*v[5] * v[6]*v[6]);
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
	ss->setAlpha(lua_tonumber(L, 2));
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
	ss->setGamma(lua_tonumber(L, 2));
	return 0;
}
static int l_getgamma(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	lua_pushnumber(L, ss->gamma);
	return 1;
}


static int l_netmoment(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	
	double v8[8];
	double m = 1;
	if(lua_isnumber(L, -1))
		m = lua_tonumber(L, -1);
	
	dArray* arrays[4] = {0,0,0,0};
	int numArrays = 0;
	
	for(int i=2; i<=lua_gettop(L) && numArrays < 3; i++)
	{
		if(luaT_is<dArray>(L, i))
		{
			arrays[numArrays] = luaT_to<dArray>(L, i);
			numArrays++;
		}
	}
	
	ss->getNetMag(arrays[0], arrays[1], arrays[2], v8, m);

	for(int i=0; i<8; i++)
	{
		lua_pushnumber(L, v8[i]);
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
	
	int n = lua_isnumber(L, 2+r1+r2);
	
	
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


static int l_setspin_tpr(lua_State* L)
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
	
	int n = lua_isnumber(L, 2+r1+r2);
		
	int px = site[0] - 1;
	int py = site[1] - 1;
	int pz = site[2] - 1;
	
	const double t = spin[0];
	const double p = spin[1];
	const double r = spin[2];
	
	double sx = r * cos(t) * sin(p);
	double sy = r * sin(t) * sin(p);
	double sz = r * cos(p);

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
	
	lua_pushnumber(L, (*ss->x)[idx]);
	lua_pushnumber(L, (*ss->y)[idx]);
	lua_pushnumber(L, (*ss->z)[idx]);
	
	double len2 = (*ss->x)[idx]*(*ss->x)[idx] 
		+ (*ss->y)[idx]*(*ss->y)[idx]
		+ (*ss->z)[idx]*(*ss->z)[idx];
	
	lua_pushnumber(L, sqrt(len2));
	
	return 4;
}


static int l_getspin_tpr(lua_State* L)
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
	
	const double xx = (*ss->x)[idx];
	const double yy = (*ss->y)[idx];
	const double zz = (*ss->z)[idx];
	
	const double rr = sqrt(xx*xx + yy*yy + zz*zz);
	
	if(r == 0)
	{
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
	}
	else
	{
		const double t = atan2(yy,xx);
		const double p = acos(zz/rr);
	
		lua_pushnumber(L, t);
		lua_pushnumber(L, p);
		lua_pushnumber(L, rr);
	}
			
	return 3;
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

	if((*ss->ms)[idx] == 0)
	{
		lua_pushnumber(L, 1);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
		return 3;
	}
	
	double im = 1.0 / (*ss->ms)[idx];

	lua_pushnumber(L, (*ss->x)[idx]*im);
	lua_pushnumber(L, (*ss->y)[idx]*im);
	lua_pushnumber(L, (*ss->z)[idx]*im);
	
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
	
	if(ss->hx[slot])
	{
		xyz[0] = ss->hx[slot]->sum();
		xyz[1] = ss->hy[slot]->sum();
		xyz[2] = ss->hz[slot]->sum();
		
		lua_pushnumber(L, xyz[0] / ((double)nxyz));
		lua_pushnumber(L, xyz[1] / ((double)nxyz));
		lua_pushnumber(L, xyz[2] / ((double)nxyz));
	}
	else
	{
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
	}
	return 3;
}



static int l_getfieldarrayx(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	const char* name = lua_tostring(L, 2);
	
	if(ss->getSlot(name) < 0)
		return luaL_error(L, "Failed to lookup field name. Consider using :registeredSlots() to see what fields are available.");
	
	luaT_push<dArray>(L, ss->getFeildArray(0, name));
	return 1;
}
static int l_getfieldarrayy(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	const char* name = lua_tostring(L, 2);
	
	if(ss->getSlot(name) < 0)
		return luaL_error(L, "Failed to lookup field name. Consider using :registeredSlots() to see what fields are available.");
	
	luaT_push<dArray>(L, ss->getFeildArray(1, name));
	return 1;
}
static int l_getfieldarrayz(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	const char* name = lua_tostring(L, 2);
	
	if(ss->getSlot(name) < 0)
		return luaL_error(L, "Failed to lookup field name. Consider using :registeredSlots() to see what fields are available.");
	
	luaT_push<dArray>(L, ss->getFeildArray(2, name));
	return 1;
}

static int l_setfieldarrayx(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	const char* name = lua_tostring(L, 2);
	LUA_PREAMBLE(dArray, a, 3);

	if(ss->getSlot(name) < 0)
		return luaL_error(L, "Failed to lookup field name. Consider using :registeredSlots() to see what fields are available.");
	
	ss->setFeildArray(0, name, a);
	return 0;	
}
static int l_setfieldarrayy(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	const char* name = lua_tostring(L, 2);
	LUA_PREAMBLE(dArray, a, 3);

	if(ss->getSlot(name) < 0)
		return luaL_error(L, "Failed to lookup field name. Consider using :registeredSlots() to see what fields are available.");
	
	ss->setFeildArray(1, name, a);
	return 0;	
}
static int l_setfieldarrayz(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	const char* name = lua_tostring(L, 2);
	LUA_PREAMBLE(dArray, a, 3);

	if(ss->getSlot(name) < 0)
		return luaL_error(L, "Failed to lookup field name. Consider using :registeredSlots() to see what fields are available.");
	
	ss->setFeildArray(2, name, a);
	return 0;	
}


#if 0
static int l_spindotfield(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);

	const char* name = lua_tostring(L, 2);

	if(!name)
		return luaL_error(L, "Argument must a string");
	
	int slot = ss->getSlot(name);

	if(slot < 0)
		return luaL_error(L, "Unknown field type`%s'", name);

	dArray* hx = ss->hx[slot];
	dArray* hy = ss->hy[slot];
	dArray* hz = ss->hz[slot];
	
	dArray* sx = ss->x;
	dArray* sy = ss->y;
	dArray* sz = ss->z;
	
	if(!hx || !hy || !hz)
		return luaL_error(L, "Field not computed");
	if(!sx || !sy || !sz)
		return luaL_error(L, "Spins not defined");

	bool bx, by, bz;
	double dx, dy, dz;
	
	bx = sx->dot(hx, dx);
	by = sy->dot(hy, dy);
	bz = sz->dot(hz, dz);
	
	if(!bx || !by || !bz)
		return luaL_error(L, "Size mismatch in dot products");
	
	lua_pushnumber(L, dx+dy+dz);

	return 1;
}
#endif

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

	if(ss->hx[slot])
	{
		const double xx = (*ss->hx[slot])[idx];
		const double yy = (*ss->hy[slot])[idx];
		const double zz = (*ss->hz[slot])[idx];
		
		lua_pushnumber(L, xx);
		lua_pushnumber(L, yy);
		lua_pushnumber(L, zz);

		lua_pushnumber(L, sqrt(xx*xx+yy*yy+zz*zz));
	}
	else
	{
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
	}
	
	return 4;
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

	LUA_PREAMBLE(SpinSystem, dest, 3);

	if(!dest->copyFieldFrom(L, src, slotname))
		return luaL_error(L, "Failed to copyTo");
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

static int l_siteiterator(lua_State* L)
{
	const char* f = 
	"return function(_ss, _include_vac)\n"
	"	local ss, include_vac = _ss, _include_vac\n"
	"	local x,y,z = 1,1,1\n"
	"	local f = function() end\n" // adding f to local scope
	"	f = function()\n" //recoding f
	"		if x == 0 then\n"
	"			return nil\n"
	"		end\n"
	"		local t = {x,y,z}\n"
	"		x = x + 1\n"
	"		if x > ss:nx() then\n"
	"			x, y = 1, y+1\n"
	"			if y > ss:ny() then\n"
	"				y, z = 1, z+1\n"
	"				if z > ss:nz() then\n"
	"					x = 0\n"
	"				end\n"
	"			end\n"
	"		end\n"
	"		local sx,sy,sz,sm = ss:spin(t)\n"
	"		if (sm == 0) and (include_vac == false) then\n"
	"			return f()\n" //calling recoded f closure
	"		end\n"
	"		return t, {sx,sy,sz,sm}\n"
	"	end\n"
	"	return f\n"
	"end\n";

	LUA_PREAMBLE(SpinSystem, ss,  1);
	int skip_vac = 0;
	if(lua_gettop(L) >= 2)
		skip_vac = lua_toboolean(L, 2);
	
	if(luaL_dostring(L, f))
		return luaL_error(L, lua_tostring(L, -1));

	lua_pushvalue(L, 1);
	lua_pushboolean(L, !skip_vac);
	
	if(lua_pcall(L, 2, 1, 0))
		return luaL_error(L, lua_tostring(L, -1));
	
	return 1;
}

static int l_exdatit(lua_State* L)
{
	const char* f = 
	"return function(_ss, _include_nil)\n"
	"	local ss, include_nil = _ss, _include_nil\n"
	"	local x,y,z = 1,1,1\n"
	"	local f = function() end\n" // adding f to local scope
	"	f = function()\n" //recoding f
	"		if x == 0 then\n"
	"			return nil\n"
	"		end\n"
	"		local t = {x,y,z}\n"
	"		x = x + 1\n"
	"		if x > ss:nx() then\n"
	"			x, y = 1, y+1\n"
	"			if y > ss:ny() then\n"
	"				y, z = 1, z+1\n"
	"				if z > ss:nz() then\n"
	"					x = 0\n"
	"				end\n"
	"			end\n"
	"		end\n"
	"		local ed = ss:extraData(t)\n"
	"		if (ed == nil) and (include_nil == false) then\n"
	"			return f()\n" //calling recoded f closure
	"		end\n"
	"		return t, ed\n"
	"	end\n"
	"	return f\n"
	"end\n";

	LUA_PREAMBLE(SpinSystem, ss,  1);
	int skip_nil = 0;
	if(lua_gettop(L) >= 2)
		skip_nil = lua_toboolean(L, 2);
	
	if(luaL_dostring(L, f))
		return luaL_error(L, lua_tostring(L, -1));

	lua_pushvalue(L, 1);
	lua_pushboolean(L, !skip_nil);
	
	if(lua_pcall(L, 2, 1, 0))
		return luaL_error(L, lua_tostring(L, -1));
	
	return 1;
}

static int l_getextradata(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss,  1);


	int site[3];
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int px = site[0] - 1;
	const int py = site[1] - 1;
	const int pz = site[2] - 1;
	
	int idx = ss->getidx(px, py, pz);
	
	if(ss->extra_data[idx] && ss->extra_data_size[idx] > 0)
	{
		importLuaVariable(L, ss->extra_data[idx], ss->extra_data_size[idx]);
	}
	else
	{
		lua_pushnil(L);
	}
	
	return 1;
}

static int l_setextradata(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss,  1);

	int site[3];
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int px = site[0] - 1;
	const int py = site[1] - 1;
	const int pz = site[2] - 1;
	
	int idx = ss->getidx(px, py, pz);
	
	int* extra_data_size; //used for site specific lua data
	char** extra_data;

	if(ss->extra_data[idx])
	{
		free(ss->extra_data[idx]);
		ss->extra_data[idx] = 0;
		ss->extra_data_size[idx] = 0;
	}
	
	ss->extra_data[idx] = exportLuaVariable(L, r+2, &(ss->extra_data_size[idx]));

	return 0;
}

static int l_getinversespinX(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss,  1);
	ss->fft(0); //0 == X
	
	int site[3];
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int px = site[0] - 1;
	const int py = site[1] - 1;
	const int pz = site[2] - 1;
	
	if(!ss->member(px, py, pz))
		return luaL_error(L, "(%d %d %d) is not a member of the system", px+1, py+1, pz+1);
	
	int idx = ss->getidx(px, py, pz);
	return luaT<doubleComplex>::push(L, (*ss->qx)[idx]);
}

static int l_getinversespinY(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss,  1);
	ss->fft(1); //1 == Y
	
	int site[3];
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int px = site[0] - 1;
	const int py = site[1] - 1;
	const int pz = site[2] - 1;
	
	if(!ss->member(px, py, pz))
		return luaL_error(L, "(%d %d %d) is not a member of the system", px+1, py+1, pz+1);
	
	int idx = ss->getidx(px, py, pz);
	return luaT<doubleComplex>::push(L, (*ss->qy)[idx]);

}

static int l_getinversespinZ(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss,  1);
	ss->fft(2); //2 == Z
	
	int site[3];
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int px = site[0] - 1;
	const int py = site[1] - 1;
	const int pz = site[2] - 1;
	
	if(!ss->member(px, py, pz))
		return luaL_error(L, "(%d %d %d) is not a member of the system", px+1, py+1, pz+1);
	
	int idx = ss->getidx(px, py, pz);
	return luaT<doubleComplex>::push(L, (*ss->qz)[idx]);

}

static int l_getinversespin(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss,  1);
	ss->fft();

	int site[3];
	int r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	const int px = site[0] - 1;
	const int py = site[1] - 1;
	const int pz = site[2] - 1;
	
	if(!ss->member(px, py, pz))
		return luaL_error(L, "(%d %d %d) is not a member of the system", px+1, py+1, pz+1);
	
	int idx = ss->getidx(px, py, pz);
	
	r = 0;
	r += luaT<doubleComplex>::push(L, (*ss->qx)[idx]);
	r += luaT<doubleComplex>::push(L, (*ss->qy)[idx]);
	r += luaT<doubleComplex>::push(L, (*ss->qz)[idx]);
	return r;
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




static int l_setarrayx(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	LUA_PREAMBLE(dArray, a, 2);
	
	if(s->x->sameSize(a))
	{
		luaT_inc<dArray>(a);
		luaT_dec<dArray>(s->x);
		s->x = a;
	}
	else
	{
		return luaL_error(L, "Array size mismatch");
	}
	return 0;
}
static int l_setarrayy(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	LUA_PREAMBLE(dArray, a, 2);
	
	if(s->y->sameSize(a))
	{
		luaT_inc<dArray>(a);
		luaT_dec<dArray>(s->y);
		s->y = a;
	}
	else
	{
		return luaL_error(L, "Array size mismatch");
	}
	return 0;	
}
static int l_setarrayz(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	LUA_PREAMBLE(dArray, a, 2);
	
	if(s->z->sameSize(a))
	{
		luaT_inc<dArray>(a);
		luaT_dec<dArray>(s->z);
		s->z = a;
	}
	else
	{
		return luaL_error(L, "Array size mismatch");
	}
	return 0;	
}
static int l_setarraym(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	LUA_PREAMBLE(dArray, a, 2);
	
	if(s->ms->sameSize(a))
	{
		luaT_inc<dArray>(a);
		luaT_dec<dArray>(s->ms);
		s->ms = a;
	}
	else
	{
		return luaL_error(L, "Array size mismatch");
	}
	return 0;
}

static int l_movetoward(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s1,  1);
	LUA_PREAMBLE(SpinSystem, s2,  2);
	const double r = lua_tonumber(L, 3);
	
	if(!s1->sameSize(s2))
		return luaL_error(L, "Systems are not the same size");
	if(r <0 || r > 1)
		return luaL_error(L, "Ratio is not between 0 and 1");
	s1->moveToward(s2, r);
	return 0;
}



static int l_rotatetoward(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s1,  1);
	LUA_PREAMBLE(SpinSystem, s2,  2);
	const double r = lua_tonumber(L, 3);
	
	if(lua_isnumber(L, 3))
	{
		const double max_angle = lua_tonumber(L, 3);
		s1->rotateToward(s2, max_angle, 0);
		return 0;
	}
	else
	{
		if(luaT_is<dArray>(L, 3))
		{
			dArray* max_per_site = luaT_to<dArray>(L, 3);
			if(max_per_site->nxyz != s1->nxyz)
				return luaL_error(L, "Array size mismatch");
			s1->rotateToward(s2, 0, max_per_site);
			return 0;
		}
	}

	return luaL_error(L, "rotateToward expects a goal SpinSystem and a max angle number or max angle array");
}



static int l_getarrayx(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	luaT_push<dArray>(L, s->x);
	return 1;
	return 0;	
}
static int l_getarrayy(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	luaT_push<dArray>(L, s->y);
	return 1;
	return 0;	
}
static int l_getarrayz(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	luaT_push<dArray>(L, s->z);
	return 1;
	return 0;	
}
static int l_getarraym(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	luaT_push<dArray>(L, s->ms);
	return 1;
	return 0;	
}

static int l_getslotused(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);

	const char* name = lua_tostring(L, 2);
	if(!name)                             
		return luaL_error(L, "Argument must a string");
	int slot = s->getSlot(name);                      
	if(slot < 0)                                       
		return luaL_error(L, "Unknown field type`%s'", name); 
	
	lua_pushboolean(L, s->slot_used[slot]);
	return 1;
}
static int l_ensureslotexists(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss,  1);

	const char* name = lua_tostring(L, 2);
	if(!name)                             
		return luaL_error(L, "Argument must be a string");
	
	ss->ensureSlotExists(ss->register_slot_name(name));

	return 0;
}

static int l_setslotused(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);

	const char* name = lua_tostring(L, 2);
	if(!name)                             
		return luaL_error(L, "Argument must a string");
	int slot = s->getSlot(name);                      
	if(slot < 0)                                       
		return luaL_error(L, "Unknown field type`%s'", name); 
	
	if(!lua_isnone(L, 3))
		s->slot_used[slot] = lua_toboolean(L, 3);
	else
		s->slot_used[slot] = 1;

	return 0;
}

static int l_registeredSlots(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss,  1);

	lua_newtable(L);
	
	char** registered_slot_names = ss->registered_slot_names;
	const int nslots = ss->nslots;
	
	int i = 0;
	
	if(registered_slot_names)
	{
		while(registered_slot_names[i] && i < nslots)
		{
			lua_pushinteger(L, i+1);
			lua_pushstring(L, registered_slot_names[i]);
			lua_settable(L, -3);
			i++;
		}
	}
	
	return 1;
}


		// new site a, g
static int l_setsitealphaarray(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	LUA_PREAMBLE(dArray, a, 2);
	
	if(s->x->sameSize(a))
	{
		dArray* old = s->site_alpha;
		s->site_alpha = luaT_inc<dArray>(a);
		luaT_dec<dArray>(old);
	}
	else
	{
		return luaL_error(L, "Array size mismatch");
	}
	
	
	return 0;
}
static int l_setsitegammaarray(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	LUA_PREAMBLE(dArray, a, 2);
	
	if(s->x->sameSize(a))
	{
		dArray* old = s->site_gamma;
		s->site_gamma = luaT_inc<dArray>(a);
		luaT_dec<dArray>(old);
	}
	else
	{
		return luaL_error(L, "Array size mismatch");
	}

	
	return 0;
}
static int l_getsitealphaarray(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	
	if(! s->site_alpha)
	{
		s->site_alpha = luaT_inc<dArray>(new dArray(s->nx,s->ny,s->nz));
		s->site_alpha->setAll(s->alpha);
	}
	luaT_push<dArray>(L, s->site_alpha);
	
	return 1;
}
static int l_getsitegammaarray(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);

	if(! s->site_gamma)
	{
		s->site_gamma = luaT_inc<dArray>(new dArray(s->nx,s->ny,s->nz));
		s->site_gamma->setAll(s->gamma);
	}
	luaT_push<dArray>(L, s->site_gamma);
	
	return 1;
}


static int l_setsitealpha(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	
	int r1;
	int site[3];
	double value;
	
	r1 = lua_getNint(L, 3, site, 2, 1);
	if(r1 < 0)
		return luaL_error(L, "invalid site");

	if(!lua_isnumber(L, 2+r1))
		return luaL_error(L, "missing numeric value");
	
	value = lua_tonumber(L, 2+r1);
	
	int px = site[0] - 1;
	int py = site[1] - 1;
	int pz = site[2] - 1;
	
	ss->setSiteAlpha(px, py, pz, value);
	
	return 0;
}

static int l_setsitegamma(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, ss, 1);
	
	int r1;
	int site[3];
	double value;
	
	r1 = lua_getNint(L, 3, site, 2, 1);
	if(r1 < 0)
		return luaL_error(L, "invalid site");

	if(!lua_isnumber(L, 2+r1))
		return luaL_error(L, "missing numeric value");
	
	value = lua_tonumber(L, 2+r1);
	
	int px = site[0] - 1;
	int py = site[1] - 1;
	int pz = site[2] - 1;
	
	ss->setSiteGamma(px, py, pz, value);
	
	return 0;
}
static int l_getsitealpha(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);

	int r1;
	int site[3];
	double value;
	
	r1 = lua_getNint(L, 3, site, 2, 1);
	if(r1 < 0)
		return luaL_error(L, "invalid site");
	const int idx = s->getidx(site[0], site[1], site[2]);

	if(s->site_alpha)
	{
		lua_pushnumber(L, (*(s->site_alpha))[idx] );
	}
	else
	{
		lua_pushnumber(L, s->alpha);
	}
	
	return 1;
}
static int l_getsitegamma(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);

	int r1;
	int site[3];
	double value;
	
	r1 = lua_getNint(L, 3, site, 2, 1);
	if(r1 < 0)
		return luaL_error(L, "invalid site");
	const int idx = s->getidx(site[0], site[1], site[2]);

	if(s->site_gamma)
	{
		lua_pushnumber(L, (*(s->site_gamma))[idx] );
	}
	else
	{
		lua_pushnumber(L, s->gamma);
	}
	
	return 1;
}



static int l_invalidatefourierdata(lua_State* L)
{
	LUA_PREAMBLE(SpinSystem, s,  1);
	s->invalidateFourierData();
	return 0;
}


int SpinSystem::help(lua_State* L)
{
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

	if(func == l_siteiterator)
	{
		lua_pushstring(L, "Convenience function to iterate over sites in a *SpinSystem*. Example:\n"
							"<pre>\n"
							"for position, moment in ss:siteIterator() do\n"
							"	ss:setSpin(position, {1,0,0})\n"
							"end\n"
							"</pre>");
		lua_pushstring(L, "1 Optional Boolean: Skip vacancy flag. By default vacant sites will be included. If the optional boolean is true then the vacant sites will be skipped.");
		lua_pushstring(L, "1 Iterator Function: Each function call returns a table of position as {i,j,k} and the moment direction and magnitude as a table {x,y,z,m}.");
		return 3;
	}
	
	if(func == l_netmoment)
	{
		lua_pushstring(L, "Calculate and return net magnetization of a spin system");
		lua_pushstring(L, "Up to 3 Optional Array.Double, 1 Optional Number: The optional double arrays scale each site by the product of their values, the optional number scales all sites by a single number. A combination of arrays and a single value can be supplied.");
		lua_pushstring(L, "8 numbers: mean(x), mean(y), mean(z), vector length of {x,y,z}, mean(x^2), mean(y^2), mean(z^2), length of {x^2, y^2, z^2}");
		return 3;
	}
	
	if(func == l_netfield)
	{
		lua_pushstring(L, "Return average field due to an interaction. This field must be calculated with the appropriate operator.");
		lua_pushfstring(L, "1 String: The name of the field type to return. Must match a :slotName() of an applied operator.");
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
		lua_pushstring(L, "4 Numbers: The spin vector at the lattice site and magnitude.");
		return 3;
	}
	
	//
	if(func == l_setspin_tpr)
	{
		lua_pushstring(L, "Set the orientation and magnitude of a spin at a site using spherical coodinates. Note, the theta and phi follow math conventions, not physics conventions. Theta is the azimuthal angle renging from 0 to 2pi, Phi is the zenith angle ranging from 0 to pi.");
		lua_pushstring(L, "2 *3Vector*s: The first argument represents a lattice site. The second represents the spin vector in spherical coordinates.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_getspin_tpr)
	{
		lua_pushstring(L, "Get the orientation and magnitude of a spin at a site using spherical coodinates. Note, the theta and phi follow math conventions, not physics conventions. Theta is the azimuthal angle renging from 0 to 2pi, Phi is the zenith angle ranging from 0 to pi.");
		lua_pushstring(L, "1 *3Vector*: The lattice site.");
		lua_pushstring(L, "3 Numbers: The azimutal, zenith and radial components.");
		return 3;
	}
	//


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
		lua_pushfstring(L, "1 String, 1 *3Vector*: The first argument identifies the field interaction type, must match a :slotName() of an applied operator. The second argument selects the lattice site.");
		lua_pushstring(L, "4 Numbers: The field vector at the site and the magnitude fo the field");
		return 3;
	}
	
	if(func == l_getinversespin)
	{
		lua_pushstring(L, "Return an element of the Fourier Transform of the lattice.");
		lua_pushstring(L, "1 *3Vector*: The lattice site");
		lua_pushstring(L, "6 numbers: The s(q) value represented as a triplet of tables. Each table represents the x, y or z component and the table values [1] and [2] are the real and imaginary parts");
		return 3;
	}
	
	if(func == l_getinversespinX)
	{
		lua_pushstring(L, "Return a component of an element of the Fourier Transform of the lattice.");
		lua_pushstring(L, "1 *3Vector*: The lattice site");
		lua_pushstring(L, "1 table: The sx(q) value represented a tables with values [1] and [2] as the real and imaginary parts");
		return 3;
	}	
	if(func == l_getinversespinY)
	{
		lua_pushstring(L, "Return a component of an element of the Fourier Transform of the lattice.");
		lua_pushstring(L, "1 *3Vector*: The lattice site");
		lua_pushstring(L, "1 table: The sy(q) value represented a tables with values [1] and [2] as the real and imaginary parts");
		return 3;
	}	
	if(func == l_getinversespinZ)
	{
		lua_pushstring(L, "Return a component of an element of the Fourier Transform of the lattice.");
		lua_pushstring(L, "1 *3Vector*: The lattice site");
		lua_pushstring(L, "1 table: The sz(q) value represented a tables with values [1] and [2] as the real and imaginary parts");
		return 3;
	}
	
	
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
	
	if(func == l_exdatit)
	{
		lua_pushstring(L, "Convenience function to iterate over extraData in a *SpinSystem*. Example:\n"
							"<pre>\n"
							"for pos, data in ss:extraDataIterator() do\n"
							"	print(table.concat(pos, \",\"), data)\n"
							"end\n"
							"</pre>");
		lua_pushstring(L, "1 Optional Boolean: Skip nil flag. By default nil entries will be included. If the optional boolean is true then the nil entries will be skipped.");
		lua_pushstring(L, "1 Iterator Function: Each function call returns a table of {i,j,k} and the data at that position.");
		return 3;
	}

	if(func == l_getdiff)
	{
		lua_pushstring(L, "Compute the absolute difference between the current *SpinSystem* and a given *SpinSystem*. dx = Sum( |x[i] - other:x[i]|)");
		lua_pushstring(L, "1 *SpinSystem*: to compare against.");
		lua_pushstring(L, "4 Numbers: The differences in the x, y and z components and the length of the difference vector.");
		return 3;
	}
	
	if(func == l_getarrayx)
	{
		lua_pushstring(L, "Get an array representing the X components of all sites. This array is connected to the SpinSystem so changes to the returned array will change the SpinSystem.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array: The X components of the sites.");
		return 3;
	}
	if(func == l_getarrayy)
	{
		lua_pushstring(L, "Get an array representing the Y components of all sites. This array is connected to the SpinSystem so changes to the returned array will change the SpinSystem.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array: The Y components of the sites.");
		return 3;
	}
	if(func == l_getarrayz)
	{
		lua_pushstring(L, "Get an array representing the Z components of all sites. This array is connected to the SpinSystem so changes to the returned array will change the SpinSystem.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array: The Z components of the sites.");
		return 3;
	}
	if(func == l_getarraym)
	{
		lua_pushstring(L, "Get an array representing the magnitude of all sites. This array is connected to the SpinSystem so changes to the returned array will change the SpinSystem.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array: The magnitude of the sites.");
		return 3;
	}
	
	if(func == l_setarrayx)
	{
		lua_pushstring(L, "Set the entire array representing the X components of the sites to the given array.");
		lua_pushstring(L, "1 Array: The new X components of the sites.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setarrayy)
	{
		lua_pushstring(L, "Set the entire array representing the Y components of the sites to the given array.");
		lua_pushstring(L, "1 Array: The new Y components of the sites.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setarrayz)
	{
		lua_pushstring(L, "Set the entire array representing the Z components of the sites to the given array.");
		lua_pushstring(L, "1 Array: The new Z components of the sites.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setarraym)
	{
		lua_pushstring(L, "Set the entire array representing the magnitude of the sites to the given array.");
		lua_pushstring(L, "1 Array: The new magnitude of the sites.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_movetoward)
	{
		lua_pushstring(L, "Make the calling *SpinSystem* look like a blend between itself and the given *SpinSystem* by a certain ratio. 0 = no change, 1 = completely like the given.");
		lua_pushstring(L, "1 *SpinSystem*, 1 Number: The goal SpinSystem and the amount to change.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_rotatetoward)
	{
		lua_pushstring(L, "Make the calling *SpinSystem* look like a blend between itself and the given *SpinSystem* by rotating the calling system toward the goal system.");
		lua_pushstring(L, "1 *SpinSystem*, 1 Number or 1 Array: The goal SpinSystem and the maximum rotation angle (global value for the Number, site by site maximum for the array).");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getfieldarrayx)
	{
		lua_pushstring(L, "Get the X components of the field vectors for a given type");
		lua_pushstring(L, "1 String: The name of the field");
		lua_pushstring(L, "1 Array: The X components of the field for each sites.");
		return 3;
	}
	if(func == l_getfieldarrayy)
	{
		lua_pushstring(L, "Get the Y components of the field vectors for a given type");
		lua_pushstring(L, "1 String: The name of the field");
		lua_pushstring(L, "1 Array: The Y components of the field for each sites.");
		return 3;
	}
	if(func == l_getfieldarrayz)
	{
		lua_pushstring(L, "Get the Z components of the field vectors for a given type");
		lua_pushstring(L, "1 String: The name of the field");
		lua_pushstring(L, "1 Array: The Z components of the field for each sites.");
		return 3;
	}
	
	if(func == l_setfieldarrayx)
	{
		lua_pushstring(L, "Set the X components of the field vectors for a given type");
		lua_pushstring(L, "1 String, 1 Array: The name of the field, the new X components of the field for each sites.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setfieldarrayy)
	{
		lua_pushstring(L, "Set the Y components of the field vectors for a given type");
		lua_pushstring(L, "1 String, 1 Array: The name of the field, the new Y components of the field for each sites.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setfieldarrayz)
	{
		lua_pushstring(L, "Set the Z components of the field vectors for a given type");
		lua_pushstring(L, "1 String, 1 Array: The name of the field, the new Z components of the field for each sites.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ensureslotexists)
	{
		lua_pushstring(L, "Ensure the *SpinSystem* has a field slot with the given name");
		lua_pushstring(L, "1 String: A field name");
		lua_pushstring(L, "");
		return 3;
	}	
	if(func == l_getslotused)
	{
		lua_pushstring(L, "Determine if an internal field slot has been set");
		lua_pushstring(L, "1 String: A field name");
		lua_pushstring(L, "1 Boolean: The return value");
		return 3;
	}
	if(func == l_setslotused)
	{
		lua_pushstring(L, "Set an internal variable. If true this field type will be added in the sum fields method.");
		lua_pushstring(L, "1 String, 0 or 1 Boolean: A field name and a flag to include or exclude the field in the summation method. Default value is true");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_registeredSlots)
	{
		lua_pushstring(L, "Get all the slot names registered with this SpinSystem.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Table of Strings: Slot names registered with this SpinSystem");
		return 3;
	}
#if 0
	if(func == l_spindotfield)
	{
		lua_pushstring(L, "Computed the dot product of each moment with the given field");
		lua_pushstring(L, "1 String: The name of the field");
		lua_pushstring(L, "1 Number: The dot product");
		return 3;
	}
#endif

	if(func == l_setsitealphaarray)
	{
		lua_pushstring(L, "Set the internal site by site damping array to a new array");
		lua_pushstring(L, "1 Array: New damping array");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setsitegammaarray)
	{
		lua_pushstring(L, "Set the internal site by site gyromagnetic array to a new array");
		lua_pushstring(L, "1 Array: New gyromagnetic array");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getsitealphaarray)
	{
		lua_pushstring(L, "Get the internal site by site damping array");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array: Internal damping array");
		return 3;
	}
	if(func == l_getsitegammaarray)
	{
		lua_pushstring(L, "Get the internal site by site gyromagnetic array");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array: Internal gyromagnetic array");
		return 3;
	}
	if(func == l_setsitealpha)
	{
		lua_pushstring(L, "Set an individual site's damping to a unique value");
		lua_pushstring(L, "1 *3Vector*, 1 Number: Site and value");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setsitegamma)
	{
		lua_pushstring(L, "Set an individual site's gyromagnetic value to a unique value");
		lua_pushstring(L, "1 *3Vector*, 1 Number: Site and value");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getsitealpha)
	{
		lua_pushstring(L, "Get an individual site's damping value");
		lua_pushstring(L, "1 *3Vector*: Site");
		lua_pushstring(L, "1 Number: Value");
		return 3;
	}	
	if(func == l_getsitegamma)
	{
		lua_pushstring(L, "Get an individual site's gyromagnetic value");
		lua_pushstring(L, "1 *3Vector*: Site");
		lua_pushstring(L, "1 Number: Value");
		return 3;
	}	

	if(func == l_invalidatefourierdata)
	{
		lua_pushstring(L, "Invalidates the cache of the Fourier transform of the spin system. If the time changes or :setSpin "
						  "is called then the cache is invalidated but there are cases, such as when the internal arrays are exported "
						  "and modified, when the SpinSystem isn't aware of changes. This function help to deal with those extreme cases");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}

	return LuaBaseObject::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* SpinSystem::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		{"__tostring",   l_tostring},
		{"moveToward",   l_movetoward},
		{"rotateToward",   l_rotatetoward},
		{"netMoment",    l_netmoment},
		{"netField",     l_netfield},
		{"siteIterator", l_siteiterator},
		{"eachSite",     l_siteiterator},
		{"setSpin",      l_setspin},
		{"spin"   ,      l_getspin},
		{"setSpinTPR",      l_setspin_tpr},
		{"spinTPR"   ,      l_getspin_tpr},
		{"unitSpin",     l_getunitspin},
		{"nx",           l_nx},
		{"ny",           l_ny},
		{"nz",           l_nz},
		{"sumFields",    l_sumfields},
		{"resetFields",  l_zerofields},
		{"setTime",      l_settime},
		{"time",         l_gettime},
		{"field",        l_getfield},
// 		{"spinDotField", l_spindotfield},
		{"inverseSpin",  l_getinversespin},
		{"inverseSpinX",  l_getinversespinX},
		{"inverseSpinY",  l_getinversespinY},
		{"inverseSpinZ",  l_getinversespinZ},
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
		{"extraDataIterator", l_exdatit},
		{"eachExtraData", l_exdatit},
		{"diff",         l_getdiff},
		
		{"spinArrayX",  l_getarrayx},
		{"spinArrayY",  l_getarrayy},
		{"spinArrayZ",  l_getarrayz},
		{"spinArrayM",  l_getarraym},
	
		{"setSpinArrayX",  l_setarrayx},
		{"setSpinArrayY",  l_setarrayy},
		{"setSpinArrayZ",  l_setarrayz},
		{"setSpinArrayM",  l_setarraym},

		{"fieldArrayX",  l_getfieldarrayx},
		{"fieldArrayY",  l_getfieldarrayy},
		{"fieldArrayZ",  l_getfieldarrayz},

		{"setFieldArrayX",  l_setfieldarrayx},
		{"setFieldArrayY",  l_setfieldarrayy},
		{"setFieldArrayZ",  l_setfieldarrayz},
		
		{"slotUsed", l_getslotused},
		{"setSlotUsed", l_setslotused},
		{"ensureSlotExists", l_ensureslotexists},

		// new site a, g
		{"setSiteAlphaArray", l_setsitealphaarray},
		{"setSiteGammaArray", l_setsitegammaarray},
		{"siteAlphaArray",    l_getsitealphaarray},
		{"siteGammaArray",    l_getsitegammaarray},
		{"setSiteAlpha",      l_setsitealpha},
		{"setSiteGamma",      l_setsitegamma},
		{"siteAlpha",         l_getsitealpha},
		{"siteGamma",         l_getsitegamma},
		
		{"registeredSlots", l_registeredSlots},
		
		{"invalidateFourierData", l_invalidatefourierdata},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


