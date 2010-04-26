#include "spinsystem.h"
#include "spinoperation.h"
#include <iostream>
#include <math.h>
#include <strings.h>
#include <string.h>

using namespace std;
#define CLAMP(x, m) ((x<0)?0:(x>m?m:x))

SpinSystem::SpinSystem(const int NX, const int NY, const int NZ)
	: Encodable(ENCODE_SPINSYSTEM), x(0), y(0), z(0), 
		ms(0), nx(NX), ny(NY), nz(NZ), refcount(0),
		nslots(NSLOTS), time(0)
{
	init();
}

SpinSystem::~SpinSystem()
{
	deinit();
}

void SpinSystem::deinit()
{
	if(x)
	{
		delete [] x;
		delete [] y;
		delete [] z;
		delete [] ms;

		
		for(int i=0; i<nslots; i++)
		{
			delete [] hx[i];
			delete [] hy[i];
			delete [] hz[i];
		}

		delete [] hx;
		delete [] hy;
		delete [] hz;

		delete [] rx;
		delete [] ry;
		delete [] rz;

		delete [] qx;
		delete [] qy;
		delete [] qz;
		
		
		fftw_destroy_plan(r2q);
	}
}


void SpinSystem::init()
{
	nxyz = nx * ny * nz;

	x = new double[nxyz];
	y = new double[nxyz];
	z = new double[nxyz];
	ms= new double[nxyz];
	
	for(int i=0; i<nxyz; i++)
		set(i, 0, 0, 0);

	hx = new double* [nslots];
	hy = new double* [nslots];
	hz = new double* [nslots];
	
	for(int i=0; i<nslots; i++)
	{
		hx[i] = new double[nxyz];
		hy[i] = new double[nxyz];
		hz[i] = new double[nxyz];
		
		for(int j=0; j<nxyz; j++)
		{
			hx[i][j] = 0;
			hy[i][j] = 0;
			hz[i][j] = 0;
		}
	}

	rx = new complex<double>[nxyz];
	ry = new complex<double>[nxyz];
	rz = new complex<double>[nxyz];

	qx = new complex<double>[nxyz];
	qy = new complex<double>[nxyz];
	qz = new complex<double>[nxyz];
	
	fftw_iodim dims[2];
	dims[0].n = nx;
	dims[0].is= 1;
	dims[0].os= 1;
	dims[1].n = ny;
	dims[1].is= nx;
	dims[1].os= nx;

	r2q = fftw_plan_guru_dft(2, dims, 0, dims,
				reinterpret_cast<fftw_complex*>(rx),
				reinterpret_cast<fftw_complex*>(qx),
				FFTW_FORWARD, FFTW_PATIENT);
				
	fft_time = time - 1.0;
}

//   void encodeBuffer(const void* s, int len, buffer* b);
//   void encodeDouble(const double d, buffer* b);
//   void encodeInteger(const int i, buffer* b);
//    int decodeInteger(const char* buf, int* pos);
// double decodeDouble(const char* buf, int* pos);


void SpinSystem::encode(buffer* b) const
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);

	encodeDouble(time, b);
	
	for(int i=0; i<nxyz; i++)
	{
		encodeDouble(x[i], b);
		encodeDouble(y[i], b);
		encodeDouble(z[i], b);
		
		encodeDouble(qx[i].real(), b);
		encodeDouble(qx[i].imag(), b);

		encodeDouble(qy[i].real(), b);
		encodeDouble(qy[i].imag(), b);

		encodeDouble(qz[i].real(), b);
		encodeDouble(qz[i].imag(), b);


		for(int j=0; j<nslots; j++)
		{
			encodeDouble(hx[j][i], b);
			encodeDouble(hy[j][i], b);
			encodeDouble(hz[j][i], b);
		}
	}
}

int  SpinSystem::decode(buffer* b)
{
	deinit();
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	time = decodeDouble(b);
	init();

	for(int i=0; i<nxyz; i++)
	{
		x[i] = decodeDouble(b);
		y[i] = decodeDouble(b);
		z[i] = decodeDouble(b);
		
		qx[i] = complex<double>(decodeDouble(b), decodeDouble(b));
		qy[i] = complex<double>(decodeDouble(b), decodeDouble(b));
		qz[i] = complex<double>(decodeDouble(b), decodeDouble(b));
				
		for(int j=0; j<nslots; j++)
		{
			hx[j][i] = decodeDouble(b);
			hy[j][i] = decodeDouble(b);
			hz[j][i] = decodeDouble(b);
		}
	}

	return 0;
}


void SpinSystem::sumFields()
{
	for(int j=0; j<nxyz; j++)
	{
		hx[SUM_SLOT][j] = hx[1][j];
		hy[SUM_SLOT][j] = hy[1][j];
		hz[SUM_SLOT][j] = hz[1][j];
	}

	for(int i=2; i<NSLOTS; i++)
	{
		for(int j=0; j<nxyz; j++)
		{
			hx[SUM_SLOT][j] += hx[i][j];
			hy[SUM_SLOT][j] += hy[i][j];
			hz[SUM_SLOT][j] += hz[i][j];
		}
	}
}

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

void SpinSystem::fft()
{
	for(int i=0; i<nxyz; i++) rx[i] = x[i];
	for(int i=0; i<nxyz; i++) ry[i] = y[i];
	for(int i=0; i<nxyz; i++) rz[i] = z[i];
	
	for(int k=0; k<nz; k++)
	{
		fftw_execute_dft(r2q, 
			reinterpret_cast<fftw_complex*>(&rx[k*nx*ny]),
			reinterpret_cast<fftw_complex*>(&qx[k*nx*ny]));
		fftw_execute_dft(r2q, 
			reinterpret_cast<fftw_complex*>(&ry[k*nx*ny]),
			reinterpret_cast<fftw_complex*>(&qy[k*nx*ny]));
		fftw_execute_dft(r2q, 
			reinterpret_cast<fftw_complex*>(&rz[k*nx*ny]),
			reinterpret_cast<fftw_complex*>(&qz[k*nx*ny]));
	}

	fft_time = time;
}


void SpinSystem::zeroFields()
{
	for(int i=0; i<NSLOTS; i++)
	{
		for(int j=0; j<nxyz; j++)
		{
			hx[i][j] = 0;
			hy[i][j] = 0;
			hz[i][j] = 0;
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

void  SpinSystem::set(const int i, double sx, double sy, double sz)
{
	x[i] = sx;
	y[i] = sy;
	z[i] = sz;

	ms[i]= sqrt(sx*sx+sy*sy+sz*sz);
}


void SpinSystem::set(const int px, const int py, const int pz, const double sx, const double sy, const double sz)
{
	const int i = getidx(px, py, pz);
	set(i, sx, sy, sz);
	if(i < 0 || i >= nxyz)
	{
		printf("%i %i %i %i\n", i, px, py, pz);

		int* i = 0;
		*i = 4;
	}
}

int  SpinSystem::getidx(const int px, const int py, const int pz) const
{
	const int x = CLAMP(px, nx-1);
	const int y = CLAMP(py, ny-1);
	const int z = CLAMP(pz, nz-1);
	
	return x + y*nx + z*nx*ny;
}

void SpinSystem::getNetMag(double* v4)
{
	v4[0] = 0; v4[1] = 0; v4[2] = 0; v4[3] = 0; 
	
	for(int i=0; i<nxyz; i++)
		v4[0] += x[i];
	for(int i=0; i<nxyz; i++)
		v4[1] += y[i];
	for(int i=0; i<nxyz; i++)
		v4[2] += z[i];

	v4[3] = sqrt(v4[0]*v4[0] + v4[1]*v4[1] + v4[2]*v4[2]);
}

bool SpinSystem::copy(SpinSystem* src)
{
	if(nx != src->nx) return false;
	if(ny != src->ny) return false;
	if(nz != src->nz) return false;
	
	memcpy(hx[SUM_SLOT], src->hx[SUM_SLOT], nxyz * sizeof(double));
	memcpy(hy[SUM_SLOT], src->hy[SUM_SLOT], nxyz * sizeof(double));
	memcpy(hz[SUM_SLOT], src->hz[SUM_SLOT], nxyz * sizeof(double));
	
	memcpy(x, src->x, nxyz * sizeof(double));
	memcpy(y, src->y, nxyz * sizeof(double));
	memcpy(z, src->z, nxyz * sizeof(double));
	
	fft_time = time - 1.0;
	
	return true;
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
				n[i] = lua_tonumber(L, i+1);
			else
				n[i] = 1.0;
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
		delete ss;
	
	return 0;
}

int l_ss_netmag(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
	double m = 1;
	if(lua_isnumber(L, 2))
		m = lua_tonumber(L, 2);

	double v4[4];
	
	ss->getNetMag(v4);
	
	lua_pushnumber(L, v4[0]*m);
	lua_pushnumber(L, v4[1]*m);
	lua_pushnumber(L, v4[2]*m);
	lua_pushnumber(L, v4[3]*m);
	
	return 4;
}

int l_ss_setspin(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
	int r;
	int site[3];
	double spin[3];
	
	r = lua_getNint(L, 3, site, 2, 1);
	if(r < 0)
		return luaL_error(L, "invalid site");
	
	r = lua_getNdouble(L, 3, spin, 2+r, 0);
	if(r < 0)
		return luaL_error(L, "invalid spin");
	
	int px = site[0] - 1;
	int py = site[1] - 1;
	int pz = site[2] - 1;
	
	double sx = spin[0];
	double sy = spin[1];
	double sz = spin[2];
	
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
		return luaL_error(L, "invalid site");
	
	int px = site[0] - 1;
	int py = site[1] - 1;
	int pz = site[2] - 1;
	
	if(!ss->member(px, py, pz))
		return 0;
	
	int idx = ss->getidx(px, py, pz);
	
	lua_pushnumber(L, ss->x[idx]);
	lua_pushnumber(L, ss->y[idx]);
	lua_pushnumber(L, ss->z[idx]);
	
	return 3;
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

	if(ss->ms[idx] == 0)
	{
		lua_pushnumber(L, 1);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
		return 3;
	}
	
	double im = 1.0 / ss->ms[idx];

	lua_pushnumber(L, ss->x[idx]*im);
	lua_pushnumber(L, ss->y[idx]*im);
	lua_pushnumber(L, ss->z[idx]*im);
	
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

int l_ss_getfield(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;

	const char* name = lua_tostring(L, 2);

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


	lua_pushnumber(L, ss->hx[slot][idx]);
	lua_pushnumber(L, ss->hy[slot][idx]);
	lua_pushnumber(L, ss->hz[slot][idx]);

	return 3;
}

int l_ss_addfields(lua_State* L)
{
	SpinSystem* dest = checkSpinSystem(L, 1);
	if(!dest) return 0;
	
	SpinSystem* src = 0;
	double mult;
	
	if(lua_isnumber(L, 2))
	{
		mult = lua_tonumber(L, 2);
		src  = checkSpinSystem(L, 3);
	}
	else
	{
		mult = 1.0;
		src  = checkSpinSystem(L, 2);
	}
	if(!src) return 0;
	
	if(!dest->addFields(mult, src))
		return luaL_error(L, "Failed to sum fields");
	
	return 0;
}

int l_ss_copy(lua_State* L)
{
	SpinSystem* dest = checkSpinSystem(L, 1);
	if(!dest) return 0;
	
	SpinSystem* src = checkSpinSystem(L, 2);
	if(!src) return 0;
	
	if(!dest->copy(src))
		return luaL_error(L, "Failed to copy");
	
	return 0;
}

int l_ss_getinversespin(lua_State* L)
{
	SpinSystem* ss = checkSpinSystem(L, 1);
	if(!ss) return 0;
	
	if(ss->time != ss->fft_time)
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
	
	lua_newtable(L);
	lua_pushinteger(L, 1);
	lua_pushnumber(L, real(ss->qx[idx]));
	lua_settable(L, -3);
	lua_pushinteger(L, 2);
	lua_pushnumber(L, imag(ss->qx[idx]));
	lua_settable(L, -3);
	
	lua_newtable(L);
	lua_pushinteger(L, 1);
	lua_pushnumber(L, real(ss->qy[idx]));
	lua_settable(L, -3);
	lua_pushinteger(L, 2);
	lua_pushnumber(L, imag(ss->qy[idx]));
	lua_settable(L, -3);
	
	lua_newtable(L);
	lua_pushinteger(L, 1);
	lua_pushnumber(L, real(ss->qz[idx]));
	lua_settable(L, -3);
	lua_pushinteger(L, 2);
	lua_pushnumber(L, imag(ss->qz[idx]));
	lua_settable(L, -3);
	
	return 3;
}


void registerSpinSystem(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_ss_gc},
		{"__tostring",   l_ss_tostring},
		{"netMag",       l_ss_netmag},
		{"setSpin",      l_ss_setspin},
		{"spin"   ,      l_ss_getspin},
		{"unitSpin",     l_ss_getunitspin},
		{"nx",           l_ss_nx},
		{"ny",           l_ss_ny},
		{"nz",           l_ss_nz},
		{"sumFields",    l_ss_sumfields},
		{"zeroFields",   l_ss_zerofields},
		{"setTime",      l_ss_settime},
		{"time",         l_ss_gettime},
		{"getField",     l_ss_getfield},
		{"inverseSpin",  l_ss_getinversespin},
		{"addFields",    l_ss_addfields},
		{"copy",         l_ss_copy},
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
		{NULL, NULL}
	};
		
	luaL_register(L, "SpinSystem", functions);
	lua_pop(L,1);
}
