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

#include "spinoperationdipole.h"
#include "spinsystem.h"
#include "dipolesupport.h"
#include "info.h"

#include <stdlib.h>
#include <math.h>

Dipole::Dipole(int nx, int ny, int nz)
	: SpinOperation("Dipole", DIPOLE_SLOT, nx, ny, nz, ENCODE_DIPOLE)
{
	g = 1;
	gmax = 2000;
	qXX = 0;
	ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
	ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
	ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	hasMatrices = false;
}

void Dipole::init()
{
	if(qXX)
		deinit();
	
	hqx = new complex<double> [nxyz];
	hqy = new complex<double> [nxyz];
	hqz = new complex<double> [nxyz];

	hrx = new complex<double> [nxyz];
	hry = new complex<double> [nxyz];
	hrz = new complex<double> [nxyz];

	int s = nx*ny * (nz*2-1);
	qXX = new complex<double>[s];
	qXY = new complex<double>[s];
	qXZ = new complex<double>[s];

	qYY = new complex<double>[s];
	qYZ = new complex<double>[s];
	qZZ = new complex<double>[s];
	
	fftw_iodim dims[2];
	dims[0].n = nx;
	dims[0].is= 1;
	dims[0].os= 1;
	dims[1].n = ny;
	dims[1].is= nx;
	dims[1].os= nx;

	forward = fftw_plan_guru_dft(2, dims, 0, dims,
								reinterpret_cast<fftw_complex*>(qXX),
								reinterpret_cast<fftw_complex*>(qYY),
								FFTW_FORWARD, FFTW_PATIENT);

	backward= fftw_plan_guru_dft(2, dims, 0, dims,
								reinterpret_cast<fftw_complex*>(qXX),
								reinterpret_cast<fftw_complex*>(qYY),
								FFTW_BACKWARD, FFTW_PATIENT);
								
	hasMatrices = false;
}

void Dipole::deinit()
{
	if(!qXX)
		return;
	delete [] qXX;
	delete [] qXY;
	delete [] qXZ;

	delete [] qYY;
	delete [] qYZ;

	delete [] qZZ;

	delete [] hqx;
	delete [] hqy;
	delete [] hqz;

	delete [] hrx;
	delete [] hry;
	delete [] hrz;

	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	qXX = 0;
}

void Dipole::encode(buffer* b)
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	encodeInteger(gmax, b);
	encodeDouble(g, b);

	for(int i=0; i<9; i++)
	{
		encodeDouble(ABC[i], b);
	}
}

int  Dipole::decode(buffer* b)
{
	deinit();
	
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	gmax = decodeInteger(b);
	g = decodeDouble(b);
	nxyz = nx*ny*nz;

	for(int i=0; i<9; i++)
	{
		ABC[i] = decodeDouble(b);
	}

	hasMatrices = false;
	
	return 0;
}

Dipole::~Dipole()
{
	deinit();
}

void Dipole::getMatrices()
{
	init();
	
	int s = nx*ny * (nz*2-1);
	double* XX = new double[s];
	double* XY = new double[s];
	double* XZ = new double[s];
	double* YY = new double[s];
	double* YZ = new double[s];
	double* ZZ = new double[s];
	
	dipoleLoad(
		nx, ny, nz,
		gmax, ABC,
		XX, XY, XZ,
		YY, YZ, ZZ);

	fftw_iodim dims[2];
	dims[0].n = nx;
	dims[0].is= 1;
	dims[0].os= 1;
	dims[1].n = ny;
	dims[1].is= nx;
	dims[1].os= nx;

	complex<double>* r = new complex<double>[nx*ny];
	complex<double>* q = new complex<double>[nx*ny];
	
	double* arrs[6];
	arrs[0] = XX;
	arrs[1] = XY;
	arrs[2] = XZ;
	arrs[3] = YY;
	arrs[4] = YZ;
	arrs[5] = ZZ;
	
	complex<double>* qarrs[6];
	qarrs[0] = qXX;
	qarrs[1] = qXY;
	qarrs[2] = qXZ;
	qarrs[3] = qYY;
	qarrs[4] = qYZ;
	qarrs[5] = qZZ;
	
	for(int a=0; a<6; a++)
		for(int k=0; k<2*nz-1; k++)
		{
			for(int i=0; i<nx*ny; i++)
				r[i] = complex<double>(arrs[a][k*nx*ny + i],0);
		
			fftw_execute_dft(forward, 
					reinterpret_cast<fftw_complex*>(r),
					reinterpret_cast<fftw_complex*>(q));

			for(int i=0; i<nx*ny; i++)
				qarrs[a][k*nx*ny + i] = q[i];
		}
	
	
	
	delete [] q;
	delete [] r;
	
	delete [] XX;
	delete [] XY;
	delete [] XZ;
	delete [] YY;
	delete [] YZ;
	delete [] ZZ;
	
	hasMatrices = true;
}

void Dipole::ifftAppliedForce(SpinSystem* ss)
{
	double d = g / (double)(nx*ny);
// 	printf("%g\n", d);
	double* hx = ss->hx[slot];
	double* hy = ss->hy[slot];
	double* hz = ss->hz[slot];
	const int nxy = nx*ny;
	
	for(int i=0; i<nz; i++)
	{
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqx[i*nxy]),
				reinterpret_cast<fftw_complex*>(&hrx[i*nxy]));
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqy[i*nxy]),
				reinterpret_cast<fftw_complex*>(&hry[i*nxy]));
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqz[i*nxy]),
				reinterpret_cast<fftw_complex*>(&hrz[i*nxy]));
	}

	for(int i=0; i<nxyz; i++)
		hx[i] = hrx[i].real() * d;

	for(int i=0; i<nxyz; i++)
		hy[i] = hry[i].real() * d;

	for(int i=0; i<nxyz; i++)
		hz[i] = hrz[i].real() * d;
}


void Dipole::collectIForces(SpinSystem* ss)
{
	int c;
	int sourceLayer, targetLayer;// !!Source layer, Target Layer
	int sourceOffset;
	int targetOffset;
	int demagOffset;
	const int nxy = nx*ny;
	//int LL = nz*2-1;
	
	complex<double>* sqx = ss->qx;
	complex<double>* sqy = ss->qy;
	complex<double>* sqz = ss->qz;

	if(!hasMatrices)
		getMatrices();
	
	const complex<double> cz(0,0);
	for(c=0; c<nxyz; c++) hqx[c] = cz;
	for(c=0; c<nxyz; c++) hqy[c] = cz;
	for(c=0; c<nxyz; c++) hqz[c] = cz;

	
# define cSo c+sourceOffset
# define cDo c+ demagOffset
# define cTo c+targetOffset


	for(targetLayer=0; targetLayer<nz; targetLayer++)
	for(sourceLayer=0; sourceLayer<nz; sourceLayer++)
	{
		targetOffset = targetLayer * nxy;
		sourceOffset = sourceLayer * nxy;
		demagOffset  = ( sourceLayer - targetLayer + nz - 1 ) * nxy;
		//these are complex multiplies and adds
		for(c=0; c<nxy; c++) hqx[cTo]+=qXX[cDo]*sqx[cSo];
		for(c=0; c<nxy; c++) hqx[cTo]+=qXY[cDo]*sqy[cSo];
		for(c=0; c<nxy; c++) hqx[cTo]+=qXZ[cDo]*sqz[cSo];

		for(c=0; c<nxy; c++) hqy[cTo]+=qXY[cDo]*sqx[cSo];
		for(c=0; c<nxy; c++) hqy[cTo]+=qYY[cDo]*sqy[cSo];
		for(c=0; c<nxy; c++) hqy[cTo]+=qYZ[cDo]*sqz[cSo];

		for(c=0; c<nxy; c++) hqz[cTo]+=qXZ[cDo]*sqx[cSo];
		for(c=0; c<nxy; c++) hqz[cTo]+=qYZ[cDo]*sqy[cSo];
		for(c=0; c<nxy; c++) hqz[cTo]+=qZZ[cDo]*sqz[cSo];
	}
}

// struct dipss
// {
// 	Dipole* dip;
// 	SpinSystem* ss;
// };
// 
// static void* thread_dipole_wrapper(void* ds)
// {
// 	struct dipss* x = (struct dipss*)ds;
// 	
// 	x->dip->apply(x->ss);
// 	
// 	delete x;
// 	return 0;
// }
// 
// void Dipole::threadApply(SpinSystem* ss)
// {
// 	struct dipss* x = new struct dipss;
// 	x->dip = this;
// 	x->ss  = ss;
// 	
// 	ss->start_thread(DIPOLE_SLOT, thread_dipole_wrapper, x);
// }
	
bool Dipole::apply(SpinSystem* ss)
{
	markSlotUsed(ss);

	ss->fft();
	collectIForces(ss);
	ifftAppliedForce(ss);

	return true;
}







Dipole* checkDipole(lua_State* L, int idx)
{
	Dipole** pp = (Dipole**)luaL_checkudata(L, idx, "MERCER.dipole");
    luaL_argcheck(L, pp != NULL, 1, "`Dipole' expected");
    return *pp;
}

void lua_pushDipole(lua_State* L, Encodable* _dip)
{
	Dipole* dip = dynamic_cast<Dipole*>(_dip);
	if(!dip) return;
	dip->refcount++;
	Dipole** pp = (Dipole**)lua_newuserdata(L, sizeof(Dipole**));
	
	*pp = dip;
	luaL_getmetatable(L, "MERCER.dipole");
	lua_setmetatable(L, -2);
}

int l_dip_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	lua_pushDipole(L, new Dipole(n[0], n[1], n[2]));
	return 1;
}


int l_dip_setstrength(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	dip->g = lua_tonumber(L, 2);
	return 0;
}

int l_dip_gc(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	dip->refcount--;
	if(dip->refcount == 0)
		delete dip;
	
	return 0;
}

int l_dip_apply(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!dip->apply(ss))
		return luaL_error(L, dip->errormsg.c_str());
	
	return 0;
}

// int l_dip_threadapply(lua_State* L)
// {
// 	Dipole* dip = checkDipole(L, 1);
// 	if(!dip) return 0;
// 	SpinSystem* ss = checkSpinSystem(L, 2);
// 	
// 	dip->threadApply(ss);
// 	
// 	return 0;
// }

int l_dip_getstrength(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	lua_pushnumber(L, dip->g);

	return 1;
}
int l_dip_setunitcell(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	double A[3];
	double B[3];
	double C[3];
	
	int r1 = lua_getNdouble(L, 3, A, 2, 0);
	int r2 = lua_getNdouble(L, 3, B, 2+r1, 0);
	int r3 = lua_getNdouble(L, 3, C, 2+r1+r2, 0);
	
	for(int i=0; i<3; i++)
	{
		dip->ABC[i+0] = A[i];
		dip->ABC[i+3] = B[i];
		dip->ABC[i+6] = C[i];
	}

	return 0;
}
int l_dip_getunitcell(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	double* ABC[3];
	ABC[0] = &(dip->ABC[0]);
	ABC[1] = &(dip->ABC[3]);
	ABC[2] = &(dip->ABC[6]);
	
	for(int i=0; i<3; i++)
	{
		lua_newtable(L);
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_pushnumber(L, ABC[i][j]);
			lua_settable(L, -3);
		}
	}
	
	return 3;
}
int l_dip_settrunc(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	dip->gmax = lua_tointeger(L, 2);

	return 0;
}
int l_dip_gettrunc(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	lua_pushnumber(L, dip->gmax);

	return 1;
}

static int l_dip_tostring(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;
	
	lua_pushfstring(L, "Dipole (%dx%dx%d)", dip->nx, dip->ny, dip->nz);
	
	return 1;
}

static int l_dip_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.dipole");
	return 1;
}

static int l_dip_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the dipolar field of a *SpinSystem*");
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
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_dip_new)
	{
		lua_pushstring(L, "Create a new Dipole Operator.");
		lua_pushstring(L, "1 *3Vector*: system size"); 
		lua_pushstring(L, "1 Dipole object");
		return 3;
	}
	
	
	if(func == l_dip_apply)
	{
		lua_pushstring(L, "Calculate the dipolar field of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This spin system will receive the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dip_setstrength)
	{
		lua_pushstring(L, "Set the strength of the Dipolar Field");
		lua_pushstring(L, "1 number: strength of the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dip_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Dipolar Field");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: strength of the field");
		return 3;
	}
	
	if(func == l_dip_setunitcell)
	{
		lua_pushstring(L, "Set the unit cell of a lattice site");
		lua_pushstring(L, "3 *3Vector*: The A, B and C vectors defining the unit cell. By default, this is {1,0,0},{0,1,0},{0,0,1} or a cubic system.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_dip_getunitcell)
	{
		lua_pushstring(L, "Get the unit cell of a lattice site");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 tables: The A, B and C vectors defining the unit cell. By default, this is {1,0,0},{0,1,0},{0,0,1} or a cubic system.");
		return 3;
	}

	if(func == l_dip_settrunc)
	{
		lua_pushstring(L, "Set the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_dip_gettrunc)
	{
		lua_pushstring(L, "Get the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		return 3;
	}

	return 0;
}

static Encodable* newThing()
{
	return new Dipole;
}

void registerDipole(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_dip_gc},
		{"__tostring",   l_dip_tostring},
		{"apply",        l_dip_apply},
// 		{"threadApply",  l_dip_threadapply},
		{"setStrength",  l_dip_setstrength},
		{"strength",     l_dip_getstrength},
		{"setUnitCell",  l_dip_setunitcell},
		{"unitCell",     l_dip_getunitcell},
		{"setTruncation",l_dip_settrunc},
		{"truncation",   l_dip_gettrunc},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.dipole");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_dip_new},
		{"help",                l_dip_help},
		{"metatable",           l_dip_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Dipole", functions);
	lua_pop(L,1);

	Factory_registerItem(ENCODE_DIPOLE, newThing, lua_pushDipole, "Dipole");
	
}

extern "C"
{
DIPOLE_API int lib_register(lua_State* L);
DIPOLE_API int lib_version(lua_State* L);
DIPOLE_API const char* lib_name(lua_State* L);
DIPOLE_API int lib_main(lua_State* L, int argc, char** argv);
}

DIPOLE_API int lib_register(lua_State* L)
{
	registerDipole(L);
	return 0;
}

DIPOLE_API int lib_version(lua_State* L)
{
	return __revi;
}


DIPOLE_API const char* lib_name(lua_State* L)
{
	return "Dipole";
}

DIPOLE_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}


