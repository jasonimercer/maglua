#include "spinoperationdipole.h"
#include "spinsystem.h"
#include "dipolesupport.h"

#include <stdlib.h>
#include <math.h>

Dipole::Dipole(int nx, int ny, int nz)
	: SpinOperation("Dipole", DIPOLE_SLOT, nx, ny, nz, ENCODE_DIPOLE)
{
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

	g = 1;
	gmax = 2000;

	ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
	ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
	ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

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

void Dipole::encode(buffer* b) const
{
	
}

int  Dipole::decode(buffer* b)
{
	
}

Dipole::~Dipole()
{
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
}

void Dipole::getMatrices()
{
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

	for(int i=0; i<nz; i++)
	{
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqx[i*nxyz]),
				reinterpret_cast<fftw_complex*>(&hrx[i*nxyz]));
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqy[i*nxyz]),
				reinterpret_cast<fftw_complex*>(&hry[i*nxyz]));
		fftw_execute_dft(backward, 
				reinterpret_cast<fftw_complex*>(&hqz[i*nxyz]),
				reinterpret_cast<fftw_complex*>(&hrz[i*nxyz]));
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
	unsigned int c;
	int sourceLayer, targetLayer;// !!Source layer, Target Layer
	int sourceOffset;
	int targetOffset;
	int demagOffset;
	const int nxy = nx*ny;
	int LL = nz*2-1;
	
	complex<double>* sqx = ss->qx;
	complex<double>* sqy = ss->qy;
	complex<double>* sqz = ss->qz;

	if(!hasMatrices)
		getMatrices();
	
	complex<double> cz(0,0);
	for(c=0; c<nxy; c++) hqx[c] = cz;
	for(c=0; c<nxy; c++) hqy[c] = cz;
	for(c=0; c<nxy; c++) hqz[c] = cz;

	
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
		for(c=0; c<nxyz; c++) hqx[cTo]+=qXX[cDo]*sqx[cSo];
		for(c=0; c<nxyz; c++) hqx[cTo]+=qXY[cDo]*sqy[cSo];
		for(c=0; c<nxyz; c++) hqx[cTo]+=qXZ[cDo]*sqz[cSo];

		for(c=0; c<nxyz; c++) hqy[cTo]+=qXY[cDo]*sqx[cSo];
		for(c=0; c<nxyz; c++) hqy[cTo]+=qYY[cDo]*sqy[cSo];
		for(c=0; c<nxyz; c++) hqy[cTo]+=qYZ[cDo]*sqz[cSo];

		for(c=0; c<nxyz; c++) hqz[cTo]+=qXZ[cDo]*sqx[cSo];
		for(c=0; c<nxyz; c++) hqz[cTo]+=qYZ[cDo]*sqy[cSo];
		for(c=0; c<nxyz; c++) hqz[cTo]+=qZZ[cDo]*sqz[cSo];
	}
}

bool Dipole::apply(SpinSystem* ss)
{
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


int l_dip_new(lua_State* L)
{
	if(lua_gettop(L) != 3)
		return luaL_error(L, "Dipole.new requires nx, ny, nz");

	Dipole* dip = new Dipole(
			lua_tointeger(L, 1),
			lua_tointeger(L, 2),
			lua_tointeger(L, 3)
	);
	dip->refcount++;
	
	if(lua_isnumber(L, 4))
	{
		dip->g = lua_tonumber(L, 4);
	}

	Dipole** pp = (Dipole**)lua_newuserdata(L, sizeof(Dipole**));
	
	*pp = dip;
	luaL_getmetatable(L, "MERCER.dipole");
	lua_setmetatable(L, -2);
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

	for(int i=0; i<9; i++)
		dip->ABC[i] = lua_tonumber(L, i+2);

	return 0;
}
int l_dip_getunitcell(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	for(int i=0; i<9; i++)
		lua_pushnumber(L, dip->ABC[i]);

	return 9;
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

void registerDipole(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_dip_gc},
		{"apply",        l_dip_apply},
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
		{NULL, NULL}
	};
		
	luaL_register(L, "Dipole", functions);
	lua_pop(L,1);	
}

