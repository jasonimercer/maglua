/******************************************************************************
* Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/
// spinoperationmagnetostatics
#include "spinoperationmagnetostatics.h"
#include "spinsystem.h"
#include "magnetostaticssupport.h"

#include <stdlib.h>
#include <math.h>

Magnetostatic::Magnetostatic(int nx, int ny, int nz)
	: SpinOperation("Magnetostatic",DIPOLE_SLOT, nx, ny, nz)
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

	volumeDimensions[0] = 1;
	volumeDimensions[1] = 1;
	volumeDimensions[2] = 1;
	
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
	
	crossover_tolerance = 0.0001;
}

void Magnetostatic::encode(buffer* b) const
{
	
}

int  Magnetostatic::decode(buffer* b)
{
	return 0;
}

Magnetostatic::~Magnetostatic()
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

void Magnetostatic::getMatrices()
{
	int s = nx*ny * (nz*2-1);
	double* XX = new double[s];
	double* XY = new double[s];
	double* XZ = new double[s];
	double* YY = new double[s];
	double* YZ = new double[s];
	double* ZZ = new double[s];
	
	magnetostaticsLoad(
		nx, ny, nz,
		gmax, ABC,
		volumeDimensions,
		XX, XY, XZ,
		YY, YZ, ZZ, crossover_tolerance);

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

void Magnetostatic::ifftAppliedForce(SpinSystem* ss)
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


void Magnetostatic::collectIForces(SpinSystem* ss)
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

bool Magnetostatic::apply(SpinSystem* ss)
{
	markSlotUsed(ss);

	ss->fft();
	collectIForces(ss);
	ifftAppliedForce(ss);

	return true;
}







Magnetostatic* checkMagnetostatic(lua_State* L, int idx)
{
	Magnetostatic** pp = (Magnetostatic**)luaL_checkudata(L, idx, "MERCER.magnetostatics");
    luaL_argcheck(L, pp != NULL, 1, "`Magnetostatic' expected");
    return *pp;
}

void lua_pushMagnetostatic(lua_State* L, Magnetostatic* mag)
{
	mag->refcount++;
	Magnetostatic** pp = (Magnetostatic**)lua_newuserdata(L, sizeof(Magnetostatic**));
	
	*pp = mag;
	luaL_getmetatable(L, "MERCER.magnetostatics");
	lua_setmetatable(L, -2);
}

int l_mag_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	lua_pushMagnetostatic(L, new Magnetostatic(n[0], n[1], n[2]));
	return 1;
}


int l_mag_setstrength(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	mag->g = lua_tonumber(L, 2);
	return 0;
}

int l_mag_gc(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	mag->refcount--;
	if(mag->refcount == 0)
		delete mag;
	
	return 0;
}

int l_mag_apply(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!mag->apply(ss))
		return luaL_error(L, mag->errormsg.c_str());
	
	return 0;
}

int l_mag_getstrength(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	lua_pushnumber(L, mag->g);

	return 1;
}
int l_mag_setunitcell(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	for(int i=0; i<9; i++)
		mag->ABC[i] = lua_tonumber(L, i+2);

	return 0;
}
int l_mag_getunitcell(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	for(int i=0; i<9; i++)
		lua_pushnumber(L, mag->ABC[i]);

	return 9;
}
int l_mag_settrunc(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	mag->gmax = lua_tointeger(L, 2);

	return 0;
}
int l_mag_gettrunc(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	lua_pushnumber(L, mag->gmax);

	return 1;
}

static int l_mag_tostring(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	
	lua_pushfstring(L, "Magnetostatic (%dx%dx%d)", mag->nx, mag->ny, mag->nz);
	
	return 1;
}

static int l_mag_setcelldims(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	if(lua_getNdouble(L, 3, mag->volumeDimensions, 2, 1) < 0)
		return luaL_error(L, "Magnetostatic.setCellDimensions requires 3 values");

	return 0;
}

static int l_mag_getcelldims(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	
	for(int i=0; i<3; i++)
		lua_pushnumber(L, mag->volumeDimensions[i]);

	return 3;
}

static int l_mag_setcrossover(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	
	mag->crossover_tolerance = lua_tonumber(L, 2);
	return 0;
}

static int l_mag_getcrossover(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	
	lua_pushnumber(L, mag->crossover_tolerance);
	return 1;
}



static int l_mag_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.magnetostatics");
	return 1;
}

static int l_mag_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the Magnetostatic field of a *SpinSystem*");
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
	
	if(func == l_mag_new)
	{
		lua_pushstring(L, "Create a new Magnetostatic Operator.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Magnetostatic object");
		return 3;
	}
	
	
	if(func == l_mag_apply)
	{
		lua_pushstring(L, "Calculate the Magnetostatic field of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This spin system will receive the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_setstrength)
	{
		lua_pushstring(L, "Set the strength of the Dipolar Field");
		lua_pushstring(L, "1 number: strength of the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Dipolar Field");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: strength of the field");
		return 3;
	}
	
	if(func == l_mag_setunitcell)
	{
		lua_pushstring(L, "Set the unit cell of a lattice site");
		lua_pushstring(L, "9 numbers: The A, B and C vectors defining the unit cell. By default, this is (1,0,0,0,1,0,0,0,1) or a cubic system.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_mag_getunitcell)
	{
		lua_pushstring(L, "Get the unit cell of a lattice site");
		lua_pushstring(L, "");
		lua_pushstring(L, "9 numbers: The A, B and C vectors defining the unit cell. By default, this is (1,0,0,0,1,0,0,0,1) or a cubic system.");
		return 3;
	}

	if(func == l_mag_settrunc)
	{
		lua_pushstring(L, "Set the truncation distance in spins of the Magnetostatic sum.");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_gettrunc)
	{
		lua_pushstring(L, "Get the truncation distance in spins of the Magnetostatic sum.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		return 3;
	}
	
	if(func == l_mag_setcelldims)
	{
		lua_pushstring(L, "Set the dimension of each Rectangular Prism");
		lua_pushstring(L, "1 *3Vector*: The x, y and z lengths of the prism");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_getcelldims)
	{
		lua_pushstring(L, "Get the dimension of each Rectangular Prism");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 Numbers: The x, y and z lengths of the prism");
		return 3;
	}
	
	if(func == l_mag_setcrossover)
	{
		lua_pushstring(L, "Set the relative error to define the crossover from magnetostatics to dipole calculations in the interaction matrix generation. Initial value is 0.0001.");
		lua_pushstring(L, "1 Number: The relative error for the crossover");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_getcrossover)
	{
		lua_pushstring(L, "Get the relative error to define the crossover from magnetostatics to dipole calculations in the interaction matrix generation. Initial value is 0.0001.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The relative error for the crossover");
		return 3;
	}
	
	return 0;
}


void registerMagnetostatic(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_mag_gc},
		{"__tostring",   l_mag_tostring},
		{"apply",        l_mag_apply},
		{"setStrength",  l_mag_setstrength},
		{"strength",     l_mag_getstrength},

		{"setCellDimensions", l_mag_setcelldims},
		{"cellDimensions",    l_mag_getcelldims},

		{"setUnitCell",  l_mag_setunitcell},
		{"unitCell",     l_mag_getunitcell},
		
		{"setTruncation",l_mag_settrunc},
		{"truncation",   l_mag_gettrunc},
		
		{"setCrossoverTolerance", l_mag_setcrossover},
		{"crossoverTolerance", l_mag_setcrossover},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.magnetostatics");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_mag_new},
		{"help",                l_mag_help},
		{"metatable",           l_mag_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Magnetostatic", functions);
	lua_pop(L,1);	
}

