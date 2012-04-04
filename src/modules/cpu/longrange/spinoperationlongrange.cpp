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
#include "info.h"
#include "spinoperationlongrange.h"

#include <stdlib.h>
#include <math.h>

#if defined NDEBUG || defined __OPTIMIZE__
#define DDD
#else
#define DDD printf("(%s:%i)\n", __FILE__, __LINE__);
#endif

LongRange::LongRange(const char* Name, const int field_slot, int nx, int ny, int nz, const int encode_tag)
	: SpinOperation(Name, field_slot, nx, ny, nz, encode_tag)
{
    qXX = 0;
    XX = 0;

	forward = 0;
    backward = 0;

	g = 1;
	gmax = 2000;
	matrixLoaded = false;
	newdata = true;
	XX = 0;
	ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
	ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
	ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	hasMatrices = false;
}
int LongRange::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	LongRange::init();
	return 0;	
}

void LongRange::push(lua_State* L)
{
	luaT_push<LongRange>(L, this);
}


void LongRange::init()
{
	if(XX) return;

    ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
    ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
    ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	deinit();
    int s = nx*ny * nz;

	hqx = new complex<double> [s];
	hqy = new complex<double> [s];
	hqz = new complex<double> [s];

	hrx = new complex<double> [s];
	hry = new complex<double> [s];
	hrz = new complex<double> [s];

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
								
 	XX = new double[s];
	XY = new double[s];
	XZ = new double[s];
	YY = new double[s];
	YZ = new double[s];
	ZZ = new double[s];
	
	hasMatrices = false;
}

static int offsetOK(int nx, int ny, int nz,  int x, int y, int z, int& offset)
{
	if(x<0 || x >= nx) return 0;
	if(y<0 || y >= ny) return 0;
	if(z<0 || z >= nz) return 0;
	
	offset = x + y*nx + z*nx*ny;
	return 1;
}

#define getsetPattern(AB) \
double LongRange::get##AB (int ox, int oy, int oz) \
{ \
    ox = (ox + 10*nx)%nx; \
    oy = (oy + 10*ny)%ny; \
    loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
		return AB [offset]; \
	return 0; \
} \
 \
void   LongRange::set##AB (int ox, int oy, int oz, double value) \
{ \
    ox = (ox + 10*nx)%nx; \
    oy = (oy + 10*ny)%ny; \
	loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
	{ \
		AB [offset] = value; \
		newdata = true; \
	} \
} 

getsetPattern(XX)
getsetPattern(XY)
getsetPattern(XZ)
getsetPattern(YY)
getsetPattern(YZ)
getsetPattern(ZZ)

double LongRange::getAB(int matrix, int ox, int oy, int oz)
{
	switch(matrix)
	{
		case 0:	return getXX(ox,oy,oz);
		case 1:	return getXY(ox,oy,oz);
		case 2:	return getXZ(ox,oy,oz);
		case 3:	return getYY(ox,oy,oz);
		case 4:	return getYZ(ox,oy,oz);
		case 5:	return getZZ(ox,oy,oz);
	}
	return 0;
}

void  LongRange::setAB(int matrix, int ox, int oy, int oz, double value)
{
	switch(matrix)
	{
		case 0:	setXX(ox,oy,oz,value); break;
		case 1:	setXY(ox,oy,oz,value); break;
		case 2:	setXZ(ox,oy,oz,value); break;
		case 3:	setYY(ox,oy,oz,value); break;
		case 4:	setYZ(ox,oy,oz,value); break;
		case 5:	setZZ(ox,oy,oz,value); break;
	}
}

void LongRange::loadMatrix()
{
	if(matrixLoaded) return;
	init();
	loadMatrixFunction(XX, XY, XZ, YY, YZ, ZZ); //implemented by child
	newdata = true;
	matrixLoaded = true;
}

void LongRange::deinit()
{
	if(qXX)
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
		qXX = 0;
	}
	if(XX)
	{
		delete [] XX;
		delete [] XY;
		delete [] XZ;
		delete [] YY;
		delete [] YZ;
		delete [] ZZ;
		XX = 0;
	}
	
	hasMatrices = false;
}

LongRange::~LongRange()
{
	deinit();
}

void LongRange::getMatrices()
{
	init();
	
	loadMatrix();

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
		for(int k=0; k<nz; k++)
		{
			for(int i=0; i<nx*ny; i++)
				r[i] = complex<double>(arrs[a][k*nx*ny + i],0);
		
			fftw_complex* fr = reinterpret_cast<fftw_complex*>(r);
			fftw_complex* fq = reinterpret_cast<fftw_complex*>(q);
//            printf("%p %p %p %p %p\n", forward, r, q, fr, fq);
            fftw_execute_dft(forward, fr, fq);

			for(int i=0; i<nx*ny; i++)
				qarrs[a][k*nx*ny + i] = q[i];
		}
	
	
	
	delete [] q;
	delete [] r;

	newdata = false;
	hasMatrices = true;
}

void LongRange::ifftAppliedForce(SpinSystem* ss)
{
	double d = g * global_scale / (double)(nx*ny);
// 	printf("%g\n", d);
	double* hx = ss->hx[slot];
	double* hy = ss->hy[slot];
	double* hz = ss->hz[slot];
	const int nxy = nx*ny;
	
    fftw_complex* q;
    fftw_complex* r;
	for(int i=0; i<nz; i++)
	{
        q = reinterpret_cast<fftw_complex*>(&hqx[i*nxy]);
        r = reinterpret_cast<fftw_complex*>(&hrx[i*nxy]);
        //printf("%p %p %p\n", backward, q, r);
        fftw_execute_dft(backward, q, r);

        q = reinterpret_cast<fftw_complex*>(&hqy[i*nxy]);
        r = reinterpret_cast<fftw_complex*>(&hry[i*nxy]);
        //printf("%p %p %p\n", backward, q, r);
        fftw_execute_dft(backward, q, r);

        q = reinterpret_cast<fftw_complex*>(&hqz[i*nxy]);
        r = reinterpret_cast<fftw_complex*>(&hrz[i*nxy]);
        //printf("%p %p %p\n", backward, q, r);
        fftw_execute_dft(backward, q, r);
    }

	for(int i=0; i<nxyz; i++)
		hx[i] = hrx[i].real() * d;

	for(int i=0; i<nxyz; i++)
		hy[i] = hry[i].real() * d;

	for(int i=0; i<nxyz; i++)
		hz[i] = hrz[i].real() * d;
}


void LongRange::collectIForces(SpinSystem* ss)
{
	int c;
	int sourceLayer, targetLayer;// !!Source layer, Target Layer
	int sourceOffset;
	int targetOffset;
	int demagOffset;
	const int nxy = nx*ny;
	
	complex<double>* sqx = ss->qx;
	complex<double>* sqy = ss->qy;
	complex<double>* sqz = ss->qz;

// 	if(!hasMatrices)
// 		getMatrices();
	
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
		int offset = sourceLayer - targetLayer;
		double sign = 1.0;
		if(offset < 0)
		{
			offset = -offset;
			sign = -sign;
		}
	
		targetOffset = targetLayer * nxy;
		sourceOffset = sourceLayer * nxy;
		demagOffset  = offset * nxy;
		//these are complex multiplies and adds
		for(c=0; c<nxy; c++) hqx[cTo]+=qXX[cDo]*sqx[cSo];
		for(c=0; c<nxy; c++) hqx[cTo]+=qXY[cDo]*sqy[cSo];
		for(c=0; c<nxy; c++) hqx[cTo]+=qXZ[cDo]*sqz[cSo]*sign;

		for(c=0; c<nxy; c++) hqy[cTo]+=qXY[cDo]*sqx[cSo];
		for(c=0; c<nxy; c++) hqy[cTo]+=qYY[cDo]*sqy[cSo];
		for(c=0; c<nxy; c++) hqy[cTo]+=qYZ[cDo]*sqz[cSo]*sign;

		for(c=0; c<nxy; c++) hqz[cTo]+=qXZ[cDo]*sqx[cSo]*sign;
		for(c=0; c<nxy; c++) hqz[cTo]+=qYZ[cDo]*sqy[cSo]*sign;
		for(c=0; c<nxy; c++) hqz[cTo]+=qZZ[cDo]*sqz[cSo];
	}
}


bool LongRange::apply(SpinSystem* ss)
{
//	DDD
	markSlotUsed(ss);

//	DDD
	if(newdata)
		getMatrices();
	
//	DDD
	ss->fft();
//	DDD
	collectIForces(ss);
//	DDD
	ifftAppliedForce(ss);
//	DDD

	return true;
}







static int l_setstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRange, lr, 1);
	lr->g = lua_tonumber(L, 2);
	return 0;
}
static int l_getstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRange, lr, 1);
	lua_pushnumber(L, lr->g);

	return 1;
}

static int l_setunitcell(lua_State* L)
{
	LUA_PREAMBLE(LongRange, lr, 1);
	
	double A[3];
	double B[3];
	double C[3];
	
	int r1 = lua_getNdouble(L, 3, A, 2, 0);
	int r2 = lua_getNdouble(L, 3, B, 2+r1, 0);
	int r3 = lua_getNdouble(L, 3, C, 2+r1+r2, 0);
	
	for(int i=0; i<3; i++)
	{
		lr->ABC[i+0] = A[i];
		lr->ABC[i+3] = B[i];
		lr->ABC[i+6] = C[i];
	}

	return 0;
}
static int l_getunitcell(lua_State* L)
{
	LUA_PREAMBLE(LongRange, lr, 1);

	double* ABC[3];
	ABC[0] = &(lr->ABC[0]);
	ABC[1] = &(lr->ABC[3]);
	ABC[2] = &(lr->ABC[6]);
	
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
static int l_settrunc(lua_State* L)
{
	LUA_PREAMBLE(LongRange, lr, 1);

	lua_getglobal(L, "math");
	lua_pushstring(L, "huge");
	lua_gettable(L, -2);
	lua_pushvalue(L, 2);
	int huge = lua_equal(L, -2, -1);
	
	if(huge)
	{
		lr->gmax = -1;
	}
	else
	{
		lr->gmax = lua_tointeger(L, 2);
	}
	return 0;
}
static int l_gettrunc(lua_State* L)
{
	LUA_PREAMBLE(LongRange, lr, 1);

	if(lr->gmax == -1)
	{
		lua_getglobal(L, "math");
		lua_pushstring(L, "huge");
		lua_gettable(L, -2);
		lua_remove(L, -2);//remove table (not really needed);
	}
	else
		lua_pushnumber(L, lr->gmax);

	return 1;
}

static int l_setmatrix(lua_State* L)
{
	LUA_PREAMBLE(LongRange, lr, 1);
	const char* badname = "1st argument must be matrix name: XX, XY, XZ, YY, YZ or ZZ";
	
	if(!lua_isstring(L, 2))
	    return luaL_error(L, badname);

	const char* type = lua_tostring(L, 2);

	const char* names[6] = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
	int mat = -1;
	for(int i=0; i<6; i++)
	{
	    if(strcasecmp(type, names[i]) == 0)
	    {
			mat = i;
	    }
	}

	if(mat < 0)
	    return luaL_error(L, badname);


	int offset[3];

	int r1 = lua_getNint(L, 3, offset, 3, 0);
        if(r1<0)
	    return luaL_error(L, "invalid offset");

	double val = lua_tonumber(L, 3+r1);

	// not altering zero base here:
	lr->setAB(mat, offset[0], offset[1], offset[2], val);

	return 0;
}

static int l_getmatrix(lua_State* L)
{
	LUA_PREAMBLE(LongRange, lr, 1);

	const char* badname = "1st argument must be matrix name: XX, XY, XZ, YY, YZ or ZZ";
	
	if(!lua_isstring(L, 2))
	    return luaL_error(L, badname);

	const char* type = lua_tostring(L, 2);

	const char* names[6] = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
	int mat = -1;
	for(int i=0; i<6; i++)
	{
	    if(strcasecmp(type, names[i]) == 0)
	    {
			mat = i;
	    }
	}

	if(mat < 0)
	    return luaL_error(L, badname);

	int offset[3];

	int r1 = lua_getNint(L, 3, offset, 3, 0);
        if(r1<0)
	    return luaL_error(L, "invalid offset");

	// not altering zero base here:
	double val = lr->getAB(mat, offset[0], offset[1], offset[2]);
	lua_pushnumber(L, val);
	return 1;
}




int LongRange::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates a Long Range field for a *SpinSystem*. This is an abstract base class inherited by other operators. This operator by itself does nothing.");
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

	
	
	if(func == l_setstrength)
	{
		lua_pushstring(L, "Set the strength of the Long Range Field");
		lua_pushstring(L, "1 number: strength of the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Long Range Field");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: strength of the field");
		return 3;
	}
	
	if(func == l_setunitcell)
	{
		lua_pushstring(L, "Set the unit cell of a lattice site");
		lua_pushstring(L, "3 *3Vector*: The A, B and C vectors defining the unit cell. By default, this is {1,0,0},{0,1,0},{0,0,1} or a cubic system.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_getunitcell)
	{
		lua_pushstring(L, "Get the unit cell of a lattice site");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 tables: The A, B and C vectors defining the unit cell. By default, this is {1,0,0},{0,1,0},{0,0,1} or a cubic system.");
		return 3;
	}

	if(func == l_settrunc)
	{
		lua_pushstring(L, "Set the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to. If set to math.huge then extrapolation will be used to approximate infinite radius.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_gettrunc)
	{
		lua_pushstring(L, "Get the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		return 3;
	}
	if(func == l_getmatrix)
	{
		lua_pushstring(L, "Get an element of an interaction matrix");
		lua_pushstring(L, "1 string, 1 *3Vector*: The string indicates which AB matrix to access. Can be XX, XY, XZ, YY, YZ or ZZ. The *3Vector* indexes into the matrix. Note: indexes are zero-based and are interpreted as offsets.");
		lua_pushstring(L, "1 number: The fetched value.");
		return 3;
	}

	if(func == l_setmatrix)
	{
		lua_pushstring(L, "Set an element of an interaction matrix");
		lua_pushstring(L, "1 string, 1 *3Vector*, 1 number: The string indicates which AB matrix to access. Can be XX, XY, XZ, YY, YZ or ZZ. The *3Vector* indexes into the matrix. The number is the value that is set at the index. Note: indexes are zero-based and are interpreted as offsets.");
		lua_pushstring(L, "");
		return 3;
	}
	
	return SpinOperation::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* LongRange::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setStrength",  l_setstrength},
		{"strength",     l_getstrength},
		{"setUnitCell",  l_setunitcell},
		{"unitCell",     l_getunitcell},
		{"setTruncation",l_settrunc},
		{"truncation",   l_gettrunc},
		{"getMatrix",    l_getmatrix},
		{"setMatrix",    l_setmatrix},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}





extern "C"
{
LONGRANGE_API int lib_register(lua_State* L);
LONGRANGE_API int lib_version(lua_State* L);
LONGRANGE_API const char* lib_name(lua_State* L);
LONGRANGE_API int lib_main(lua_State* L);
}

LONGRANGE_API int lib_register(lua_State* L)
{
	luaT_register<LongRange>(L);
	return 0;
}

LONGRANGE_API int lib_version(lua_State* L)
{
	return __revi;
}

LONGRANGE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LongRange";
#else
	return "LongRange-Debug";
#endif
}

LONGRANGE_API int lib_main(lua_State* L)
{
	return 0;
}



