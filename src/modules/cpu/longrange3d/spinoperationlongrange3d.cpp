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
#include "spinoperationlongrange3d.h"

#include <stdlib.h>
#include <math.h>

#if defined NDEBUG || defined __OPTIMIZE__
#define DDD
#else
#define DDD printf("(%s:%i)\n", __FILE__, __LINE__);
#endif

LongRange3D::LongRange3D(int nx, int ny, int nz)
	: SpinOperation("LongRange3D", DIPOLE_SLOT, nx, ny, nz, hash32("LongRange3D"))
{
    qXX = 0;
    XX = 0;
	ws1 = 0;
	g = 1;
	gmax[0] = 100;
	gmax[1] = 100;
	gmax[2] = 100;
	gmax[3] = 100;
	matrixLoaded = false;
	newdata = false;
	XX = 0;
	ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
	ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
	ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	hasMatrices = false;
}
int LongRange3D::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	LongRange3D::init();
	return 0;	
}

void LongRange3D::push(lua_State* L)
{
	luaT_push<LongRange3D>(L, this);
}


void LongRange3D::encode(buffer* b)
{
	SpinOperation::encode(b);
	encodeInteger(gmax[0], b);
	encodeInteger(gmax[1], b);
	encodeInteger(gmax[2], b);
	encodeInteger(gmax[3], b);
	encodeDouble(g, b);
	for(int i=0; i<9; i++)
	{
		encodeDouble(ABC[i], b);
	}
}

int  LongRange3D::decode(buffer* b)
{
	SpinOperation::decode(b);

	gmax[0] = decodeInteger(b);
	gmax[1] = decodeInteger(b);
	gmax[2] = decodeInteger(b);
	gmax[3] = decodeInteger(b);
	g = decodeDouble(b);

	for(int i=0; i<9; i++)
	{
		ABC[i] = decodeDouble(b);
	}
	return 0;
}



void LongRange3D::init()
{
	if(XX) return;

    ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
    ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
    ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	deinit();

	hqx = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hqy = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hqz = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	hrx = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hry = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hrz = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	qXX = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qXY = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qXZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	qYY = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qYZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qZZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	XX = luaT_inc<dArray>(new dArray(nx,ny,nz));
	XY = luaT_inc<dArray>(new dArray(nx,ny,nz));
	XZ = luaT_inc<dArray>(new dArray(nx,ny,nz));
	YY = luaT_inc<dArray>(new dArray(nx,ny,nz));
	YZ = luaT_inc<dArray>(new dArray(nx,ny,nz));
	ZZ = luaT_inc<dArray>(new dArray(nx,ny,nz));

	XX->zero();
	XY->zero();
	XZ->zero();
	YY->zero();
	YZ->zero();
	ZZ->zero();
	
	ws1 = new dcArray(nx,ny,nz);
	ws2 = new dcArray(nx,ny,nz);
	wsX = new dcArray(nx,ny,nz);
	wsY = new dcArray(nx,ny,nz);
	wsZ = new dcArray(nx,ny,nz);
	
	hasMatrices = false;
}

static int offsetOK(int nx, int ny, int nz,  int x, int y, int z, int& offset)
{
	if(x<0  || x >= nx) return 0;
	if(y<0  || y >= ny) return 0;
	if(z<0  || z >= nz) return 0;
	
	offset = x + y*nx + z*nx*ny;
	return 1;
}

#define getsetPattern(AB) \
double LongRange3D::get##AB (int ox, int oy, int oz) \
{ \
    ox = (ox + 100*nx)%nx; \
    oy = (oy + 100*ny)%ny; \
    oz = (oz + 100*nz)%nz; \
    loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
		return (*AB) [offset]; \
	return 0; \
} \
 \
void   LongRange3D::set##AB (int ox, int oy, int oz, double value) \
{ \
    ox = (ox + 100*nx)%nx; \
    oy = (oy + 100*ny)%ny; \
    oz = (oz + 100*nz)%nz; \
    loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
	{ \
		(*AB) [offset] = value; \
		newdata = true; \
	} \
} 

getsetPattern(XX)
getsetPattern(XY)
getsetPattern(XZ)
getsetPattern(YY)
getsetPattern(YZ)
getsetPattern(ZZ)

double LongRange3D::getAB(int matrix, int ox, int oy, int oz)
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

void  LongRange3D::setAB(int matrix, int ox, int oy, int oz, double value)
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


static void longrange3DLoad(
	const int nx, const int ny, const int nz,
	const int* gmax, double* ABC,
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ)
{

}

void LongRange3D::loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ)
{
	longrange3DLoad(
		nx, ny, nz,
		gmax, ABC,
		XX, XY, XZ,
		YY, YZ, ZZ);
}


void LongRange3D::loadMatrix()
{
	if(newdata)
	{
		matrixLoaded = true; //user made a custom matrix
		return;
	}
	
	if(matrixLoaded) return;
	init();
	loadMatrixFunction(XX->data(), XY->data(), XZ->data(), YY->data(), YZ->data(), ZZ->data()); //implemented by child

	matrixLoaded = true;
	newdata = true;
}

void LongRange3D::deinit()
{
	if(qXX)
	{
		luaT_dec<dcArray>(qXX); qXX=0;
		luaT_dec<dcArray>(qXY);
		luaT_dec<dcArray>(qXZ);

		luaT_dec<dcArray>(qYY);
		luaT_dec<dcArray>(qYZ);

		luaT_dec<dcArray>(qZZ);

		luaT_dec<dcArray>(hqx);
		luaT_dec<dcArray>(hqy);
		luaT_dec<dcArray>(hqz);

		luaT_dec<dcArray>(hrx);
		luaT_dec<dcArray>(hry);
		luaT_dec<dcArray>(hrz);
	}
	if(XX)
	{
		luaT_dec<dArray>(XX); XX=0;
		luaT_dec<dArray>(XY);
		luaT_dec<dArray>(XZ);
		luaT_dec<dArray>(YY);
		luaT_dec<dArray>(YZ);
		luaT_dec<dArray>(ZZ);
	}
	
	
	if(ws1)
	{
		delete ws1;
		delete ws2;
		delete wsX;
		delete wsY;
		delete wsZ;
		ws1 = 0;
	}
	hasMatrices = false;
}

LongRange3D::~LongRange3D()
{
	deinit();
}

bool LongRange3D::updateData()
{
	if(!matrixLoaded)
	{
		loadMatrix();
	}
	
	if(!newdata)
		return true;
	newdata = false;
	
	ws1->zero();
	arraySetRealPart(ws1->data(), XX->data(), ws1->nxyz);  ws1->fft3DTo(qXX);
	arraySetRealPart(ws1->data(), XY->data(), ws1->nxyz);  ws1->fft3DTo(qXY);
	arraySetRealPart(ws1->data(), XZ->data(), ws1->nxyz);  ws1->fft3DTo(qXZ);

	arraySetRealPart(ws1->data(), YY->data(), ws1->nxyz);  ws1->fft3DTo(qYY);
	arraySetRealPart(ws1->data(), YZ->data(), ws1->nxyz);  ws1->fft3DTo(qYZ);
	arraySetRealPart(ws1->data(), ZZ->data(), ws1->nxyz);  ws1->fft3DTo(qZZ);
	
	//prescaling by 1/xyz
	qXX->scaleAll(doubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qXY->scaleAll(doubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qXZ->scaleAll(doubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qYY->scaleAll(doubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qYZ->scaleAll(doubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qZZ->scaleAll(doubleComplex(1.0/((double)(nx*ny*nz)), 0));

	return true;
}



bool LongRange3D::apply(SpinSystem* ss)
{
	markSlotUsed(ss);
	updateData();
	const int nxyz = nx*ny*nz;

	doubleComplex one = luaT<doubleComplex>::one();
	
	ws1->zero(); arraySetRealPart(ws1->data(), ss->x->data(), nxyz); ws1->fft3DTo(wsX);
	ws1->zero(); arraySetRealPart(ws1->data(), ss->y->data(), nxyz); ws1->fft3DTo(wsY);
	ws1->zero(); arraySetRealPart(ws1->data(), ss->z->data(), nxyz); ws1->fft3DTo(wsZ);

	dcArray* sqx = wsX;
	dcArray* sqy = wsY;
	dcArray* sqz = wsZ;

	dArray* hx = ss->hx[slot];
	dArray* hy = ss->hy[slot];
	dArray* hz = ss->hz[slot];
	
	// HX
	ws1->zero();
	arrayMultAll(ws1->data(), qXX->data(), sqx->data(), nxyz);	//arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	arrayMultAll(ws2->data(), qXY->data(), sqy->data(), nxyz);	arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	arrayMultAll(ws2->data(), qXZ->data(), sqz->data(), nxyz);	arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	ws1->ifft3DTo(ws2);
	arrayGetRealPart(hx->data(),  ws2->data(), nxyz);
	
	// HY
	ws1->zero();
	arrayMultAll(ws1->data(), qXY->data(), sqx->data(), nxyz);	//arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	arrayMultAll(ws2->data(), qYY->data(), sqy->data(), nxyz);	arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	arrayMultAll(ws2->data(), qYZ->data(), sqz->data(), nxyz);	arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	ws1->ifft3DTo(ws2);
	arrayGetRealPart(hy->data(),  ws2->data(), nxyz);

	// HZ
	ws1->zero();
	arrayMultAll(ws1->data(), qXZ->data(), sqx->data(), nxyz);	//arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	arrayMultAll(ws2->data(), qYZ->data(), sqy->data(), nxyz);	arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	arrayMultAll(ws2->data(), qZZ->data(), sqz->data(), nxyz);	arraySumAll(ws1->data(), ws2->data(), ws1->data(), nxyz);
	ws1->ifft3DTo(ws2);
	arrayGetRealPart(hz->data(),  ws2->data(), nxyz);

	
	hx->scaleAll(g * global_scale);
	hy->scaleAll(g * global_scale);
	hz->scaleAll(g * global_scale);
	
// 	for(int i=0; i<nxyz; i++)
// 	{
// 		printf("%i  %g %g %g\n", i, hx->data()[i], hy->data()[i], hz->data()[i]);
// 	}
		
	return true;
}







static int l_setstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	lr->g = lua_tonumber(L, 2);
	return 0;
}
static int l_getstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	lua_pushnumber(L, lr->g);

	return 1;
}

static int l_setunitcell(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	
	double A[3];
	double B[3];
	double C[3];
	
	int r1 = lua_getNdouble(L, 3, A, 2, 0);
	int r2 = lua_getNdouble(L, 3, B, 2+r1, 0);
	/*int r3 =*/ lua_getNdouble(L, 3, C, 2+r1+r2, 0);
	
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
	LUA_PREAMBLE(LongRange3D, lr, 1);

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
	LUA_PREAMBLE(LongRange3D, lr, 1);

	if(lua_istable(L, 2))
	{
		lr->gmax[0] = 0;
		for(int i=1; i<=3; i++)
		{
			lua_pushinteger(L, i);
			lua_gettable(L, 2);
			if(lua_isnumber(L, -1))
			{
				lr->gmax[i] = lua_tointeger(L, -1);
				if(lr->gmax[i] <= 0)
					lr->gmax[i] = 1;
			}
			else
			{
				lr->gmax[i] = 1;
			}
			lua_pop(L, 1);
			if(lr->gmax[i] > lr->gmax[0])
				lr->gmax[0] = lr->gmax[i];
		}

		return 0;
	}
	if(lua_gettop(L) > 2)
	{
		lr->gmax[0] = 0;
		for(int i=1; i<=3; i++)
		{
			if(lua_isnumber(L, i+1))
			{
				lr->gmax[i] = lua_tointeger(L, i+1);
				if(lr->gmax[i] <= 0)
					lr->gmax[i] = 1;
			}
			else
			{
				lr->gmax[i] = 1;
			}
			if(lr->gmax[i] > lr->gmax[0])
				lr->gmax[0] = lr->gmax[i];
		}
		
		return 0;
	}
	
	lua_getglobal(L, "math");
	lua_pushstring(L, "huge");
	lua_gettable(L, -2);
	lua_pushvalue(L, 2);
	int huge = lua_equal(L, -2, -1);
	
	if(huge)
	{
		lr->gmax[0] = -1;
	}
	else
	{
		lr->gmax[0] = lua_tointeger(L, 2);
	}
	for(int i=0; i<3; i++)
	{
		lr->gmax[i+1] = lr->gmax[0];
	}
	return 0;
}
static int l_gettrunc(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);

	for(int i=0; i<4; i++)
	{
		if(lr->gmax[i] == -1)
		{
			lua_getglobal(L, "math");
			lua_pushstring(L, "huge");
			lua_gettable(L, -2);
			lua_remove(L, -2);//remove table (not really needed);
		}
		else
			lua_pushnumber(L, lr->gmax[i]);
	}

	return 4;
}


static int l_setmatrix(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
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
	// printf("mat(%i) {%i %i %i} %g\n", mat, offset[0], offset[1], offset[2], val);
	lr->setAB(mat, offset[0], offset[1], offset[2], val);

	return 0;
}

static int l_getmatrix(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);

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

static int l_getarray(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
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
	
	switch(mat)
	{
		case 0: luaT_push<dArray>(L, lr->XX); break;
		case 1: luaT_push<dArray>(L, lr->XY); break;
		case 2: luaT_push<dArray>(L, lr->XZ); break;
		case 3: luaT_push<dArray>(L, lr->YY); break;
		case 4: luaT_push<dArray>(L, lr->YZ); break;
		case 5: luaT_push<dArray>(L, lr->ZZ); break;
	}
	
	return 1;
}

static int l_setarray(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
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

	dArray* a = luaT_to<dArray>(L, 3);
	if(!a)
		return luaL_error(L, "2nd argument must be an array");
	
	if(!a->sameSize(lr->XX))
		return luaL_error(L, "Array size mismatch");
	
	luaT_inc<dArray>(a);
	
	switch(mat)
	{
		case 0: luaT_dec<dArray>(lr->XX); lr->XX = a; break;
		case 1: luaT_dec<dArray>(lr->XY); lr->XY = a; break;
		case 2: luaT_dec<dArray>(lr->XZ); lr->XZ = a; break;
		case 3: luaT_dec<dArray>(lr->YY); lr->YY = a; break;
		case 4: luaT_dec<dArray>(lr->YZ); lr->YZ = a; break;
		case 5: luaT_dec<dArray>(lr->ZZ); lr->ZZ = a; break;
	}
	return 0;
}


int LongRange3D::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates a Long Range 3D PBC field for a *SpinSystem*. Tensor elements must be populated explicitly.");
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
		lua_pushstring(L, "1 Integer or Table of 3 Integers or 3 Integers: Radius of spins to sum out to. If set to math.huge then extrapolation will be used to approximate infinite radius. If input is more than 1 value then the input is considered as the hard truncation limit for each Cartesian coordinate.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_gettrunc)
	{
		lua_pushstring(L, "Get the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "");
		lua_pushstring(L, "4 Integers: Radius of spins to sum out to, hard Limit in X, Y and Z direction.");
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
	
	if(func == l_setarray)
	{
		lua_pushstring(L, "Set a named interaction matrix to a new array");
		lua_pushstring(L, "1 string, 1 Array: The string indicates which AB matrix to set. Can be XX, XY, XZ, YY, YZ or ZZ. The Array must be of appropriate dimensions");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getarray)
	{
		lua_pushstring(L, "Get a named interaction matrix as an array");
		lua_pushstring(L, "1 string: The string indicates which AB matrix to access. Can be XX, XY, XZ, YY, YZ or ZZ. ");
		lua_pushstring(L, "1 Array: The interaction matrix for given AB components");
		return 3;
	}
	
	return SpinOperation::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* LongRange3D::luaMethods()
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
		{"getArray",    l_getarray},
		{"setArray",    l_setarray},
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
	luaT_register<LongRange3D>(L);
	return 0;
}

LONGRANGE_API int lib_version(lua_State* L)
{
	return __revi;
}

LONGRANGE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LongRange3D";
#else
	return "LongRange3D-Debug";
#endif
}

LONGRANGE_API int lib_main(lua_State* L)
{
	return 0;
}



