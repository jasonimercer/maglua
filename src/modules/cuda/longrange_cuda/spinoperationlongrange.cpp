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

#include "spinoperationlongrange.h"
#include "spinsystem.h"
#include "spinsystem.hpp"
#include "info.h"

#include <stdlib.h>
#include <math.h>

LongRangeCuda::LongRangeCuda(const char* name, const int field_slot, int nx, int ny, int nz, const int encode_tag)
	: SpinOperation(name, field_slot, nx, ny, nz, encode_tag)
{
	g = 1;
	gmax = 2000;

	ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
	ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
	ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	matrixLoaded = false;
	newHostData = false;
	XX = 0;
	registerWS();
}

int LongRangeCuda::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	init();
	return 0;	
}

void LongRangeCuda::push(lua_State* L)
{
	luaT_push<LongRangeCuda>(L, this);
}

void LongRangeCuda::encode(buffer* b)
{
	SpinOperation::encode(b);
	encodeInteger(gmax, b);
	encodeDouble(g, b);
	for(int i=0; i<9; i++)
	{
		encodeDouble(ABC[i], b);
	}
}

// implemented by child classes. This is here incase someone instanstiates an empty longrange operator
void LongRangeCuda::loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ)
{
	for(int i=0; i<nxyz; i++)
		XX[i] = 0;

	memcpy(XY, XX, sizeof(double)*nxyz);
	memcpy(XZ, XX, sizeof(double)*nxyz);
	memcpy(YY, XX, sizeof(double)*nxyz);
	memcpy(YZ, XX, sizeof(double)*nxyz);
	memcpy(ZZ, XX, sizeof(double)*nxyz);
}

int  LongRangeCuda::decode(buffer* b)
{
	SpinOperation::decode(b);

	gmax = decodeInteger(b);
	g = decodeDouble(b);

	for(int i=0; i<9; i++)
	{
		ABC[i] = decodeDouble(b);
	}
	return 0;
}


void LongRangeCuda::init()
{
    if(XX) return;
	deinit();
	
	XX = luaT_inc<dArray>(new dArray(nx,ny,nz));
	XY = luaT_inc<dArray>(new dArray(nx,ny,nz));
	XZ = luaT_inc<dArray>(new dArray(nx,ny,nz));

	YY = luaT_inc<dArray>(new dArray(nx,ny,nz));
	YZ = luaT_inc<dArray>(new dArray(nx,ny,nz));
	ZZ = luaT_inc<dArray>(new dArray(nx,ny,nz));
	
	
	qXX = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qXY = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qXZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	qYY = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qYZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qZZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
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
double LongRangeCuda::get##AB (int ox, int oy, int oz) \
{ \
    ox = (ox + 10*nx)%nx; \
    oy = (oy + 10*ny)%ny; \
    loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
		return AB->data() [offset]; \
	return 0; \
} \
 \
void   LongRangeCuda::set##AB (int ox, int oy, int oz, double value) \
{ \
    ox = (ox + 10*nx)%nx; \
    oy = (oy + 10*ny)%ny; \
	loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
	{ \
		AB->data() [offset] = value; \
		newHostData = true; \
	} \
} 

getsetPattern(XX)
getsetPattern(XY)
getsetPattern(XZ)
getsetPattern(YY)
getsetPattern(YZ)
getsetPattern(ZZ)

double LongRangeCuda::getAB(int matrix, int ox, int oy, int oz)
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

void  LongRangeCuda::setAB(int matrix, int ox, int oy, int oz, double value)
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
void LongRangeCuda::deinit()
{
	
	if(XX)
	{
		luaT_dec<dArray>(XX);
		luaT_dec<dArray>(XY);
		luaT_dec<dArray>(XZ);
		luaT_dec<dArray>(YY);
		luaT_dec<dArray>(YZ);
		luaT_dec<dArray>(ZZ);
		
		luaT_dec<dcArray>(qXX);
		luaT_dec<dcArray>(qXY);
		luaT_dec<dcArray>(qXZ);
		luaT_dec<dcArray>(qYY);
		luaT_dec<dcArray>(qYZ);
		luaT_dec<dcArray>(qZZ);

		XX = 0;
	}
}

LongRangeCuda::~LongRangeCuda()
{
	unregisterWS();
	deinit();
}

void LongRangeCuda::loadMatrix()
{
	if(newHostData)
	{
		matrixLoaded = true; //user made a custom matrix
		return;
	}
	
	if(matrixLoaded) return;
	init();
	loadMatrixFunction(XX->data(), XY->data(), XZ->data(), YY->data(), YZ->data(), ZZ->data()); //implemented by child

	matrixLoaded = true;
	newHostData = true;
}

static void setRealZeroImagCpu(dcArray* d, dArray* s)
{
	for(int i=0; i<d->nxyz; i++)
	{
		d->data()[i].x = s->data()[i];
		d->data()[i].y = 0;
	}
}

bool LongRangeCuda::updateData()
{
	if(!matrixLoaded)
	{
		loadMatrix();
	}
	
	if(!newHostData)
		return true;
	newHostData = false;
	
	doubleComplex* d_ws1;
	doubleComplex* d_ws2;
	
	getWSMem2(&d_ws1, sizeof(doubleComplex)*nxyz, 
			  &d_ws2, sizeof(doubleComplex)*nxyz);
	
	dcArray ws1(nx,ny,nz, d_ws1);
	
	setRealZeroImagCpu(&ws1, XX); ws1.new_host = true; ws1.new_device = false; ws1.fft2DTo(qXX, d_ws2);
	setRealZeroImagCpu(&ws1, XY); ws1.new_host = true; ws1.new_device = false; ws1.fft2DTo(qXY, d_ws2);
	setRealZeroImagCpu(&ws1, XZ); ws1.new_host = true; ws1.new_device = false; ws1.fft2DTo(qXZ, d_ws2);

	setRealZeroImagCpu(&ws1, YY); ws1.new_host = true; ws1.new_device = false; ws1.fft2DTo(qYY, d_ws2);
	setRealZeroImagCpu(&ws1, YZ); ws1.new_host = true; ws1.new_device = false; ws1.fft2DTo(qYZ, d_ws2);
	setRealZeroImagCpu(&ws1, ZZ); ws1.new_host = true; ws1.new_device = false; ws1.fft2DTo(qZZ, d_ws2);

	//prescaling by 1/xy
	qXX->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
	qXY->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
	qXZ->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
	qYY->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
	qYZ->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
	qZZ->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));

	return true;
}

bool LongRangeCuda::applyToSum(SpinSystem* ss)
{
	updateData();
	const int nxy = nx*ny;
	
	doubleComplex* d_ws1;
	doubleComplex* d_ws2;
	doubleComplex* d_ws3;
	
	getWSMem3(&d_ws1, sizeof(doubleComplex)*nxyz, 
			  &d_ws2, sizeof(doubleComplex)*nxyz,
			  &d_ws3, sizeof(doubleComplex)*nxyz);
	

	dcArray ws(nx,ny,nz, d_ws1);
	dcArray ws2(nx,ny,nz, d_ws2);
	dcArray ws3(nx,ny,nz, d_ws3);
	 dArray ws4(nx,ny,nz, (double*) d_ws3); //note: ws4 and ws3 share memory
	
	ss->fft();
	dcArray* sqx = ss->qx;
	dcArray* sqy = ss->qy;
	dcArray* sqz = ss->qz;

	doubleComplex one = luaT<doubleComplex>::one();
	
	// HX
	ws.zero();
	for(int targetLayer=0; targetLayer<nz; targetLayer++)
	for(int sourceLayer=0; sourceLayer<nz; sourceLayer++)
	{
		int offset = sourceLayer - targetLayer;
		doubleComplex sign = luaT<doubleComplex>::one();
		if(offset < 0)
		{
			offset = -offset;
			sign = luaT<doubleComplex>::neg_one();
		}
	
		arrayLayerMult(ws.ddata(), targetLayer, qXX->ddata(), offset, sqx->ddata(), sourceLayer, one, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws.ddata(), targetLayer, qXY->ddata(), offset, sqy->ddata(), sourceLayer, one, 0, nxy); 
		arrayLayerMult(ws.ddata(), targetLayer, qXZ->ddata(), offset, sqz->ddata(), sourceLayer,sign, 0, nxy); 
	}
	ws.ifft2DTo(&ws2, &ws3); //ifft to ws2 using ws3 as a work space
	arrayGetRealPart(ws4.ddata(),  ws2.ddata(), nxyz);
	ws4.scaleAll(g * global_scale);
	(*ss->hx[SUM_SLOT]) += ws4;
	
	// HY
	ws.zero();
	for(int targetLayer=0; targetLayer<nz; targetLayer++)
	for(int sourceLayer=0; sourceLayer<nz; sourceLayer++)
	{
		int offset = sourceLayer - targetLayer;
		doubleComplex sign = luaT<doubleComplex>::one();
		if(offset < 0)
		{
			offset = -offset;
			sign = luaT<doubleComplex>::neg_one();
		}
		
		arrayLayerMult(ws.ddata(), targetLayer, qXY->ddata(), offset, sqx->ddata(), sourceLayer, one, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws.ddata(), targetLayer, qYY->ddata(), offset, sqy->ddata(), sourceLayer, one, 0, nxy); 
		arrayLayerMult(ws.ddata(), targetLayer, qYZ->ddata(), offset, sqz->ddata(), sourceLayer,sign, 0, nxy); 
	}
	ws.ifft2DTo(&ws2, &ws3); //ifft to ws2 using ws3 as a work space
	arrayGetRealPart(ws4.ddata(),  ws2.ddata(), nxyz);
	ws4.scaleAll(g * global_scale);
	(*ss->hy[SUM_SLOT]) += ws4;

	// HZ
	ws.zero();
	for(int targetLayer=0; targetLayer<nz; targetLayer++)
	for(int sourceLayer=0; sourceLayer<nz; sourceLayer++)
	{
		int offset = sourceLayer - targetLayer;
		doubleComplex sign = luaT<doubleComplex>::one();
		if(offset < 0)
		{
			offset = -offset;
			sign = luaT<doubleComplex>::neg_one();
		}
	
		arrayLayerMult(ws.ddata(), targetLayer, qXZ->ddata(), offset, sqx->ddata(), sourceLayer,sign, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws.ddata(), targetLayer, qYZ->ddata(), offset, sqy->ddata(), sourceLayer,sign, 0, nxy); 
		arrayLayerMult(ws.ddata(), targetLayer, qZZ->ddata(), offset, sqz->ddata(), sourceLayer, one, 0, nxy); 
	}
	ws.ifft2DTo(&ws2, &ws3); //ifft to ws2 using ws3 as a work space
	arrayGetRealPart(ws4.ddata(),  ws2.ddata(), nxyz);
	ws4.scaleAll(g * global_scale);
	(*ss->hz[SUM_SLOT]) += ws4;
	
	
	ss->slot_used[SUM_SLOT] = true;
	return true;
}

bool LongRangeCuda::apply(SpinSystem* ss)
{
	markSlotUsed(ss);
	updateData();
	const int nxy = nx*ny;

	doubleComplex one = luaT<doubleComplex>::one();
	doubleComplex* d_ws1;
	doubleComplex* d_ws2;
	doubleComplex* d_ws3;
	
	getWSMem3(&d_ws1, sizeof(doubleComplex)*nxyz, 
			  &d_ws2, sizeof(doubleComplex)*nxyz,
			  &d_ws3, sizeof(doubleComplex)*nxyz);

	dcArray ws(nx,ny,nz, d_ws1);
	dcArray ws2(nx,ny,nz, d_ws2);
	dcArray ws3(nx,ny,nz, d_ws3);
	
	ss->fft();
	dcArray* sqx = ss->qx;
	dcArray* sqy = ss->qy;
	dcArray* sqz = ss->qz;

	dArray* hx = ss->hx[slot];
	dArray* hy = ss->hy[slot];
	dArray* hz = ss->hz[slot];
	
	// HX
	ws.zero();
	for(int targetLayer=0; targetLayer<nz; targetLayer++)
	for(int sourceLayer=0; sourceLayer<nz; sourceLayer++)
	{
		int offset = sourceLayer - targetLayer;
		doubleComplex sign = luaT<doubleComplex>::one();
		if(offset < 0)
		{
			offset = -offset;
			sign = luaT<doubleComplex>::neg_one();
		}
	
		arrayLayerMult(ws.ddata(), targetLayer, qXX->ddata(), offset, sqx->ddata(), sourceLayer, one, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws.ddata(), targetLayer, qXY->ddata(), offset, sqy->ddata(), sourceLayer, one, 0, nxy); 
		arrayLayerMult(ws.ddata(), targetLayer, qXZ->ddata(), offset, sqz->ddata(), sourceLayer,sign, 0, nxy); 
	}
	ws.ifft2DTo(&ws2, &ws3); //ifft to ws2 using ws3 as a work space
	arrayGetRealPart(hx->ddata(),  ws2.ddata(), nxyz);
	
	// HY
	ws.zero();
	for(int targetLayer=0; targetLayer<nz; targetLayer++)
	for(int sourceLayer=0; sourceLayer<nz; sourceLayer++)
	{
		int offset = sourceLayer - targetLayer;
		doubleComplex sign = luaT<doubleComplex>::one();
		if(offset < 0)
		{
			offset = -offset;
			sign = luaT<doubleComplex>::neg_one();
		}
	
		arrayLayerMult(ws.ddata(), targetLayer, qXY->ddata(), offset, sqx->ddata(), sourceLayer, one, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws.ddata(), targetLayer, qYY->ddata(), offset, sqy->ddata(), sourceLayer, one, 0, nxy); 
		arrayLayerMult(ws.ddata(), targetLayer, qYZ->ddata(), offset, sqz->ddata(), sourceLayer,sign, 0, nxy); 
	}
	ws.ifft2DTo(&ws2, &ws3); //ifft to ws2 using ws3 as a work space
	arrayGetRealPart(hy->ddata(),  ws2.ddata(), nxyz);

	// HZ
	ws.zero();
	for(int targetLayer=0; targetLayer<nz; targetLayer++)
	for(int sourceLayer=0; sourceLayer<nz; sourceLayer++)
	{
		int offset = sourceLayer - targetLayer;
		doubleComplex sign = luaT<doubleComplex>::one();
		if(offset < 0)
		{
			offset = -offset;
			sign = luaT<doubleComplex>::neg_one();
		}
	
		arrayLayerMult(ws.ddata(), targetLayer, qXZ->ddata(), offset, sqx->ddata(), sourceLayer,sign, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws.ddata(), targetLayer, qYZ->ddata(), offset, sqy->ddata(), sourceLayer,sign, 0, nxy); 
		arrayLayerMult(ws.ddata(), targetLayer, qZZ->ddata(), offset, sqz->ddata(), sourceLayer, one, 0, nxy); 
	}
	ws.ifft2DTo(&ws2, &ws3); //ifft to ws2 using ws3 as a work space
	arrayGetRealPart(hz->ddata(),  ws2.ddata(), nxyz);
	
	hx->new_host = false;
	hy->new_host = false;
	hz->new_host = false;
	hx->new_device = true;
	hy->new_device = true;
	hz->new_device = true;
	
	
/*	printf("%f\n", g );
	printf("%f\n", g * global_scale);
	printf("%f\n", global_scale);*/
	
	hx->scaleAll(g * global_scale);
	hy->scaleAll(g * global_scale);
	hz->scaleAll(g * global_scale);
		
	return true;
}











static int l_setstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRangeCuda, lr, 1);
	lr->g = lua_tonumber(L, 2);
	return 0;
}
static int l_getstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRangeCuda, lr, 1);
	lua_pushnumber(L, lr->g);

	return 1;
}

static int l_setunitcell(lua_State* L)
{
	LUA_PREAMBLE(LongRangeCuda, lr, 1);
	
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
	LUA_PREAMBLE(LongRangeCuda, lr, 1);

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
	LUA_PREAMBLE(LongRangeCuda, lr, 1);

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
	LUA_PREAMBLE(LongRangeCuda, lr, 1);

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
	LUA_PREAMBLE(LongRangeCuda, lr, 1);
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
	LUA_PREAMBLE(LongRangeCuda, lr, 1);

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




int LongRangeCuda::help(lua_State* L)
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
const luaL_Reg* LongRangeCuda::luaMethods()
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
LONGRANGECUDA_API int lib_register(lua_State* L);
LONGRANGECUDA_API int lib_version(lua_State* L);
LONGRANGECUDA_API const char* lib_name(lua_State* L);
LONGRANGECUDA_API int lib_main(lua_State* L);
}

int lib_register(lua_State* L)
{
	luaT_register<LongRangeCuda>(L);
	return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LongRange-Cuda";
#else
	return "LongRange-Cuda-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}


