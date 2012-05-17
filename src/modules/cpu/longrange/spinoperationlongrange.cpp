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
	ws1 = 0;
	g = 1;
	gmax = 2000;
	matrixLoaded = false;
	newdata = false;
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


void LongRange::encode(buffer* b)
{
	SpinOperation::encode(b);
	encodeInteger(gmax, b);
	encodeDouble(g, b);
	for(int i=0; i<9; i++)
	{
		encodeDouble(ABC[i], b);
	}
}

int  LongRange::decode(buffer* b)
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



void LongRange::init()
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

	ws1 = new dcArray(nx,ny,nz);
	ws2 = new dcArray(nx,ny,nz);
	
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
		return (*AB) [offset]; \
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

void LongRange::deinit()
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
		ws1 = 0;
	}
	hasMatrices = false;
}

LongRange::~LongRange()
{
	deinit();
}

bool LongRange::updateData()
{
	if(!matrixLoaded)
	{
		loadMatrix();
	}
	
	if(!newdata)
		return true;
	newdata = false;
	
	ws1->zero();
	
	arraySetRealPart(ws1->data(), XX->data(), nxyz);  ws1->fft2DTo(qXX);
	arraySetRealPart(ws1->data(), XY->data(), nxyz);  ws1->fft2DTo(qXY);
	arraySetRealPart(ws1->data(), XZ->data(), nxyz);  ws1->fft2DTo(qXZ);

	arraySetRealPart(ws1->data(), YY->data(), nxyz);  ws1->fft2DTo(qYY);
	arraySetRealPart(ws1->data(), YZ->data(), nxyz);  ws1->fft2DTo(qYZ);
	arraySetRealPart(ws1->data(), ZZ->data(), nxyz);  ws1->fft2DTo(qZZ);

	//prescaling by 1/xy
	qXX->scaleAll(doubleComplex(1.0/((double)(nx*ny)), 0));
	qXY->scaleAll(doubleComplex(1.0/((double)(nx*ny)), 0));
	qXZ->scaleAll(doubleComplex(1.0/((double)(nx*ny)), 0));
	qYY->scaleAll(doubleComplex(1.0/((double)(nx*ny)), 0));
	qYZ->scaleAll(doubleComplex(1.0/((double)(nx*ny)), 0));
	qZZ->scaleAll(doubleComplex(1.0/((double)(nx*ny)), 0));


	return true;
}


void LongRange::getMatrices()
{
	init();
	
	loadMatrix();
	
	dArray* arrs[6];
	arrs[0] = XX;
	arrs[1] = XY;
	arrs[2] = XZ;
	arrs[3] = YY;
	arrs[4] = YZ;
	arrs[5] = ZZ;
	
	dcArray* qarrs[6];
	qarrs[0] = qXX;
	qarrs[1] = qXY;
	qarrs[2] = qXZ;
	qarrs[3] = qYY;
	qarrs[4] = qYZ;
	qarrs[5] = qZZ;
	
	for(int a=0; a<6; a++)
	{
		ws1->zero();
		arraySetRealPart(ws1->data(), arrs[a]->data(), nxyz);
		ws1->fft2DTo(qarrs[a]);
	}

	newdata = false;
	hasMatrices = true;
}




bool LongRange::apply(SpinSystem* ss)
{
	markSlotUsed(ss);
	updateData();
	const int nxy = nx*ny;

	doubleComplex one = luaT<doubleComplex>::one();

	
	ss->fft();
	dcArray* sqx = ss->qx;
	dcArray* sqy = ss->qy;
	dcArray* sqz = ss->qz;

	dArray* hx = ss->hx[slot];
	dArray* hy = ss->hy[slot];
	dArray* hz = ss->hz[slot];
	
	// HX
	ws1->zero();
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
	
		arrayLayerMult(ws1->data(), targetLayer, qXX->data(), offset, sqx->data(), sourceLayer, one, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws1->data(), targetLayer, qXY->data(), offset, sqy->data(), sourceLayer, one, 0, nxy); 
		arrayLayerMult(ws1->data(), targetLayer, qXZ->data(), offset, sqz->data(), sourceLayer,sign, 0, nxy); 
	}
	ws1->ifft2DTo(ws2);
	arrayGetRealPart(hx->data(),  ws2->data(), nxyz);
	
	// HY
	ws1->zero();
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
	
		arrayLayerMult(ws1->data(), targetLayer, qXY->data(), offset, sqx->data(), sourceLayer, one, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws1->data(), targetLayer, qYY->data(), offset, sqy->data(), sourceLayer, one, 0, nxy); 
		arrayLayerMult(ws1->data(), targetLayer, qYZ->data(), offset, sqz->data(), sourceLayer,sign, 0, nxy); 
	}
	ws1->ifft2DTo(ws2);
	arrayGetRealPart(hy->data(),  ws2->data(), nxyz);

	// HZ
	ws1->zero();
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
	
		arrayLayerMult(ws1->data(), targetLayer, qXZ->data(), offset, sqx->data(), sourceLayer,sign, 0, nxy); //0 = sum (1 would be set)
		arrayLayerMult(ws1->data(), targetLayer, qYZ->data(), offset, sqy->data(), sourceLayer,sign, 0, nxy); 
		arrayLayerMult(ws1->data(), targetLayer, qZZ->data(), offset, sqz->data(), sourceLayer, one, 0, nxy); 
	}
	ws1->ifft2DTo(ws2);
	arrayGetRealPart(hz->data(),  ws2->data(), nxyz);
	
	hx->scaleAll(g * global_scale);
	hy->scaleAll(g * global_scale);
	hz->scaleAll(g * global_scale);
		
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

static int l_getarray(lua_State* L)
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



