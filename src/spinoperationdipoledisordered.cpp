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

#include "spinoperationdipoledisordered.h"
#include "spinsystem.h"

#include <stdlib.h>
#include <math.h>

DipoleDisordered::DipoleDisordered(int nx, int ny, int nz)
	: SpinOperation("DipoleDisordered", DIPOLE_SLOT, nx, ny, nz, ENCODE_DIPOLE)
{
	
	posx = new double [nxyz];
	posy = new double [nxyz];
	posz = new double [nxyz];
	
	for(int i=0; i<nxyz; i++)
	{
		posx[i] = 0;
		posy[i] = 0;
		posz[i] = 0;
	}
}

void DipoleDisordered::encode(buffer* b) const
{
	
}

int  DipoleDisordered::decode(buffer* b)
{
	return 0;
}

DipoleDisordered::~DipoleDisordered()
{
	delete [] posx;
	delete [] posy;
	delete [] posz;
}

void DipoleDisordered::setPosition(int site, double px, double py, double pz)
{
	if(site >= 0 && site <= nxyz)
	{
		posx[site] = px;
		posy[site] = py;
		posz[site] = pz;
	}
}



bool DipoleDisordered::apply(SpinSystem* ss)
{
	double* x = ss->x;
	double* y = ss->y;
	double* z = ss->z;
	
	double* hx = ss->hx[DIPOLE_SLOT];
	double* hy = ss->hx[DIPOLE_SLOT];
	double* hz = ss->hx[DIPOLE_SLOT];
	
	double r1[3], r2[3];
	double rij[3];
	double lenr;
	
	for(int i=0; i<nxyz; i++)
	{
		hx[i] = 0;
		hy[i] = 0;
		hz[i] = 0;
		r1[0] = posx[i];
		r1[1] = posy[i];
		r1[2] = posz[i];
		
		for(int j=0; j<nxyz; j++)
		{
			if(i != j)
			{
				r2[0] = posx[j];
				r2[1] = posy[j];
				r2[2] = posz[j];
				
				for(int k=0; k<3; k++)
					rij[k] = r2[k] - r1[k];
				
				lenr = sqrt(pow(rij[0], 2) + pow(rij[1], 2) + pow(rij[2], 2));
				
				if(lenr > 0)
				{
					double sr = x[j]*rij[0] + y[j]*rij[1] + z[j]*rij[2];
					hx[i] += g * (x[j] / pow(lenr,3) - 3.0 * rij[0] * sr / pow(lenr,5));
					hy[i] += g * (y[j] / pow(lenr,3) - 3.0 * rij[1] * sr / pow(lenr,5));
					hz[i] += g * (z[j] / pow(lenr,3) - 3.0 * rij[2] * sr / pow(lenr,5));
				}
			}
		}
	}
	
	
	return true;
}







DipoleDisordered* checkDipoleDisordered(lua_State* L, int idx)
{
	DipoleDisordered** pp = (DipoleDisordered**)luaL_checkudata(L, idx, "MERCER.dipoledisordered");
    luaL_argcheck(L, pp != NULL, 1, "`DipoleDisordered' expected");
    return *pp;
}

void lua_pushDipoleDisordered(lua_State* L, DipoleDisordered* dip)
{
	dip->refcount++;
	DipoleDisordered** pp = (DipoleDisordered**)lua_newuserdata(L, sizeof(DipoleDisordered**));
	
	*pp = dip;
	luaL_getmetatable(L, "MERCER.dipoledisordered");
	lua_setmetatable(L, -2);
}

int l_dipdis_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	lua_pushDipoleDisordered(L, new DipoleDisordered(n[0], n[1], n[2]));
	return 1;
}


int l_dipdis_setstrength(lua_State* L)
{
	DipoleDisordered* dip = checkDipoleDisordered(L, 1);
	if(!dip) return 0;

	dip->g = lua_tonumber(L, 2);
	return 0;
}

int l_dipdis_gc(lua_State* L)
{
	DipoleDisordered* dip = checkDipoleDisordered(L, 1);
	if(!dip) return 0;

	dip->refcount--;
	if(dip->refcount == 0)
		delete dip;
	
	return 0;
}

int l_dipdis_apply(lua_State* L)
{
	DipoleDisordered* dip = checkDipoleDisordered(L, 1);
	if(!dip) return 0;
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!dip->apply(ss))
		return luaL_error(L, dip->errormsg.c_str());
	
	return 0;
}

int l_dipdis_getstrength(lua_State* L)
{
	DipoleDisordered* dip = checkDipoleDisordered(L, 1);
	if(!dip) return 0;

	lua_pushnumber(L, dip->g);

	return 1;
}

int l_dipdis_setsiteposition(lua_State* L)
{
	DipoleDisordered* dip = checkDipoleDisordered(L, 1);
	if(!dip) return 0;
	
	int r1;
	int s[3];
	
	r1 = lua_getNint(L, 3, s, 2, 1);
	if(r1 < 0) 
		return luaL_error(L, "invalid site");
	
	int site = dip->getSite(s[0]-1, s[1]-1, s[2]-1);
	
	double a[3];
	int r2 = lua_getNdouble(L, 3, a, 2+r1, 0);
	if(r2<0)
		return luaL_error(L, "invalid position direction");
	
	
	dip->setPosition(site, a[0], a[1], a[2]);
	
	return 0;
}

int l_dipdis_siteposition(lua_State* L)
{
	DipoleDisordered* dip = checkDipoleDisordered(L, 1);
	if(!dip) return 0;
	
	int r1;
	int s[3];
	
	r1 = lua_getNint(L, 3, s, 2, 1);
	if(r1 < 0) 
		return luaL_error(L, "invalid site");
	
	int site = dip->getSite(s[0]-1, s[1]-1, s[2]-1);
	
	lua_pushnumber(L, dip->posx[site]);
	lua_pushnumber(L, dip->posy[site]);
	lua_pushnumber(L, dip->posz[site]);
	return 3;
}

static int l_dipdis_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.dipoledisordered");
	return 1;
}

static int l_dipdis_help(lua_State* L)
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
	
	if(func == l_dipdis_new)
	{
		lua_pushstring(L, "Create a new DipoleDisordered Operator.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 DipoleDisordered object");
		return 3;
	}
	
	
	if(func == l_dipdis_apply)
	{
		lua_pushstring(L, "Calculate the dipolar field of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This spin system will receive the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dipdis_setstrength)
	{
		lua_pushstring(L, "Set the strength of the Dipolar Field");
		lua_pushstring(L, "1 number: strength of the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dipdis_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Dipolar Field");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: strength of the field");
		return 3;
	}
	
	if(func == l_dipdis_setsiteposition)
	{
		lua_pushstring(L, "Maps the lattice coordinate to a real world position");
		lua_pushstring(L, "2 *3Vector*s: First is lattice site, second is real world position");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dipdis_siteposition)
	{
		lua_pushstring(L, "Lookup the real world position from a lattice site");
		lua_pushstring(L, "1 *3Vector*: Lattice site");
		lua_pushstring(L, "1 *3Vector*: Real world postition");
		return 3;
	}
	
	

	return 0;
}


void registerDipoleDisordered(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_dipdis_gc},
		{"apply",        l_dipdis_apply},
		{"setStrength",  l_dipdis_setstrength},
		{"strength",     l_dipdis_getstrength},
		{"setSitePosition", l_dipdis_setsiteposition},
		{"sitePosition", l_dipdis_siteposition},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.dipoledisordered");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_dipdis_new},
		{"help",                l_dipdis_help},
		{"metatable",           l_dipdis_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "DisorderedDipole", functions);
	lua_pop(L,1);	
}

