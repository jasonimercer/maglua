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

#include "wood.h"
#include "spinsystem.h"
#include "Vector_stuff.h"
#include "wood_calculations.h"

// create a unique identifier for the Wood object based on 
// a hash of the string "Wood"
#define ENCODE_WOOD (hash32("Wood"))

Wood::Wood()
 : SpinOperation("Wood", SUM_SLOT, nx, ny, nz, hash32("Wood"))
{
	DN = 0.0;
}


int Wood::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	init();
	return 0;
}

void Wood::push(lua_State* L)
{
	luaT_push<Wood>(L, this);
}

void Wood::init()
{
	DN = 0.0;
}

void Wood::deinit()
{
}

void Wood::encode(buffer* b)
{
	SpinOperation::encode(b); //nx,ny,nz,global_scale
	encodeDouble(DN, b);
}

int  Wood::decode(buffer* b)
{
	deinit();
	SpinOperation::decode(b); //nx,ny,nz,global_scale
	DN = decodeDouble(b);
	return 0;
}

// index = 0 : Choose the closest minimum orientation
// index = 1 : Choose the farthest minimum orientation
// index = 2 : Choose the lowest maximum
bool Wood::apply(SpinSystem* ss_src, Anisotropy* ani, SpinSystem* ss_dest, int& updates, int index)
{
	updates = 0;
	if(ss_src != ss_dest) //then need to make sure they're the same size
	{
		if(ss_src->nx != ss_dest->nx) return false;
		if(ss_src->ny != ss_dest->ny) return false;
		if(ss_src->nz != ss_dest->nz) return false;
	}
	
	// anisotropy isn't a flat list, we will iterate over anisotropy sites,
	// not spin sites. (anisotropy operator doesn't need to be densly/completely
	// populated)
	for(int i=0; i<ani->num; i++)
	{
		double kx = ani->ops[i].axis[0];
		double ky = ani->ops[i].axis[1];
		double kz = ani->ops[i].axis[2];
		double ks = ani->ops[i].strength;
		int site = ani->ops[i].site;

		// now we are dealing with spin at "site". 
		double mx = (*ss_src->x)[site];
		double my = (*ss_src->y)[site];
		double mz = (*ss_src->z)[site];
		
		// here is the effective field at "site"
		double hx = (*ss_src->hx)[SUM_SLOT][site];
		double hy = (*ss_src->hy)[SUM_SLOT][site];
		double hz = (*ss_src->hz)[SUM_SLOT][site];
		
		// make the CVectrs 
		Cvctr H(hx, hy, hz);
		Cvctr M(mx, my, mz);
		Cvctr K(kx*ks,ky*ks,kz*ks); 
		Cvctr Moutput;
		
//		updates += do_wood_calculation(H, M, K, Moutput);
		if (index == 0) updates += do_wood_calculation_demag_min_close(H, M, K, Moutput, DN);
		if (index == 1) updates += do_wood_calculation_demag_min_far(H, M, K, Moutput, DN);
		if (index == 2) updates += do_wood_calculation_demag_max(H, M, K, Moutput, DN);

		(*ss_dest->x)[site] = Moutput.x;
		(*ss_dest->y)[site] = Moutput.y;
		(*ss_dest->z)[site] = Moutput.z;
	}
	
	return true;
}



Wood::~Wood()
{
	
}



static int l_setDemag(lua_State* L)
{
	LUA_PREAMBLE(Wood, wood, 1);
	wood->DN = lua_tonumber(L, 2);
	return 0;
}

static int l_getDemag(lua_State* L)
{
	LUA_PREAMBLE(Wood, wood, 1);
	lua_pushnumber(L, wood->DN);
	return 1;
}


static int l_apply(lua_State* L)
{
	LUA_PREAMBLE(Wood, wood, 1);
	
	SpinSystem* ss[2]; //src, dest
	
	ss[0] = luaT_to<SpinSystem>(L, 2);
	if(!ss[0])
		return 0;
	ss[1] = ss[0];

	LUA_PREAMBLE(Anisotropy, ani, 3);
	
	if(luaT_is<SpinSystem>(L, 4)) //if there is a spin sys at pos 4 (3rd arg) then make it the dest
	{
		ss[1] = luaT_to<SpinSystem>(L, 4);
	}

	int index = lua_tonumber(L, 5);
	int updates;

	if(!wood->apply(ss[0], ani, ss[1], updates,index))
		return luaL_error(L, "Failed to apply Wood operator (system size mismatch?)");
	
	lua_pushinteger(L, updates);
	return 1; //1 piece of information: the update count;
}

static int l_effani(lua_State* L)
{
  double _K[3];
  double _M[3];

  int r1 = lua_getNdouble(L, 3, _K, 1, 0);
  int r2 = lua_getNdouble(L, 3, _M, 1+r1, 0);

  double XX = lua_tonumber(L, 1+r1+r2);
  double YY = lua_tonumber(L, 1+r1+r2+1);
  double ZZ = lua_tonumber(L, 1+r1+r2+2);
  

  Cvctr M(_M[0], _M[1], _M[2]);
  Cvctr K(_K[0], _K[1], _K[2]); 

  EffAni2(K,M,XX,YY,ZZ); 

  lua_pushnumber(L, K.x);
  lua_pushnumber(L, K.y);
  lua_pushnumber(L, K.z);
  lua_pushnumber(L, K.mag);
  return 4;
}

static int l_energyBarrier(lua_State* L)
{
  LUA_PREAMBLE(Wood, wood, 1);
  //Wood* wood = checkWood(L,1);
  double _H[3];
  double _K[3];
  double _M[3];
  double EDB;

  int r1 = lua_getNdouble(L, 3, _H, 2, 0);
  int r2 = lua_getNdouble(L, 3, _K, 2+r1, 0);
  int r3 = lua_getNdouble(L, 3, _M, 2+r1+r2,0);
  
  Cvctr H(_H[0], _H[1], _H[2]);
  Cvctr M(_M[0], _M[1], _M[2]);
  Cvctr K(_K[0], _K[1], _K[2]); 

  EDB = DU(H,K,M,wood->DN); 

  lua_pushnumber(L, EDB);
  return 1;
}

// this function helps build the help page
int Wood::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Wood updates a *SpinSystem* using a form of the Woods equation.");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_isfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_effani)
	{
		lua_pushstring(L, "CXYAHSDFRDHFG!!!reate a new Wood object.");
		lua_pushstring(L, "INPUT");
		lua_pushstring(L, "OUTPUT");
		return 3;
	}
	
	if(func == l_energyBarrier)
	{
		lua_pushstring(L, "Energy Density Barrier!!!reate a new Wood object.");
		lua_pushstring(L, "INPUT");
		lua_pushstring(L, "OUTPUT");
		return 3;
	}

	if(func == l_apply)
	{
		lua_pushstring(L, "Compute 1 Wood Step.");
		lua_pushstring(L, "1 *SpinSystem*, 1 *Anisotropy*, 1 optional *SpinSystem*: " 
		"Calculate 1 Wood iteration from the 1st SpinSystem using the calculated effective fields and the "
		"given Anisotropy operator writing the new state either into the 2nd SpinSystem (if provided) or "
		"back into the 1st SpinSystem");
		lua_pushstring(L, "1 Integer: Number of sites that were updated");
		return 3;
	}
	
	if(func == l_setDemag)
	{
		lua_pushstring(L, "Set the demagnetizing factor");
		lua_pushstring(L, "1 Number: The parameter DN"); //input
		lua_pushstring(L, ""); //output
		return 3;
	}
	
	if(func == l_getDemag)
	{
		lua_pushstring(L, "Get the demagnetizing factor");
		lua_pushstring(L, ""); //input
		lua_pushstring(L, "1 Number: The parameter DN"); //output
		return 3;
	}

	return 0;
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Wood::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"apply",        l_apply},
		{"setDemag", l_setDemag},
		{"getDemag", l_getDemag},
		{"energyB",l_energyBarrier},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



#include "info.h"
extern "C"
{
WOOD_API int lib_register(lua_State* L);
WOOD_API int lib_version(lua_State* L);
WOOD_API const char* lib_name(lua_State* L);
WOOD_API int lib_main(lua_State* L);
}

WOOD_API int lib_register(lua_State* L)
{
	luaT_register<Wood>(L);

	lua_getglobal(L, "Wood");
	lua_pushstring(L, "effectiveAnisotropy");
	lua_pushcfunction(L, l_effani);
	lua_settable(L, -3);
	lua_pop(L, 1);
	
	return 0;
}

WOOD_API int lib_version(lua_State* L)
{
	return __revi;
}

WOOD_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Wood";
#else
	return "Wood-Debug";
#endif
}

WOOD_API int lib_main(lua_State* L)
{
	return 0;
}


