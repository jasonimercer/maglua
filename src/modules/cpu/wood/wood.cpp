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

#include <strings.h>
#include "wood.h"
#include "spinsystem.h"
#include "Vector_stuff.h"
#include "wood_calculations.h"


#define W_MIN_CLOSE 0
#define W_MIN_FAR   1
#define W_MAX       2
	
#ifndef M_PI
#define M_PI 3.141592565358979
#endif
	

// create a unique identifier for the Wood object based on 
// a hash of the string "Wood"
#define ENCODE_WOOD (hash32("Wood"))

Wood::Wood()
 : SpinOperation("Wood", SUM_SLOT, nx, ny, nz, hash32("Wood"))
{
	DN = 0;
	grain_size = 0;
	energyBarriers = 0;
}


int Wood::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	init();
	return 0;
}

void Wood::init()
{
	DN = new double[nz];
	grain_size = new double[nz];
	energyBarriers = new dArray(nx,ny,nz);
}

void Wood::deinit()
{
	delete [] DN;
	delete [] grain_size;
	delete energyBarriers;
	
	DN = 0;
}

void Wood::encode(buffer* b)
{
	SpinOperation::encode(b); //nx,ny,nz,global_scale
	char version = 0;
	encodeChar(version, b);
	for(int i=0; i<nz; i++)
	{
		encodeDouble(DN[i], b);
		encodeDouble(grain_size[i], b);
	}
	energyBarriers->encode(b);
}

int  Wood::decode(buffer* b)
{
	deinit();
	SpinOperation::decode(b); //nx,ny,nz,global_scale
	char version = decodeChar(b);
	if(version == 0)
	{
		init();
		for(int i=0; i<nz; i++)
		{
			DN[i] = decodeDouble(b);
			grain_size[i] = decodeDouble(b);
		}

		energyBarriers->decode(b);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}

	return 0;
}

void Wood::adjustMagnetostatics(Magnetostatics2D* mag)
{
	mag->makeNewData(); //no guarantee that the data is already calculated before this point
	
	for(int i=0; i<nz; i++)
	{
		grain_size[i] = mag->getGrainSize(i);
	}
	
	for(int i=0; i<nz; i++)
	{
		dArray* XX = mag->XX[i][i];
		dArray* ZZ = mag->ZZ[i][i];
		double xx = XX->get(0);
		double zz = ZZ->get(0);
		DN[i] = 4.0 * M_PI * (zz-xx);
		
		mag->XX[i][i]->set(0,0,0,  0);
		mag->YY[i][i]->set(0,0,0,  0);
		mag->ZZ[i][i]->set(0,0,0,  0);
	}
}
	

void Wood::calcAllEnergyBarrier(SpinSystem* ss_src, Anisotropy* ani, Magnetostatics2D* mag)
{
	energyBarriers->zero();
	int sitex, sitey, sitez;

	for(int i=0; i<ani->num; i++)
	{
		const int site = ani->ops[i].site;
		ss_src->idx2xyz(site, sitex, sitey, sitez);
		const double CELL = grain_size[sitez];

		double kx = ani->ops[i].axis[0];
		double ky = ani->ops[i].axis[1];
		double kz = ani->ops[i].axis[2];
		double ks = ani->ops[i].strength/CELL;
		
		// now we are dealing with spin at "site". 
		double mx = (*ss_src->x)[site]/CELL;
		double my = (*ss_src->y)[site]/CELL;
		double mz = (*ss_src->z)[site]/CELL;
		
		// here is the effective field at "site"
		double hx = (*ss_src->hx)[SUM_SLOT][site];
		double hy = (*ss_src->hy)[SUM_SLOT][site];
		double hz = (*ss_src->hz)[SUM_SLOT][site];
		
		Cvctr H(hx, hy, hz);
		Cvctr M(mx, my, mz);
		Cvctr K(kx*ks, ky*ks, kz*ks);
	
		double EDB = DU(H,K,M,DN[sitez]); 
		
		energyBarriers->set(sitex, sitey, sitez, EDB*CELL);
	}
	
}

// index = 0 : Choose the closest minimum orientation
// index = 1 : Choose the farthest minimum orientation
// index = 2 : Choose the lowest maximum

// cell_size is the size of the cells on each layer
bool Wood::apply(SpinSystem* ss_src, Anisotropy* ani,  Magnetostatics2D* mag, SpinSystem* ss_dest, int& updates, int index)
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
		const int site = ani->ops[i].site;
		int sitex, sitey, sitez;
		ss_src->idx2xyz(site, sitex, sitey, sitez);
		const double CELL = grain_size[sitez];

// 		printf("sitez: %i\n", sitez);
		
		double kx = ani->ops[i].axis[0];
		double ky = ani->ops[i].axis[1];
		double kz = ani->ops[i].axis[2];
		double ks = ani->ops[i].strength/CELL;
		
		// now we are dealing with spin at "site". 
		double mx = (*ss_src->x)[site]/CELL;
		double my = (*ss_src->y)[site]/CELL;
		double mz = (*ss_src->z)[site]/CELL;
		
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
		if (index == 0) updates += do_wood_calculation_demag_min_close(H, M, K, Moutput, DN[sitez]);
		if (index == 1) updates += do_wood_calculation_demag_min_far(H, M, K, Moutput, DN[sitez]);
		if (index == 2) updates += do_wood_calculation_demag_max(H, M, K, Moutput, DN[sitez]);

		(*ss_dest->x)[site] = Moutput.x*CELL;
		(*ss_dest->y)[site] = Moutput.y*CELL;
		(*ss_dest->z)[site] = Moutput.z*CELL;
	}
	
	return true;
}



Wood::~Wood()
{
	
}


static int l_adj_mag(lua_State* L)
{
	LUA_PREAMBLE(Wood, wood, 1);
	LUA_PREAMBLE(Magnetostatics2D, mag, 2);
	
	wood->adjustMagnetostatics(mag);
	return 0;
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
	
	LUA_PREAMBLE(Magnetostatics2D, mag2d, 4);
	
	
	
	int offset = 0;
	if(luaT_is<SpinSystem>(L, 5)) //if there is a spin sys at pos 4 (3rd arg) then make it the dest
	{
		ss[1] = luaT_to<SpinSystem>(L, 5);
		offset = 1;
	}

	int index = -1;
	const char* method_name = lua_tostring(L, 5+offset);
	
	if(strcasecmp(method_name, "MIN_CLOSE") == 0)
		index = W_MIN_CLOSE;
	if(strcasecmp(method_name, "MIN_FAR") == 0)
		index = W_MIN_FAR;
	if(strcasecmp(method_name, "MAX") == 0)
		index = W_MAX;
	
	if(index == -1)
		return luaL_error(L, "Method argument of apply should be one of `Min_close', `Min_far' or `Max'");
	
	int updates;
	
	if(!wood->apply(ss[0], ani, mag2d, ss[1], updates, index))
		return luaL_error(L, "Failed to apply Wood operator (system size mismatch?)");
	
	lua_pushinteger(L, updates);
	return 1; //1 piece of information: the update count;
}

static int l_getdemag(lua_State* L)
{
	LUA_PREAMBLE(Wood, wood, 1);
	
	int layer = lua_tointeger(L, 2);
	if(layer < 1 || layer > wood->nz)
		return luaL_error(L, "Invalid layer");
	
	lua_pushnumber(L, wood->DN[layer-1]);
	return 1;
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

	LUA_PREAMBLE(SpinSystem, ss, 2);
	LUA_PREAMBLE(Anisotropy, ani, 3);
	LUA_PREAMBLE(Magnetostatics2D, mag2d, 4);

	wood->calcAllEnergyBarrier(ss, ani, mag2d);

	return 0;
}

static int l_geteb(lua_State* L)
{
	LUA_PREAMBLE(Wood, wood, 1);

	int x = lua_tointeger(L, 2);
	int y = lua_tointeger(L, 3);
	int z = lua_tointeger(L, 4);

// 	printf("%i %i %i\n", x, y, z);
// 	printf("%i %i %i\n", wood->nx, wood->ny,wood->nz);
	
	if(x < 1 || y < 1 || z < 1 || x > wood->nx || y > wood->ny || z > wood->nz)
		return luaL_error(L, "Invalid site");
  
	lua_pushnumber(L, wood->energyBarriers->get(x-1, y-1, z-1));
	return 1;
}
static int l_seteb(lua_State* L)
{
	LUA_PREAMBLE(Wood, wood, 1);

	int x = lua_tointeger(L, 2);
	int y = lua_tointeger(L, 3);
	int z = lua_tointeger(L, 4);

	double v = lua_tonumber(L, 5);

	if(x < 1 || y < 1 || z < 1 || x > wood->nx || y > wood->ny || z > wood->nz)
		return luaL_error(L, "Invalid site");
  
	wood->energyBarriers->set(x-1, y-1, z-1, v);
	return 0;
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
		lua_pushstring(L, "No documentation. Check the source code.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_energyBarrier)
	{
		lua_pushstring(L, "Calculate the energy barriers at each site");
		lua_pushstring(L, "1 *SpinSystem*, 1 *Anisotropy*, 1 *Magnetostatics2D*: Using the spin configuration, fields, anisotropies and geometry calculate the energy barrier to flip at each site. ");
		lua_pushstring(L, "");
		return 3;
	}
	
		
	if(func == l_geteb)
	{
		lua_pushstring(L, "Return the calculated energy barrier at a site");
		lua_pushstring(L, "3 Integers: The x, y and z coordinate (base 1) of the target site.");
		lua_pushstring(L, "1 Number: Energy barrier at (x,y,z)");
		return 3;
	}
	
	if(func == l_adj_mag)
	{
		lua_pushstring(L, "Set local demag factors and remove the self term from the magnetostatic tensor.");
		lua_pushstring(L, "1 *Magnetostatics2D*: Magnetostatics object to modify.");
		lua_pushstring(L, "");
		return 3;
	}
	
		
	if(func == l_getdemag)
	{
		lua_pushstring(L, "Get the demag value at the given layer.");
		lua_pushstring(L, "1 Integer: Layer number (base 1).");
		lua_pushstring(L, "1 Number: Demag factor for the layer.");
		return 3;
	}
	

	if(func == l_apply)
	{
		lua_pushstring(L, "Compute 1 Wood Step.");
		lua_pushstring(L, "1 *SpinSystem*, 1 *Anisotropy*, 1 *Magnetostatics2D*, 1 optional *SpinSystem*, 1 String: " 
		"Calculate 1 Wood iteration from the 1st SpinSystem using the calculated effective fields and the "
		"given Anisotropy operator writing the new state either into the 2nd SpinSystem (if provided) or "
		"back into the 1st SpinSystem. The string defines the method and should be one of `Min_close', `Min_far' or `Max'."
		"The last number or table of numbers is the cell sizes/volumes for each layer.");
		lua_pushstring(L, "1 Integer: Number of sites that were updated");
		return 3;
	}

	return SpinOperation::help(L);
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Wood::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"apply",        l_apply},
		{"calculateEnergyBarriers",l_energyBarrier},
		{"energyBarrier", l_geteb},
		{"setEnergyBarrier", l_seteb},
		{"adjustMagnetostatics", l_adj_mag},
		{"getDemag", l_getdemag},
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


