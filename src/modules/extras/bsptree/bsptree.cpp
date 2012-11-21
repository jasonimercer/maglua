/******************************************************************************
* Copyright (C) 2008-2012 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "bsptree.h"
#include "luamigrate.h"

#ifndef WIN32
#include <strings.h>
#endif


BSPTree::BSPTree()
	: LuaBaseObject(hash32("BSPTree"))
{
	c[0] = 0;
	c[1] = 0;
	split_dir = 0;
	for(int i=0; i<3; i++)
	{
		p1[i] = 0;
		p2[i] = 1;
	}
	parent = 0;
	list = 0;
    init();
}

void BSPTree::push(lua_State* L)
{
	luaT_push<BSPTree>(L, this);
}

int BSPTree::luaInit(lua_State* L)
{
	deinit();
	for(int i=0; i<3; i++)
	{
		if(lua_isnumber(L, i+1))
			p1[i] = lua_tonumber(L, i+1);
		if(lua_isnumber(L, i+4))
			p2[i] = lua_tonumber(L, i+4);
	}

	init();
	return LuaBaseObject::luaInit(L);
}

int BSPTree::addData(const double x, const double y,const double z, const int ref)
{
	if(list == 0)
		list = new BSPDataList;

	list->data.push_back(new BSPData(x,y,z, ref));
	return ((int)list->data.size()) - 1;
}
int BSPTree::addIndex(const int i)
{
	idx.push_back(i);
	return idx.size();
}

void BSPTree::split(int max_at_leaf)
{
	if(c[0]) return; //already split

	//printf("data here: %i\n", idx.size());

	for(int i=0; i<2; i++)
	{
		c[i] = new BSPTree;
		c[i]->parent = this;
		c[i]->split_dir = (split_dir + 1) % 3;
		memcpy(c[i]->p1, p1, sizeof(double) * 3);
		memcpy(c[i]->p2, p2, sizeof(double) * 3);
		c[i]->list = list;
		c[i]->L = L;
	}
	const double mid = (p1[split_dir] + p2[split_dir]) * 0.5;
	c[0]->p2[split_dir] = mid;
	c[1]->p1[split_dir] = mid;
	
	
	for(unsigned int i=0; i<idx.size(); i++)
	{
		double p3[3];
		if(list->getXYZv(idx[i], p3))
		{
			if(p3[split_dir] < mid)
			{
// 				printf("Adding %i to c[0]\n", idx[i]);
				c[0]->idx.push_back(idx[i]);
			}
			else
			{
// 				printf("Adding %i to c[1]\n", idx[i]);
				c[1]->idx.push_back(idx[i]);
			}
		}
	}

	for(int i=0; i<2; i++)
		if(c[i]->idx.size() > max_at_leaf)
			c[i]->split(max_at_leaf);
}

static bool aabbPointTest(const double* p1, const double* p2, const double* test3)
{
// 	printf("aabbPT  %g,%g,%g     %g,%g,%g      %g,%g,%g\n", p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], test3[0], test3[1], test3[2]);
	for(int i=0; i<3; i++)
	{
		if(test3[i] < p1[i])
			return false;
		if(test3[i] >= p2[i])
			return false;
	}
	return true;
}

static bool aabbSphIsect(const double* p1, const double* p2, const double cx, const double cy, const double cz, const double r)
{
	double pp1[3];
	double pp2[3];
	
	for(int i=0; i<3; i++)
	{
		pp1[i] = p1[i] - r;
		pp2[i] = p2[i] + r;
	}

	double cc[3];
	cc[0] = cx;
	cc[1] = cy;
	cc[2] = cz;

	return aabbPointTest(pp1, pp2, cc);
}

int BSPTree::getDataInSphere(const double cx, const double cy, const double cz, const double r, int& next_id)
{
	int m = 0;
	if(c[0])
	{
		for(int i=0; i<2; i++)
		{
// 			printf("Test Point: %g,%g,%g     %g\n", cx,cy,cz,r);
// 			printf("Child Box (%g, %g, %g) (%g, %g, %g)\n", c[i]->p1[0], c[i]->p1[1], c[i]->p1[2], c[i]->p2[0], c[i]->p2[1], c[i]->p2[2]);
			if(aabbSphIsect(c[i]->p1, c[i]->p2, cx, cy, cz, r))
				m += c[i]->getDataInSphere(cx, cy, cz, r, next_id);
		}
		return m;
	}
	
// 	printf("%i\n", idx.size());
	for(unsigned int i=0; i<idx.size(); i++)
	{
		int j = idx[i];
// 		printf("Checking %i\n", j);
		BSPData* d = list->data[j];
		
		const double dx = d->p[0] - cx;
		const double dy = d->p[1] - cy;
		const double dz = d->p[2] - cz;
		
		if(dx*dx + dy*dy + dz*dz <= r*r)
		{
			//printf("L %p    %s\n", L, __FUNCTION__);
			lua_pushinteger(L, next_id);
			lua_rawgeti(L, LUA_REGISTRYINDEX, d->ref);
			lua_settable(L, -3);
			next_id++;
			m++;
		}
	}
	return m;
}

	




BSPTree::~BSPTree()
{
	deinit();
}

void BSPTree::deinit()
{
	if(c[0]) delete c[0];
	if(c[1]) delete c[1];
	c[0] = 0;
	c[1] = 0;
	
	if(list && parent == 0)
	{
		for(int i=0; i<(int)list->data.size(); i++)
		{
			lua_unref(L, list->data[i]->ref);
			delete list->data[i];
		}
		delete list;
		list = 0;
	}
	split_dir = 0;
	idx.clear();
}


void BSPTree::init()
{
}


void BSPTree::encode(buffer* b)
{
	for(int i=0; i<3; i++) //aabb
	{
		encodeDouble(p1[i], b);
		encodeDouble(p2[i], b);
	}
	encodeInteger(split_dir, b);
	encodeInteger((int)idx.size(), b);
	for(int i=0; i<(int)idx.size(); i++)
	{
		encodeInteger(idx[i], b);
	}
	
	
	if(parent == 0) //root, encode the lua data
	{
		encodeInteger(1, b); //1 means root
		if(list)
		{
			int s = (int)list->data.size();
			encodeInteger(s, b);
			for(int i=0; i<s; i++)
			{
				encodeDouble(list->data[i]->p[0], b);
				encodeDouble(list->data[i]->p[1], b);
				encodeDouble(list->data[i]->p[2], b);
				
				lua_rawgeti(L, LUA_REGISTRYINDEX, list->data[i]->ref);
				_exportLuaVariable(L, -1, b);
				lua_pop(L, 1);
			}
		}
		else
		{
			encodeInteger(0, b); //0 means empty list
		}
	}
	else //not root
	{
		encodeInteger(0, b); // not root
	}
	
	if(c[0])
	{
		encodeInteger(1, b); //has children
		c[0]->encode(b);
		c[1]->encode(b);
	}
	else
	{
		encodeInteger(0, b); // no children
	}
}

void BSPTree::setList(BSPDataList* l)
{
	list = l;
	if(c[0])
	{
		c[0]->setList(l);
		c[1]->setList(l);
	}
}

	
int  BSPTree::decode(buffer* b)
{
	deinit();

	for(int i=0; i<3; i++) //aabb
	{
		p1[i] = decodeDouble(b);
		p2[i] = decodeDouble(b);
	}
	split_dir = decodeInteger(b);

	int s = decodeInteger(b);
	for(int i=0; i<s; i++)
	{
		idx.push_back(decodeInteger(b));
	}

	int is_root = decodeInteger(b);
	
	if(is_root == 1) //root, decode the lua data
	{
		int lst_sz = decodeInteger(b);
		if(lst_sz)
		{
			list = new BSPDataList;
			int num_data = decodeInteger(b);
			for(int i=0; i<num_data; i++)
			{
				double x = decodeDouble(b);
				double y = decodeDouble(b);
				double z = decodeDouble(b);
				
				_importLuaVariable(L, b);
				int ref = luaL_ref(L, LUA_REGISTRYINDEX);
				
				list->data.push_back(new BSPData(x,y,z,ref));
			}
		}
		else
		{
			list = 0;
		}
	}
	else //not root, no need to encode list
	{
	}
	

	int has_children = decodeInteger(b);

	if(has_children)
	{
		for(int i=0; i<2; i++)
		{
			c[i] = new BSPTree;
			c[i]->L = L;
			c[i]->decode(b);
		}
	}
	else
	{
		c[0] = 0;
		c[1] = 0;
	}
	
	if(is_root)
	{
		setList(list);
	}
	
	return 0;
}


static int l_getdatsph(lua_State* L)
{
	LUA_PREAMBLE(BSPTree, b,  1);
	
	double cx = lua_tonumber(L, 2);
	double cy = lua_tonumber(L, 3);
	double cz = lua_tonumber(L, 4);
	double r  = lua_tonumber(L, 5);
	
	lua_newtable(L);
	int idx = 1;
	b->getDataInSphere(cx, cy, cz, r, idx);


	return 1;
}

static int l_insert(lua_State* L)
{
	LUA_PREAMBLE(BSPTree, b,  1);

	const double x = lua_tonumber(L, 2);
	const double y = lua_tonumber(L, 3);
	const double z = lua_tonumber(L, 4);
	
	lua_pushvalue(L, 5);
	int ref = luaL_ref(L, LUA_REGISTRYINDEX);
	
	b->addIndex(
		b->addData(x,y,z,ref)
	   );
	
	return 0;
}


static int l_split(lua_State* L)
{
	LUA_PREAMBLE(BSPTree, b,  1);

	if(lua_isnumber(L, 2))
	{
		int s = lua_tonumber(L, 2);
		if(s < 1)
			s = 1;
		b->split(s);
	}
	else
	{
		b->split();
	}
	return 0;
}


// static int l_invalidatefourierdata(lua_State* L)
// {
// 	LUA_PREAMBLE(BSPTree, s,  1);
// 	s->invalidateFourierData();
// 	return 0;
// }


int BSPTree::help(lua_State* L)
{
	
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Constructs a Binary Space Partition Tree where arbitrary data may be inserted and queried");
		lua_pushstring(L, "6 Numbers: x1,y1,z1,  x2,y2,z2 - The corner coordinates of the bounding box"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);

	if(func == l_insert)
	{
		lua_pushstring(L, "Add data to the BSPTree");
		lua_pushstring(L, "3 Numbers, 1 Data: The numbers are the coordinates and must be inside the bounding box, the data can be any single value including a table");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_split)
	{
		lua_pushstring(L, "Compile the BSPTree, must be done before queried.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getdatsph)
	{
		lua_pushstring(L, "Return all data in the provided sphere.");
		lua_pushstring(L, "4 Nubmers: center of sphere and radius");
		lua_pushstring(L, "1 Table: List of data");
		return 3;
	}
	
// d	if(func == l_invalidatefourierdata)
// 	{
// 		lua_pushstring(L, "Invalidates the cache of the Fourier transform of the spin system. If the time changes or :setSpin "
// 						  "is called then the cache is invalidated but there are cases, such as when the internal arrays are exported "
// 						  "and modified, when the BSPTree isn't aware of changes. This function help to deal with those extreme cases");
// 		lua_pushstring(L, "");
// 		lua_pushstring(L, "");
// 		return 3;
// 	}

	return LuaBaseObject::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* BSPTree::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		{"insert", l_insert},
		{"split", l_split},
		{"getDataInSphere", l_getdatsph},
// 		{"invalidateFourierData", l_invalidatefourierdata},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}












#include "info.h"
extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
BSPTREE_API int lib_register(lua_State* L);
BSPTREE_API int lib_version(lua_State* L);
BSPTREE_API const char* lib_name(lua_State* L);
BSPTREE_API int lib_main(lua_State* L);
}

#include <stdio.h>
BSPTREE_API int lib_register(lua_State* L)
{
	luaT_register<BSPTree>(L);
	return 0;
}

BSPTREE_API int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "BSPTree";
#else
	return "BSPTree-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}
