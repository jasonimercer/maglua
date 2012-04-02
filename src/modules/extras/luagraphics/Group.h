
class Group;
class GroupNode;

#ifndef GROUP_H_
#define GROUP_H_

#include "Volume.h"
#include "Vector.h"
#include "AABB.h"
#include <list>
using namespace std;

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}
#include "luabaseobject.h"

#include "libLuaGraphics.h"
class LUAGRAPHICS_API GroupNode
{
public:
	GroupNode(Volume* _v);
	GroupNode(GroupNode* a, GroupNode* b);
	~GroupNode();
	AABB bb;
	
	Volume* v;
	int size;
	
	// child nodes
	GroupNode* n1;
	GroupNode* n2;
};

class LUAGRAPHICS_API Group : public LuaBaseObject
{
public:
	Group();
	~Group();
		
	LINEAGE1("Group")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	
	void add(Volume* v);
	void compile();
	
	bool compiled;
	
	GroupNode* root;
	list<GroupNode*> precompiled;
};


#endif

