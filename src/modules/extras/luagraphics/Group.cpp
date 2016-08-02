#include "Atom.h"
#include "Tube.h"

#include "Group.h"
#include <math.h>
#include <string.h>

using namespace std;

GroupNode::GroupNode(Volume* _v)
{
	n1 = n2 = 0;
	v = _v;
	bb = *v->getBB();
	size = 1;
}

GroupNode::GroupNode(GroupNode* A, GroupNode* B)
{
	v = 0;
	size = A->size + B->size;
	bb.include(A->bb);
	bb.include(B->bb);
	n1 = A;
	n2 = B;
}


GroupNode::~GroupNode()
{
	if(n1) delete n1;
	if(n2) delete n2;
	
	if(v)
	{
		v->refcount--;
		if(v->refcount <= 0)
			delete v;
	}
}




Group::Group()
	: LuaBaseObject(hash32("Group"))
{
	compiled = false;
	root = 0;
}

int Group::luaInit(lua_State* L)
{
	return 0;
}

void Group::push(lua_State* L)
{
	luaT_push<Group>(L, this);
}

Group::~Group()
{
	if(root)
		delete root;
}

void Group::add(Volume* v)
{
	if(!v) return;
	precompiled.push_back(new GroupNode(v));
}


void Group::compile()
{
	//find nodes that are close together and form supernodes
	while(precompiled.size() > 1)
	{
		GroupNode* a; //smallest node
		GroupNode* b; //node that makes the smallest combined bb
#if 1
		list<GroupNode*>::iterator it;
		
		a = *precompiled.begin();

		double av = a->bb.volume();
		
		for(it=precompiled.begin(); it!=precompiled.end(); it++)
		{
			const double itv = (*it)->bb.volume();
			if(itv < av)
			{
				a = *it;
				av = itv;
			}
		}
		
		double bbv = -1; //sentinel
		
		// now have a smallest bounding box
		for(it=precompiled.begin(); it!=precompiled.end(); it++)
		{
			if(*it == a)
				continue;
			
			// calc combined volume
			AABB box;
			box.include(a->bb);
			box.include((*it)->bb);
			
			if(bbv < 0 || box.volume() < bbv)
			{
				bbv = box.volume();
				b = *it;
			}
		}
#else
		list<GroupNode*>::iterator it = precompiled.begin();

		a = *it;
		it++;
		b = *it;
#endif
		precompiled.remove(a);
		precompiled.remove(b);
		
// 		GroupNode* parent = new GroupNode(a, b);
// 		parent->pos = 0.5 * (a->pos + b->pos);
// 		parent->bb.include(a->bb);
// 		parent->bb.include(b->bb);
// 		parent->n1 = a;
// 		parent->n2 = b;
// 		parent->size = a->size + b->size;
		
		precompiled.push_back(new GroupNode(a, b));
	}
	
	if(root)
		delete root;
	root = *precompiled.begin();
	precompiled.clear();
	
	compiled = true;
// 	printGN(root);
}








static int l_add(lua_State* L)
{
	LUA_PREAMBLE(Group, c, 1);
	c->add(luaT_to<Volume>(L, 2));
	return 0;
}

static int l_compile(lua_State* L)
{
	LUA_PREAMBLE(Group, c, 1);
	c->compile();
	return 0;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Group::luaMethods()
{
	if(m[127].name)return m;

	//merge_luaL_Reg(m, Sphere::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"add",     l_add},
		{"compile", l_compile},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
